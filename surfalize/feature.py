"""
Feature parameters according to ISO 25178-2:2021, clause 5.

Feature parameters characterize a scale-limited surface through its significant topographic features (hills and dales)
rather than through the statistics of the height distribution. The feature characterization process defined by the
standard consists of five stages: selection of the type of texture feature, segmentation, determination of the
significant features, selection of feature attributes and computation of attribute statistics.

This module implements the segmentation by watershed (clause 5.3) together with Wolf pruning (height discrimination,
Table 2) and derives the named feature parameters of clause 5.8:

- ``Spd`` density of peaks
- ``Svd`` density of pits
- ``Spc`` arithmetic mean peak curvature
- ``Svc`` arithmetic mean pit curvature
- ``S5p`` five-point peak height
- ``S5v`` five-point pit depth
- ``S10z`` ten-point height

The segmentation engine is shared between hills and dales (a dale segmentation of the height data is, up to a sign, a
hill segmentation of the inverted data) and is computed once per pruning value and cached.
"""
from dataclasses import dataclass
import heapq

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cache import CachedInstance, cache

# 8-connectivity is used both for the watershed flooding and for the region adjacency, so that saddle points (where
# ridge and course lines cross, ISO 25178-2 3.3.3) are detected across diagonal pixel connections as well.
_CONNECTIVITY = np.ones((3, 3), dtype=bool)
_NEIGHBOUR_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _shifted_slices(shape, dy, dx):
    """
    Returns the pair of slice tuples (destination, source) that align an array with a copy of itself shifted by
    (dy, dx), clipped to the overlapping region. Used to compare every pixel with its neighbours without allocating
    padded arrays.
    """
    ny, nx = shape
    dst = (slice(max(0, dy), ny + min(0, dy)), slice(max(0, dx), nx + min(0, dx)))
    src = (slice(max(0, -dy), ny + min(0, -dy)), slice(max(0, -dx), nx + min(0, -dx)))
    return dst, src

# Number of points averaged for the five-point peak/pit parameters (ISO 25178-2 5.8.6/5.8.7).
_N_FIVE_POINT = 5

# Default Wolf pruning threshold as a percentage of Sz. ISO 25178-2 leaves the value to ISO 25178-3; 5 % is the value
# used in the worked examples of the standard (Figure 20) and the customary default in commercial software.
DEFAULT_PRUNING = 5.0

# By default, motifs touching the border of the evaluation area are excluded from the feature set because they are
# incomplete. This matches the default of commercial software such as MountainsMap.
DEFAULT_EXCLUDE_EDGE = True


@dataclass
class _Segmentation:
    """
    Result of a Wolf-pruned watershed segmentation of a height array into motifs (hills or dales).

    Attributes
    ----------
    extrema : ndarray
        Array of shape (n, 2) holding the (row, column) indices of the critical point (pit for a dale, peak for a hill)
        of each significant motif.
    labels : ndarray
        Watershed labels of the over-segmented surface before pruning, with values 1..k.
    region_of : ndarray
        Lookup array mapping each pre-pruning label to the label of the significant motif it was merged into. Index 0 is
        unused so that the array can be indexed directly with the (1-based) watershed labels.
    """
    extrema: np.ndarray
    labels: np.ndarray
    region_of: np.ndarray

    @property
    def count(self):
        return self.extrema.shape[0]


class FeatureParameters(CachedInstance):
    """
    Computes the ISO 25178-2:2021 feature parameters of a `Surface` by watershed segmentation and Wolf pruning.

    The height data is centred on its mean once on instantiation, so that peak heights and pit depths are referenced to
    the mean plane of the scale-limited surface. The actual segmentation is performed lazily and cached per pruning
    value, so that the seven named feature parameters which share the same pruning value only trigger a single
    segmentation of the hills and a single segmentation of the dales.

    Parameters
    ----------
    surface : Surface
        Surface object on which to compute the feature parameters. Must not contain non-measured or masked points.
    """

    def __init__(self, surface):
        if surface.has_missing_points:
            raise ValueError("Non-measured points must be filled before feature parameters can be computed.") from None
        if surface.has_masked_points:
            raise ValueError("Feature parameters are not supported on masked surfaces. Clear the mask first.") from None
        super().__init__()
        self._surface = surface
        # Reference all heights to the mean plane (ISO 25178-2 3.1.10), so that S5p/S5v are measured from the mean plane.
        self._data = np.ascontiguousarray(surface.data - surface.data.mean(), dtype=np.float64)
        self._step_x = surface.step_x
        self._step_y = surface.step_y
        # Sz of the scale-limited surface, used as the reference for the Wolf pruning threshold.
        self._sz = float(self._data.max() - self._data.min())

    # Segmentation #####################################################################################################

    @staticmethod
    def _watershed(data):
        """
        Segments a height array into catchment basins, one per regional minimum, by steepest-descent flow routing.
        Every pixel is routed to the regional minimum reached by repeatedly stepping to its lowest neighbour, so that
        each basin is the catchment area of one minimum. This over-segments the surface; the spurious motifs are removed
        afterwards by Wolf pruning.

        This is implemented in pure NumPy (vectorized neighbour comparison plus pointer doubling) which keeps it fast
        and, unlike an integer Image Foresting Transform watershed, robust on smooth floating point data where height
        quantization would otherwise merge whole regions across flat plateaus.

        Parameters
        ----------
        data : ndarray
            Height array. Dales are the catchment basins of ``data``; passing ``-data`` yields the hills.

        Returns
        -------
        labels : ndarray
            Integer label array with values 1..n.
        n : int
            Number of basins.
        """
        shape = data.shape
        n_pixels = data.size
        flat_index = np.arange(n_pixels).reshape(shape)
        # For every pixel, find its lowest neighbour (8-connected).
        lowest_value = np.full(shape, np.inf)
        lowest_index = flat_index.copy()
        for dy, dx in _NEIGHBOUR_OFFSETS:
            dst, src = _shifted_slices(shape, dy, dx)
            neighbour = data[src]
            update = neighbour < lowest_value[dst]
            block = lowest_value[dst]
            block[update] = neighbour[update]
            lowest_value[dst] = block
            block = lowest_index[dst]
            block[update] = flat_index[src][update]
            lowest_index[dst] = block
        # A pixel with a strictly lower neighbour drains to it; a pixel without one is a regional minimum (a sink), so it
        # points to itself. Connected sinks form one basin seed. Flat plateaus are over-segmented here but cleaned up by
        # the subsequent Wolf pruning.
        has_lower = lowest_value < data
        pointer = np.where(has_lower.ravel(), lowest_index.ravel(), flat_index.ravel())
        markers, n = ndimage.label(~has_lower, structure=_CONNECTIVITY)
        # Resolve each pixel to its basin's sink by pointer doubling. The pointer chains strictly descend until a sink,
        # so doubling converges in O(log(path length)) iterations.
        pointer_to_sink = pointer
        for _ in range(int(np.ceil(np.log2(n_pixels))) + 1):
            jumped = pointer_to_sink[pointer_to_sink]
            if np.array_equal(jumped, pointer_to_sink):
                break
            pointer_to_sink = jumped
        labels = markers.ravel()[pointer_to_sink].reshape(shape)
        return labels, n

    @staticmethod
    def _region_adjacency(labels, data, n):
        """
        Builds the region adjacency graph of the watershed segmentation. For every pair of adjacent basins the saddle
        (pour point) height is the lowest height at which the two basins connect, i.e. the minimum over their shared
        boundary of the higher of the two adjoining heights.

        Returns
        -------
        a, b, saddle : tuple[ndarray, ndarray, ndarray]
            Arrays describing the unique adjacent label pairs (a < b) and their saddle heights.
        """
        def boundary(la, lb, za, zb):
            differ = la != lb
            la, lb = la[differ], lb[differ]
            pour = np.maximum(za[differ], zb[differ])
            return np.minimum(la, lb), np.maximum(la, lb), pour

        lows, highs, pours = [], [], []
        # Horizontal, vertical and both diagonal neighbour pairs (8-connectivity).
        for la, lb, za, zb in (
            (labels[:, :-1], labels[:, 1:], data[:, :-1], data[:, 1:]),
            (labels[:-1, :], labels[1:, :], data[:-1, :], data[1:, :]),
            (labels[:-1, :-1], labels[1:, 1:], data[:-1, :-1], data[1:, 1:]),
            (labels[:-1, 1:], labels[1:, :-1], data[:-1, 1:], data[1:, :-1]),
        ):
            lo, hi, pour = boundary(la, lb, za, zb)
            lows.append(lo)
            highs.append(hi)
            pours.append(pour)
        lo = np.concatenate(lows)
        hi = np.concatenate(highs)
        pour = np.concatenate(pours)

        if lo.size == 0:
            empty = np.empty(0)
            return empty, empty, empty

        # Reduce to unique (lo, hi) pairs keeping the minimum pour point per pair.
        key = lo.astype(np.int64) * (n + 1) + hi
        order = np.argsort(key, kind='stable')
        key, pour, lo, hi = key[order], pour[order], lo[order], hi[order]
        first = np.empty(key.shape, dtype=bool)
        first[0] = True
        first[1:] = key[1:] != key[:-1]
        group = np.cumsum(first) - 1
        saddle = np.full(group[-1] + 1, np.inf)
        np.minimum.at(saddle, group, pour)
        return lo[first], hi[first], saddle

    @cache
    def _segment(self, invert, pruning, exclude_edge):
        """
        Performs the watershed segmentation and Wolf pruning for either dales or hills.

        Parameters
        ----------
        invert : bool
            If False, the dales (catchment basins of the height data) are segmented. If True, the height data is
            inverted first, so that the hills (catchment basins of the inverted data) are segmented.
        pruning : float
            Wolf pruning threshold as a percentage of Sz. Motifs whose local height/depth is smaller than this
            threshold are merged into a neighbouring motif across their lowest saddle.
        exclude_edge : bool
            If True, motifs whose region touches the border of the evaluation area are discarded, since such features
            are incomplete. This matches the default behaviour of commercial software.

        Returns
        -------
        _Segmentation
        """
        data = -self._data if invert else self._data
        labels, n = self._watershed(data)

        index = np.arange(1, n + 1)
        # Critical value of each basin (the pit, or the peak in inverted coordinates) and its location.
        critical = np.empty(n + 1)
        critical[0] = np.nan
        critical[1:] = ndimage.minimum(data, labels, index=index)
        positions = ndimage.minimum_position(data, labels, index=index)
        critical_loc = {r: positions[r - 1] for r in index}

        a, b, saddle = self._region_adjacency(labels, data, n)
        adjacency = {r: {} for r in index}
        for ai, bi, si in zip(a.astype(np.int64), b.astype(np.int64), saddle):
            ai, bi = int(ai), int(bi)
            adjacency[ai][bi] = si
            adjacency[bi][ai] = si

        threshold = pruning / 100 * self._sz
        # Union-find mapping each pre-pruning label to its surviving motif, maintained so that the full labelling can be
        # reconstructed afterwards for plotting.
        merged_into = {r: r for r in index}
        alive = np.ones(n + 1, dtype=bool)
        alive[0] = False
        version = {r: 0 for r in index}

        # Cached lowest bounding saddle of each motif, as the saddle value and the neighbour reached across it. The
        # cache is maintained incrementally during merging so the heap loop never rescans a motif's full adjacency
        # except for the survivor of a merge (and only when necessary). A motif with no neighbour (e.g. a single basin
        # spanning the whole surface) has saddle +inf and can never be pruned.
        min_saddle = {}
        min_neighbour = {}

        def rescan(r):
            best_neighbour, best_saddle = -1, np.inf
            for neighbour, saddle_height in adjacency[r].items():
                if saddle_height < best_saddle:
                    best_neighbour, best_saddle = neighbour, saddle_height
            min_saddle[r] = best_saddle
            min_neighbour[r] = best_neighbour

        for r in index:
            rescan(r)

        # Wolf local height/depth: vertical distance from the critical point up to the lowest bounding saddle.
        heap = [(min_saddle[r] - critical[r], r, 0) for r in index]
        heapq.heapify(heap)

        while heap:
            height, r, ver = heapq.heappop(heap)
            if not alive[r] or ver != version[r]:
                continue
            # The heap yields motifs by ascending local height; once the smallest exceeds the threshold, all remaining
            # motifs are significant and the pruning is complete.
            if height >= threshold:
                break
            # Merge the least significant motif r into the neighbour reached across its (cached) lowest saddle.
            neighbour = min_neighbour[r]
            neighbour_adj = adjacency[neighbour]
            del neighbour_adj[r]
            del adjacency[r][neighbour]
            transferred_min, transferred_neighbour = np.inf, -1
            for other, saddle_height in adjacency[r].items():
                # Replace the edge other-r by the edge other-neighbour. This leaves the minimum saddle *value* of every
                # such neighbour 'other' unchanged (an edge is merged into an equal-or-lower one), so its cache only
                # needs the neighbour label repointed from r to the survivor.
                existing = neighbour_adj.get(other)
                combined = saddle_height if (existing is None or saddle_height < existing) else existing
                neighbour_adj[other] = combined
                other_adj = adjacency[other]
                other_adj[neighbour] = combined
                del other_adj[r]
                if min_neighbour[other] == r:
                    min_neighbour[other] = neighbour
                if combined < transferred_min:
                    transferred_min, transferred_neighbour = combined, other
            # The deeper pit (lower critical value) survives as the critical point of the combined motif.
            if critical[r] < critical[neighbour]:
                critical[neighbour] = critical[r]
                critical_loc[neighbour] = critical_loc[r]
            alive[r] = False
            merged_into[r] = neighbour
            # The survivor absorbed r's edges and lost its edge to r. If its previous closest neighbour was r the
            # minimum must be recomputed; otherwise the new minimum is the smaller of the old minimum and the smallest
            # absorbed edge, since every untouched edge of the survivor is no smaller than its old minimum.
            if min_neighbour[neighbour] == r:
                rescan(neighbour)
            elif transferred_min < min_saddle[neighbour]:
                min_saddle[neighbour], min_neighbour[neighbour] = transferred_min, transferred_neighbour
            version[neighbour] += 1
            heapq.heappush(heap, (min_saddle[neighbour] - critical[neighbour], neighbour, version[neighbour]))

        # Resolve the union-find chains so that every pre-pruning label maps directly to its surviving motif.
        region_of = np.zeros(n + 1, dtype=np.int64)
        for r in index:
            root = r
            while merged_into[root] != root:
                root = merged_into[root]
            region_of[r] = root

        roots = index[alive[1:]]
        if exclude_edge:
            # A surviving motif is incomplete if any pixel of its region lies on the border of the evaluation area.
            border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
            edge_roots = set(region_of[np.unique(border)].tolist())
            roots = roots[~np.isin(roots, list(edge_roots))]
        extrema = np.array([critical_loc[r] for r in roots], dtype=int).reshape(-1, 2)

        return _Segmentation(extrema=extrema, labels=labels, region_of=region_of)

    def _curvature(self, extrema):
        """
        Computes the local mean curvature at each critical point using central second differences,
        k = -1/2 (d2z/dx2 + d2z/dy2). Peaks (convex) yield positive values, pits (concave) negative values. Critical
        points on the array border are skipped, since the second difference is not defined there.

        Returns
        -------
        ndarray
            Curvatures of the interior critical points, in reciprocal length units (1/µm).
        """
        data = self._data
        ny, nx = data.shape
        rows, cols = extrema[:, 0], extrema[:, 1]
        interior = (rows > 0) & (rows < ny - 1) & (cols > 0) & (cols < nx - 1)
        rows, cols = rows[interior], cols[interior]
        if rows.size == 0:
            return np.empty(0)
        d2x = (data[rows, cols + 1] - 2 * data[rows, cols] + data[rows, cols - 1]) / self._step_x ** 2
        d2y = (data[rows + 1, cols] - 2 * data[rows, cols] + data[rows - 1, cols]) / self._step_y ** 2
        return -0.5 * (d2x + d2y)

    @property
    def _evaluation_area(self):
        return self._surface.width_um * self._surface.height_um

    # Parameters #######################################################################################################

    @cache
    def Spd(self, pruning=DEFAULT_PRUNING, exclude_edge=DEFAULT_EXCLUDE_EDGE):
        """
        Calculates the density of peaks Spd in 1/µm², the number of significant hills per unit area.

        Returns
        -------
        Spd : float
        """
        return self._segment(invert=True, pruning=pruning, exclude_edge=exclude_edge).count / self._evaluation_area

    @cache
    def Svd(self, pruning=DEFAULT_PRUNING, exclude_edge=DEFAULT_EXCLUDE_EDGE):
        """
        Calculates the density of pits Svd in 1/µm², the number of significant dales per unit area.

        Returns
        -------
        Svd : float
        """
        return self._segment(invert=False, pruning=pruning, exclude_edge=exclude_edge).count / self._evaluation_area

    @cache
    def Spc(self, pruning=DEFAULT_PRUNING, exclude_edge=DEFAULT_EXCLUDE_EDGE):
        """
        Calculates the arithmetic mean peak curvature Spc in 1/µm, the mean of the local mean curvature at the peaks of
        the significant hills.

        Returns
        -------
        Spc : float
        """
        curvature = self._curvature(self._segment(invert=True, pruning=pruning, exclude_edge=exclude_edge).extrema)
        return float(np.mean(curvature)) if curvature.size else np.nan

    @cache
    def Svc(self, pruning=DEFAULT_PRUNING, exclude_edge=DEFAULT_EXCLUDE_EDGE):
        """
        Calculates the arithmetic mean pit curvature Svc in 1/µm, the mean of the local mean curvature at the pits of
        the significant dales. Pits are concave, so Svc is negative.

        Returns
        -------
        Svc : float
        """
        curvature = self._curvature(self._segment(invert=False, pruning=pruning, exclude_edge=exclude_edge).extrema)
        return float(np.mean(curvature)) if curvature.size else np.nan

    @cache
    def S5p(self, pruning=DEFAULT_PRUNING, exclude_edge=DEFAULT_EXCLUDE_EDGE):
        """
        Calculates the five-point peak height S5p in µm, the mean of the heights of the five highest peaks (referenced
        to the mean plane). If fewer than five significant peaks are found, the mean is taken over those available.

        Returns
        -------
        S5p : float
        """
        extrema = self._segment(invert=True, pruning=pruning, exclude_edge=exclude_edge).extrema
        if extrema.shape[0] == 0:
            return np.nan
        heights = self._data[extrema[:, 0], extrema[:, 1]]
        return float(np.mean(np.sort(heights)[::-1][:_N_FIVE_POINT]))

    @cache
    def S5v(self, pruning=DEFAULT_PRUNING, exclude_edge=DEFAULT_EXCLUDE_EDGE):
        """
        Calculates the five-point pit depth S5v in µm, the mean of the depths of the five deepest pits (referenced to
        the mean plane). If fewer than five significant pits are found, the mean is taken over those available.

        Returns
        -------
        S5v : float
        """
        extrema = self._segment(invert=False, pruning=pruning, exclude_edge=exclude_edge).extrema
        if extrema.shape[0] == 0:
            return np.nan
        depths = -self._data[extrema[:, 0], extrema[:, 1]]
        return float(np.mean(np.sort(depths)[::-1][:_N_FIVE_POINT]))

    @cache
    def S10z(self, pruning=DEFAULT_PRUNING, exclude_edge=DEFAULT_EXCLUDE_EDGE):
        """
        Calculates the ten-point height S10z in µm, the sum of the five-point peak height and the five-point pit depth.

        Returns
        -------
        S10z : float
        """
        return (self.S5p(pruning=pruning, exclude_edge=exclude_edge)
                + self.S5v(pruning=pruning, exclude_edge=exclude_edge))

    # Plotting #########################################################################################################

    def plot_segmentation(self, kind='dale', pruning=DEFAULT_PRUNING, exclude_edge=DEFAULT_EXCLUDE_EDGE, ax=None,
                          cmap='jet'):
        """
        Plots the surface together with the boundaries of the significant motifs and their critical points.

        Parameters
        ----------
        kind : {'dale', 'hill'}, default 'dale'
            Whether to plot the dale (pit) or hill (peak) segmentation.
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Sz.
        exclude_edge : bool, default True
            If True, motifs touching the border of the evaluation area are not marked as significant features.
        ax : matplotlib axis, default None
            If specified, the plot is drawn on the given axis.
        cmap : str | mpl.cmap, default 'jet'
            Colormap applied to the height data.

        Returns
        -------
        plt.Figure, plt.Axes
        """
        if kind not in ('dale', 'hill'):
            raise ValueError("kind must be either 'dale' or 'hill'.")
        segmentation = self._segment(invert=(kind == 'hill'), pruning=pruning, exclude_edge=exclude_edge)
        # Relabel the surviving motifs to consecutive integers so the boundaries can be drawn as contour levels.
        motifs = np.unique(segmentation.region_of[segmentation.labels], return_inverse=True)[1].reshape(
            segmentation.labels.shape)

        if ax is None:
            fig, ax = plt.subplots(dpi=150)
        else:
            fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        extent = (0, self._surface.width_um, 0, self._surface.height_um)
        im = ax.imshow(self._data, cmap=cmap, extent=extent, origin='lower')
        fig.colorbar(im, cax=cax, label='z [µm]')
        # Draw the ridge/course lines as the boundaries between adjacent motifs.
        ax.contour(motifs, levels=np.arange(motifs.max() + 1) + 0.5, colors='k', linewidths=0.5, extent=extent,
                   origin='lower')
        extrema = segmentation.extrema
        if extrema.shape[0]:
            xs = extrema[:, 1] * self._step_x
            ys = extrema[:, 0] * self._step_y
            marker = 'v' if kind == 'dale' else '^'
            ax.scatter(xs, ys, c='white', edgecolors='k', marker=marker, s=20,
                       label='pits' if kind == 'dale' else 'peaks')
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')
        ax.legend(loc='upper right', fontsize=6)
        return fig, ax
