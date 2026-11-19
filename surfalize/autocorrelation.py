import warnings

import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cache import CachedInstance, cache
from .exceptions import CalculationError

# Behaviours for Sal/Str when the parameter is undefined because the autocorrelation function does not decay below the
# threshold within the evaluation area in the relevant direction (see AutocorrelationFunction.Str).
ON_UNDEFINED_OPTIONS = ('nan', 'bound', 'exception')


def _check_on_undefined(on_undefined):
    if on_undefined not in ON_UNDEFINED_OPTIONS:
        raise ValueError(f"on_undefined must be one of {ON_UNDEFINED_OPTIONS}, got {on_undefined!r}.")


class AutocorrelationFunction(CachedInstance):
    """
    Represents the 2d autocorrelation function of a Surface object and provides methods to calculate the autocorrelation
    length Sal and texture aspect ratio Str. The autocorrelation is computed as the unbiased linear (non-circular)
    estimator, i.e. each lag is normalized by the number of overlapping points, which avoids both the edge wrap-around
    of a circular estimator and the large-lag suppression of a biased (division by the total number of points) one.

    Parameters
    ----------
    surface : Surface
        Surface object on which to calculate the 2d autocorrelation function.
    """
    def __init__(self, surface):
        if surface.has_missing_points:
            raise ValueError("Missing points must be filled before "
                             "the autocorrelation function can be instantiated.") from None
        super().__init__()
        # For now we level and center. In the future, we should replace that with lookups of booleans
        # to avoid double computation
        self._surface = surface
        self._current_threshold = None
        self.data = self.calculate_autocorrelation()
        self.center = np.array(self.data.shape) // 2

    def calculate_autocorrelation(self):
        data = self._surface.center().data
        ny, nx = data.shape
        # Compute the linear (non-circular) autocorrelation via a zero-padded FFT. Correlating the data with a
        # point-reflected copy of itself yields the autocorrelation with the zero lag located at index (ny-1, nx-1).
        # In contrast to a plain (non-padded) fft2, the zero-padding prevents the surface from wrapping around its
        # edges, so points are only ever multiplied with genuinely overlapping neighbours instead of with values
        # tiled in from the opposite edge.
        raw = fftconvolve(data, data[::-1, ::-1], mode='full')
        # Apply the unbiased normalization: each lag is divided by the number of overlapping points rather than by
        # the total number of points. Dividing by the total number of points (the biased estimator) systematically
        # suppresses large lags, an effect that is strongest along diagonal directions where both lag components are
        # non-zero, and which biases the autocorrelation length Sal and texture aspect ratio Str.
        lag_y = np.abs(np.arange(-(ny - 1), ny))
        lag_x = np.abs(np.arange(-(nx - 1), nx))
        overlap_counts = np.outer(ny - lag_y, nx - lag_x)
        acf_data = raw / overlap_counts
        return acf_data

    @cache
    def _calculate_decay_lengths(self, s):
        """
        Calculates the decay lengths of the 2d autocorrelation function of the surface height data. The decay length
        is measured in every direction in which the autocorrelation function decays below the threshold within the
        evaluation area, and the shortest and longest of those lengths are returned (used for Sal and Str
        respectively).

        Parameters
        ----------
        s : float
            threshold value below which the data is considered to be uncorrelated. The
            point of fastest and slowest decay are calculated respective to the threshold
            value, to which the autocorrelation function decays. The threshold s is a fraction
            of the maximum value of the autocorrelation function.

        Returns
        -------
        tuple[float, float, bool]
            The shortest decay length, the longest decay length and a flag indicating whether the autocorrelation
            function decays below the threshold within the evaluation area in every direction. The shortest and
            longest lengths are np.nan if the function does not decay below the threshold anywhere within the field.
            When the flag is False (the thresholded region touches the border of the evaluation area), the function
            does not decay below the threshold in at least one direction; the shortest length is still the true
            fastest decay, but the longest length is only a lower bound on the true slowest decay, which is not
            contained within the field.
        """
        # The threshold is referenced to the autocorrelation value at zero lag (the central peak), which is the
        # maximum of a well-behaved autocorrelation function. Using the central value rather than the global maximum
        # keeps the threshold robust against the larger statistical noise that the unbiased estimator exhibits at
        # large lags, where only few points overlap.
        threshold = s * self.data[self.center[0], self.center[1]]

        mask = self.data > threshold
        labels, _ = ndimage.label(mask)
        region = labels == labels[self.center[0], self.center[1]]
        edge = region ^ ndimage.binary_dilation(region, iterations=1)

        idx_edge = np.argwhere(edge)
        if idx_edge.size == 0:
            # The autocorrelation function stays above the threshold across the entire evaluation area, so no decay
            # length can be determined in any direction. Both Sal and Str are undefined.
            return np.nan, np.nan, False

        # Measure the decay length in every edge direction and take the extremes from that single, self-consistent
        # set, rather than selecting the geometrically nearest and farthest edge pixels separately. The latter
        # decouples the direction used to pick a pixel (its geometric distance to the centre) from the quantity
        # actually measured (the interpolated threshold crossing along that direction), which does not guarantee
        # shortest <= longest and can produce Str > 1. Drawing both extremes from the same set of interpolated decay
        # lengths guarantees shortest <= longest, so Str stays within (0, 1].
        step_array = np.array([self._surface.step_y, self._surface.step_x])
        n_points = 1000
        # Interpolate the autocorrelation function along every centre->edge ray in a single map_coordinates call. Each
        # ray terminates at its edge pixel, which is the first pixel below the threshold along that direction, so the
        # threshold crossing is always bracketed within the ray.
        t = np.linspace(0, 1, n_points)
        rows = self.center[0] + (idx_edge[:, 0] - self.center[0])[:, None] * t[None, :]
        cols = self.center[1] + (idx_edge[:, 1] - self.center[1])[:, None] * t[None, :]
        coords = np.array([rows.ravel(), cols.ravel()])
        acf_along_rays = ndimage.map_coordinates(self.data, coords, order=3).reshape(len(idx_edge), n_points)

        distances = np.linalg.norm((idx_edge - self.center) * step_array, axis=1)
        lengths = distances[:, None] * t[None, :]
        crossing = np.argmin(np.abs(acf_along_rays - threshold), axis=1)
        decay_lengths = lengths[np.arange(len(idx_edge)), crossing]

        shortest_decay_length = decay_lengths.min()
        longest_decay_length = decay_lengths.max()

        # A thresholded region that reaches the border of the evaluation area means the autocorrelation function does
        # not decay below the threshold in at least one direction. Such a direction is necessarily a slow-decaying one
        # (it stays correlated across the whole field, as happens along the lamellae of a 1D or 2-beam DLIP structure),
        # so it can never be the fastest decay: the shortest length, and therefore Sal, remains valid. The slowest
        # decay, however, is not contained within the field, so the measured longest length is only a lower bound on
        # the true slowest decay, and Str is not determined by the measurement.
        region_touches_border = (region[0, :].any() or region[-1, :].any()
                                 or region[:, 0].any() or region[:, -1].any())
        decays_within_field = not region_touches_border

        return shortest_decay_length, longest_decay_length, decays_within_field


    @cache
    def Sal(self, s=0.2, on_undefined='nan'):
        """
        Calculates the autocorrelation length Sal. Sal represents the horizontal distance of the f_ACF(tx,ty)
        which has the fastest decay to a specified value s, with 0 < s < 1. s represents the fraction of the
        maximum value of the autocorrelation function. The default value for s is 0.2 according to ISO 25178-3.

        Parameters
        ----------
        s : float
            threshold value below which the data is considered to be uncorrelated. The
            point of fastest and slowest decay are calculated respective to the threshold
            value, to which the autocorrelation function decays. The threshold s is a fraction
            of the maximum value of the autocorrelation function.
        on_undefined : str, default 'nan'
            Behaviour when Sal is undefined because the autocorrelation function does not decay below the threshold
            anywhere within the evaluation area (the field is too small relative to the correlation length). Unlike
            Str, Sal has no bounded fallback in this case, so 'nan' and 'bound' behave identically:

            - 'nan' or 'bound': return np.nan and issue a RuntimeWarning.
            - 'exception': raise a CalculationError.

        Returns
        -------
        Sal : float
            autocorrelation length.
        """
        _check_on_undefined(on_undefined)
        Sal, _, _ = self._calculate_decay_lengths(s)
        if np.isnan(Sal):
            message = ("Sal is undefined because the autocorrelation function does not decay below the threshold "
                       "anywhere within the evaluation area. The evaluation area is too small relative to the "
                       "correlation length.")
            if on_undefined == 'exception':
                raise CalculationError(message)
            # Neither 'nan' nor 'bound' can supply a value: there is no decay length in any direction.
            warnings.warn(message, RuntimeWarning)
        return Sal

    @cache
    def Str(self, s=0.2, on_undefined='nan'):
        """
        Calculates the texture aspect ratio Str. Str represents the ratio of the horizontal distance of the f_ACF(tx,ty)
        which has the fastest decay to a specified value s to the horizontal distance of the fACF(tx,ty) which has the
        slowest decay to s, with 0 < s < 1. s represents the fraction of the maximum value of the autocorrelation
        function. The default value for s is 0.2 according to ISO 25178-3.

        Parameters
        ----------
        s : float
            threshold value below which the data is considered to be uncorrelated. The
            point of fastest and slowest decay are calculated respective to the threshold
            value, to which the autocorrelation function decays. The threshold s is a fraction
            of the maximum value of the autocorrelation function.
        on_undefined : str, default 'nan'
            Behaviour when Str is undefined because the autocorrelation function does not decay below the threshold
            within the evaluation area in at least one direction (for example along the lamellae of a 1D or 2-beam DLIP
            structure), so the slowest decay -- and therefore Str -- is not determined by the measurement:

            - 'nan': return np.nan and issue a RuntimeWarning.
            - 'bound': return the shortest length divided by the measured longest length and issue a RuntimeWarning.
              Because the measured longest length is only a lower bound on the true slowest decay, this ratio is an
              upper bound on the true Str rather than its value, and it depends on the size of the evaluation area. If
              the function does not decay anywhere at all, no bound exists and np.nan is returned.
            - 'exception': raise a CalculationError.

        Returns
        -------
        Str : float
            texture aspect ratio.
        """
        _check_on_undefined(on_undefined)
        shortest_decay_length, longest_decay_length, decays_within_field = self._calculate_decay_lengths(s)
        if decays_within_field:
            return shortest_decay_length / longest_decay_length
        message = ("Str is undefined because the autocorrelation function does not decay below the threshold within "
                   "the evaluation area in at least one direction (for example along the lamellae of a 1D or 2-beam "
                   "DLIP structure). The evaluation area is too small to determine the longest decay length.")
        if on_undefined == 'exception':
            raise CalculationError(message)
        warnings.warn(message, RuntimeWarning)
        if on_undefined == 'bound' and not np.isnan(longest_decay_length):
            # The measured longest length is a lower bound on the true slowest decay, so this ratio is an upper bound
            # on the true Str. It is field-size-dependent and must be interpreted as such.
            return shortest_decay_length / longest_decay_length
        return np.nan

    def plot_autocorrelation(self, ax=None, cmap='jet', show_cbar=True):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # The autocorrelation function is indexed by the lag relative to the central peak, so the axes span from the
        # negative to the positive maximum lag in each direction.
        extent_x = self.center[1] * self._surface.step_x
        extent_y = self.center[0] * self._surface.step_y
        im = ax.imshow(self.data, cmap=cmap, extent=(-extent_x, extent_x, -extent_y, extent_y))
        if show_cbar:
            fig.colorbar(im, cax=cax, label='z / µm²')
        else:
            cax.axis('off')
        ax.set_xlabel('lag x / µm')
        ax.set_ylabel('lag y / µm')

        return fig, ax