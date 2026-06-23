import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cache import CachedInstance, cache
from .mathutils import interpolate_line_on_2d_array, argmin_all, argmax_all, argclosest


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
        Calculates the decay lengths of the 2d autocorrelation function of the surface height data and
        extracts the indices of the points of minimum and maximum decay.

        Parameters
        ----------
        s : float
            threshold value below which the data is considered to be uncorrelated. The
            point of fastest and slowest decay are calculated respective to the threshold
            value, to which the autocorrelation function decays. The threshold s is a fraction
            of the maximum value of the autocorrelation function.

        Returns
        -------
        None
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
        distances_xy_px = idx_edge - self.center
        step_array = np.array([self._surface.step_y, self._surface.step_x])
        distances_xy_units = distances_xy_px * step_array
        distances = np.linalg.norm(distances_xy_units, axis=1)
        all_idx_min = idx_edge[argmin_all(distances)]
        all_idx_max = idx_edge[argmax_all(distances)]

        idx_min = all_idx_min[np.argmin(self.data[all_idx_min[:, 0], all_idx_min[:, 1]])]
        idx_max = all_idx_max[np.argmin(self.data[all_idx_max[:, 0], all_idx_max[:, 1]])]

        length_min = np.hypot(*((idx_min - self.center) * step_array))
        length_max = np.hypot(*((idx_max - self.center) * step_array))

        n_points = 1000
        interpolated_line_x = np.linspace(0, length_min, n_points)
        interpolated_line_y = interpolate_line_on_2d_array(self.data, self.center, idx_min, num_points=n_points)
        shortest_decay_length = interpolated_line_x[argclosest(threshold, interpolated_line_y)]

        interpolated_line_x = np.linspace(0, length_max, n_points)
        interpolated_line_y = interpolate_line_on_2d_array(self.data, self.center, idx_max, num_points=n_points)
        longest_decay_length = interpolated_line_x[argclosest(threshold, interpolated_line_y)]

        return shortest_decay_length, longest_decay_length


    @cache
    def Sal(self, s=0.2):
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

        Returns
        -------
        Sal : float
            autocorrelation length.
        """
        Sal, _ = self._calculate_decay_lengths(s)
        return Sal

    @cache
    def Str(self, s=0.2):
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

        Returns
        -------
        Str : float
            texture aspect ratio.
        """
        shortest_decay_length, longest_decay_length = self._calculate_decay_lengths(s)
        Str = shortest_decay_length / longest_decay_length
        return Str

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
            fig.colorbar(im, cax=cax, label='z [µm²]')
        else:
            cax.axis('off')
        ax.set_xlabel('lag x [µm]')
        ax.set_ylabel('lag y [µm]')

        return fig, ax