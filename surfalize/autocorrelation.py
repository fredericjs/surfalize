import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cache import CachedInstance, cache
from .mathutils import interpolate_line_on_2d_array, argmin_all, argmax_all, argclosest


class AutocorrelationFunction(CachedInstance):
    """
    Represents the 2d autocorrelation function of a Surface object and provides methods to calculate the autocorrelation
    length Sal and texture aspect ratio Str.

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
        data_fft = np.fft.fft2(data)
        # Compute ACF from FFT and normalize
        acf_data = np.fft.fftshift(np.fft.ifft2(data_fft * np.conj(data_fft)).real / data.size)
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
        threshold = s * self.data.max()

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
        im = ax.imshow(self.data, cmap=cmap, extent=(0, self._surface.width_um, 0, self._surface.height_um))
        if show_cbar:
            fig.colorbar(im, cax=cax, label='z [µm²]')
        else:
            cax.axis('off')
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')

        return fig, ax