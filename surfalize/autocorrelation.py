import numpy as np
import scipy.ndimage as ndimage
from skimage.measure import label
from skimage.segmentation import expand_labels
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cache import CachedInstance, cache
from .mathutils import interpolate_line_on_2d_array


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
        super().__init__()
        # For now we level and center. In the future, we should replace that with lookups of booleans
        # to avoid double computation
        self._surface = surface
        self._current_threshold = None
        self.data = self.calculate_autocorrelation()
        self._center = np.array(self.data.shape) // 2

    def calculate_autocorrelation(self):
        data = self._surface.center().data
        data_fft = np.fft.fft2(data)
        # Compute ACF from FFT and normalize
        acf_data = np.fft.fftshift(np.fft.ifft2(data_fft * np.conj(data_fft)).real / data.size)
        return acf_data

    def old_calculate_distances(self, s=0.2):
        """
        Calculates the distances of the 2d autocorrelation function of the surface height data and
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
        self.clear_cache()
        self._current_threshold = s

        threshold = s * self.acf_data.max()
        mask = (self.acf_data < threshold)

        # Find the center point of the array
        self.center = np.array(self.acf_data.shape) // 2

        # Invert mask because the function considers all 0 values to be background
        labels, _ = ndimage.label(~mask)
        feature_center_id = labels[self.center[0], self.center[1]]
        mask = (labels != feature_center_id)

        # Find the indices of the True values in the mask
        indices = np.argwhere(mask)
        # Calculate the Euclidean distance from each index to the center
        distances = np.linalg.norm(indices - self.center, axis=1)
        # Find the index with the smallest distance
        self._idx_min = indices[np.argmin(distances)]

        # Find the indices of the True values in the mask
        indices = np.argwhere(~mask)
        # Calculate the Euclidean distance from each index to the center
        distances = np.linalg.norm(indices - self.center, axis=1)
        self._idx_max = indices[np.argmax(distances)]

    def _calculate_distances(self, s):
        self.clear_cache()
        self._current_threshold = s
        self._absolute_threshold = s * self.data.max()
        mask = (self.data < self._absolute_threshold)

        # Invert mask because the function considers all 0 values to be background
        # Use image segementation to identify contiguous regions in the data
        labels = label(~mask)
        feature_center_id = labels[self.center[0], self.center[1]]
        # Remove all but the central region
        mask = (labels == feature_center_id)
        # Currently, the outer pixels of the center region are just above the threshold
        # So we expand the mask by one pixel to obtain the pixels that are just below the threshold
        expanded_mask = expand_labels(mask, distance=1)
        # Using xor we obtain the edge of the region with values just below the threshold
        edge = expanded_mask ^ mask

        # Calculate the Euclidean distance from each edge pixel to the center
        indices = np.argwhere(edge)

        # We need to calculate the distances in units for this to work for surfaces with unequal spacing
        # in x and y (looking at you OPD files, why???)
        distances_xy_px = indices - self.center
        distances_xy_units = distances_xy_px * np.array([self._surface.step_y, self._surface.step_x])
        distances = np.linalg.norm(distances_xy_units, axis=1)

        # Find the index with the smallest and largest distance
        self.idx_min = indices[np.argmin(distances)]
        self.idx_max = indices[np.argmax(distances)]


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
        if self._current_threshold != s:
            self._calculate_distances(s)
        dy, dx = np.abs(self._idx_min[0] - self.center[0]), np.abs(self._idx_min[1] - self.center[1])
        Sal = np.hypot(dx * self._surface.step_x, dy * self._surface.step_y) - self._surface.step_x/2
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
        if self._current_threshold != s:
            self._calculate_distances(s)
        dy, dx = np.abs(self._idx_max[0] - self.center[0]), np.abs(self._idx_max[1] - self.center[1])
        Str = self.Sal() / (np.hypot(dx * self._surface.step_x, dy * self._surface.step_y) - self._surface.step_x/2)
        return Str

    def plot_autocorrelation(self, ax=None, cmap='jet', show_cbar=True):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(self.acf_data, cmap=cmap, extent=(0, self._surface.width_um, 0, self._surface.height_um))
        if show_cbar:
            fig.colorbar(im, cax=cax, label='z [µm²]')
        else:
            cax.axis('off')
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')

        return fig, ax