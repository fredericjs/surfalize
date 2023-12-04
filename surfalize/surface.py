# Standard imports
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
from functools import wraps, lru_cache
from collections import namedtuple

# Scipy stack
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from scipy.linalg import lstsq
from scipy.interpolate import griddata
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage

# Custom imports
from .fileloader import load_file
from .utils import argclosest, interp1d
from .common import sinusoid
from .autocorrelation import AutocorrelationFunction
from .abbottfirestone import AbbottFirestoneCurve
from .profile import Profile
try:
    from .calculations import surface_area
    CYTHON_DEFINED = True
except ImportError:
    logger.warning('Could not import cythonized code. Surface area calculation unavailable.')
    CYTHON_DEFINED = False

size = namedtuple('Size', ['y', 'x'])

# Deprecate
def _period_from_profile(profile):
    """
    Extracts the period in pixel units from a surface profile using the Fourier transform.
    Parameters
    ----------
    profile: array or arra-like

    Returns
    -------
    period
    """
    # Estimate the period by means of fourier transform on first profile
    fft = np.abs(np.fft.fft(profile))
    freq = np.fft.fftfreq(profile.shape[0])
    peaks, properties = find_peaks(fft.flatten(), distance=10, prominence=10)
    # Find the prominence of the peaks
    prominences = properties['prominences']
    # Sort in descending order by computing sorting indices
    sorted_indices = np.argsort(prominences)[::-1]
    # Sort peaks in descending order
    peaks_sorted = peaks[sorted_indices]
    # Rearrange prominences based on the sorting of peaks
    prominences_sorted = prominences[sorted_indices]
    period = 1/np.abs(freq[peaks_sorted[0]])
    return period
           
def no_nonmeasured_points(function):
    @wraps(function)
    def wrapper_function(self, *args, **kwargs):
        if self._nonmeasured_points_exist:
            raise ValueError("Non-measured points must be filled before any other operation.")
        return function(self, *args, **kwargs)
    return wrapper_function


class Surface:
    
    AVAILABLE_PARAMETERS = ('Sa', 'Sq', 'Sp', 'Sv', 'Sz', 'Ssk', 'Sku', 'Sdr', 'Sdq', 'Sal', 'Str', 'Sk', 'Spk', 'Svk',
                            'Smr1', 'Smr2', 'Sxp', 'Vmp', 'Vmc', 'Vvv', 'Vvc', 'period', 'depth', 'aspect_ratio',
                            'homogeneity')
    CACHED_METODS = []
    
    def __init__(self, height_data, step_x, step_y):
        self.data = height_data
        self.step_x = step_x
        self.step_y = step_y
        self.width_um = (height_data.shape[1] - 1) * step_x
        self.height_um = (height_data.shape[0] - 1) * step_y
        # True if non-measured points exist on the surface
        self._nonmeasured_points_exist = np.any(np.isnan(self.data))

    @property
    def size(self):
        return size(*self.data.shape)

    def _clear_cache(self):
        for method in self.CACHED_METODS:
            method.cache_clear()
            
    def _set_data(self, data=None, step_x=None, step_y=None):
        if data is not None:
            self.data = data
        if step_x is not None:
            self.step_x = step_x
        if step_y is not None:
            self.step_y = step_y
        self.width_um = (self.size.x - 1) * self.step_x
        self.height_um = (self.size.y - 1) * self.step_y
        self._clear_cache()
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.width_um:.2f} x {self.height_um:.2f} µm²)'
    
    def _repr_png_(self):
        """
        Repr method for Jupyter notebooks. When Jupyter makes a call to repr, it checks first if a _repr_png_ is
        defined. If not, it falls back on __repr__.
        """
        self.show()

    @classmethod
    def load(cls, filepath):
        return cls(*load_file(filepath))
        
    def get_horizontal_profile(self, y, average=1, average_step=None):
        """
        Extracts a horizontal profile from the surface with optional averaging over parallel profiles.
        Profiles on the edge might be averaged over fewer profiles.

        Parameters
        ----------
        y: float
            vertical (height) value in µm from where the profile is extracted. The value is rounded to the closest data
            point.
        average: int
            number of profiles over which to average. Defaults to 1. Profiles will be extracted above and below the
            position designated by y.
        average_step: float, default None
            distance in µm between parallel profiles used for averaging. The value is rounded to the closest integer
            multiple of the pixel resolution. If the value is None, a distance of 1 px will be assumed.

        Returns
        -------
        profile: surfalize.Profile
        """
        if y > self.height_um:
            raise ValueError("y must not exceed height of surface.")
        
        if average_step is None:
            average_step_px = 1
        else:
            average_step_px = int(average_step / self.step_y)

        # vertical index of profile
        idx = int(y / self.height_um * self.size.y)
        # first index from which a profile is taken for averaging
        idx_min = idx - int(average / 2) * average_step_px
        idx_min = 0 if idx_min < 0 else idx_min
        # last index from which a profile is taken for averaging
        idx_max = idx + int(average / 2) * average_step_px
        idx_max = self.size.y if idx_max > self.size.y else idx_max
        data = self.data[idx_min:idx_max + 1:average_step_px].mean(axis=0)
        return Profile(data, self.step_x, self.width_um)
    
    def get_vertical_profile(self, x, average=1, average_step=None):
        """
         Extracts a vertical profile from the surface with optional averaging over parallel profiles.
         Profiles on the edge might be averaged over fewer profiles.

         Parameters
         ----------
         x: float
             laterial (width) value in µm from where the profile is extracted. The value is rounded to the closest data
             point.
         average: int
             number of profiles over which to average. Defaults to 1. Profiles will be extracted above and below the
             position designated by x.
         average_step: float, default None
             distance in µm between parallel profiles used for averaging. The value is rounded to the closest integer
             multiple of the pixel resolution. If the value is None, a distance of 1 px will be assumed.

         Returns
         -------
         profile: surfalize.Profile
         """
        if x > self.width_um:
            raise ValueError("x must not exceed height of surface.")
        
        if average_step is None:
            average_step_px = 1
        else:
            average_step_px = int(average_step / self.step_x)

        # vertical index of profile
        idx = int(x / self.width_um * self.size.x)
        # first index from which a profile is taken for averaging
        idx_min = idx - int(average / 2) * average_step_px
        idx_min = 0 if idx_min < 0 else idx_min
        # last index from which a profile is taken for averaging
        idx_max = idx + int(average / 2) * average_step_px
        idx_max = self.size.x if idx_max > self.size.x else idx_max
        data = self.data[:, idx_min:idx_max + 1:average_step_px].mean(axis=1)
        return Profile(data, self.step_y, self.height_um)
    
    def get_oblique_profile(self, x0, y0, x1, y1):
        x0px = int(x0 / self.width_um * self.size.x)
        y0px = int(y0 / self.height_um * self.size.y)
        x1px = int(x1 / self.width_um * self.size.x)
        y1px = int(y1 / self.height_um * self.size.y)

        if (not(0 <= x0px <= self.size.x) or not(0 <= y0px <= self.size.y) or
            not(0 <= x1px <= self.size.x) or not(0 <= y1px <= self.size.y)):
            raise ValueError("Start- and endpoint coordinates must lie within the surface.")

        dx = x1px - x0px
        dy = y1px - y0px

        size = int(np.hypot(dx, dy))

        m = dy/dx
        xp = np.linspace(x0px, x1px, size)
        yp = m * xp

        data = ndimage.map_coordinates(self.data, [yp, xp])

        length_um = np.hypot(dy * self.step_y, dx * self.step_x)
        step = length_um / size
        return Profile(data, step, length_um)

    # Operations #######################################################################################################
    
    def center(self, inplace=False):
        """
        Centers the data around its mean value. The height of the surface will be distributed equally around 0.

        Parameters
        ----------
        inplace: bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self. 

        Returns
        -------
        surface: surfalize.Surface
            Surface object.
        """
        data = self.data - self.data.mean()
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)
    
    def zero(self, inplace=False):
        """
        Sets the minimum height of the surface to zero.

        Parameters
        ----------
        inplace: bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self. 

        Returns
        -------
        surface: surfalize.Surface
            Surface object.
        """
        data = self.data - self.data.min()
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)

    def remove_outliers(self, n=3, method='mean', inplace=False):
        """
        Removes outliers based on the n-sigma criterion. All values that fall outside n-standard deviations of the mean
        are replaced by nan values. The default is three standard deviations. This method supports operation on data
        which contains non-measured points.

        Parameters
        ----------
        n: float, default 3
            Number of standard deviations outside of which values are considered outliers if method is 'mean'. If the
            method is 'median', n represents the number of medians distances of the data to its median value.
        method: {'mean', 'median'}, default 'mean'
            Method by which to perform the outlier detection. The default method is mean, which removes outliers outside
            an interval of n standard deviations from the mean. The method 'median' removes outliers outside n median
            distances of the data to its median.
        inplace: bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        surface: surfalize.Surface
            Surface object.
        """
        data = self.data.copy()
        if method == 'mean':
            data[np.abs(data - np.nanmean(data)) > n * np.nanstd(data)] = np.nan
        elif method == 'median':
            dist = np.abs(data - np.nanmedian(data))
            data[dist > n * np.nanmedian(dist)] = np.nan
        else:
            raise ValueError("Invalid methode.")
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)

    def threshold(self, threshold=0.5, inplace=False):
        """
        Removes data outside of threshold percentage of the material ratio curve.
        The topmost percentage (given by threshold) of hight values and the lowest percentage of height values are
        replaced with non-measured points. This method supports operation on data which contains non-measured points.

        Parameters
        ----------
        threshold: float, default 0.5
            Percentage threshold value of the material ratio.
        inplace: bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        surface: surfalize.Surface
            Surface object.
        """
        y = np.sort(self.data[~np.isnan(self.data)])[::-1]
        x = np.arange(1, y.size + 1, 1) / y.size
        idx0 = argclosest(threshold / 100, x)
        idx1 = argclosest(1 - threshold / 100, x)
        data = self.data.copy()
        data[(data > y[idx0]) | (data < y[idx1])] = np.nan
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)

    def fill_nonmeasured(self, method='nearest', inplace=False):
        if not self._nonmeasured_points_exist:
            return self
        values = self.data.ravel()
        mask = ~np.isnan(values)

        grid_x, grid_y = np.meshgrid(np.arange(self.size.x), np.arange(self.size.y))
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        data_interpolated = griddata(points[mask], values[mask], (grid_x, grid_y), method=method)
        
        if inplace:
            self._set_data(data=data_interpolated)
            self._nonmeasured_points_exist = False
            return self
        return Surface(data_interpolated, self.step_x, self.step_y)
    
    @no_nonmeasured_points
    def level(self, inplace=False):
        #self.period.cache_clear() # Clear the LRU cache of the period method
        x, y = np.meshgrid(np.arange(self.size.x), np.arange(self.size.y))
        # Flatten the x, y, and height_data arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        height_flat = self.data.flatten()
        # Create a design matrix A for linear regression
        A = np.column_stack((x_flat, y_flat, np.ones_like(x_flat)))
        # Use linear regression to fit a plane to the data
        coefficients, _, _, _ = lstsq(A, height_flat)
        # Extract the coefficients for the plane equation
        a, b, c = coefficients
        # Calculate the plane values for each point in the grid
        plane = a * x + b * y + c
        # Subtract the plane from the original height data to level it
        leveled_data = self.data - plane
        if inplace:
            self._set_data(data=leveled_data)
            return self
        return Surface(leveled_data, self.step_x, self.step_y)
    
    @no_nonmeasured_points
    def rotate(self, angle, inplace=False):
        rotated = ndimage.rotate(self.data, angle, reshape=True)

        aspect_ratio = self.size.y / self.size.x
        rotated_aspect_ratio = rotated.shape[0] / rotated.shape[1]

        if aspect_ratio < 1:
            total_height = self.size.y / rotated_aspect_ratio
        else:
            total_height = self.size.x

        pre_comp_sin = np.abs(np.sin(np.deg2rad(angle)))
        pre_comp_cos = np.abs(np.cos(np.deg2rad(angle)))

        w = total_height / (aspect_ratio * pre_comp_sin + pre_comp_cos)
        h = w * aspect_ratio

        ny, nx = rotated.shape
        ymin = int((ny - h)/2) + 1
        ymax = int(ny - (ny - h)/2) - 1
        xmin = int((nx - w)/2) + 1
        xmax = int(nx - (nx - w)/2) - 1
        
        rotated_cropped = rotated[ymin:ymax+1, xmin:xmax+1]
        width_um = (self.width_um * pre_comp_cos + self.height_um * pre_comp_sin) * w / nx
        height_um = (self.width_um * pre_comp_sin + self.height_um * pre_comp_cos) * h / ny
        step_y = height_um / rotated_cropped.shape[0]
        step_x = width_um / rotated_cropped.shape[1]

        if inplace:
            self._set_data(data=rotated_cropped, step_x=step_x, step_y=step_y)
            return self

        return Surface(rotated_cropped, step_x, step_y)
    
    @no_nonmeasured_points
    def filter(self, cutoff, *, mode, cutoff2=None, inplace=False):
        """
        Filters the surface by zeroing bins in the Fourier Transform. This introduces errors since the direct zeroing
        of bins is equivalent to applying a rectangular windowing function to the dft, which is equivalent to the
        convolution of the signal with a sinc(x) function! Use at your own risk.

        There a several possible modes of filtering:

        - 'highpass': computes spatial frequencies above the specified cutoff value
        - 'lowpass': computes spatial frequencies below the specified cutoff value
        - 'both': computes and returns both the highpass and lowpass filtered surfaces
        - 'bandpass': computes frequencies below the specified cutoff value and above the value specified for cutoff2

        The surface object's data can be changed inplace by specifying 'inplace=True' for 'highpass', 'lowpass' and
        'bandpass' mode. For mode='both', inplace=True will raise a ValueError.

        Parameters
        ----------
        cutoff: float
            Cutoff frequency in 1/µm at which the high and low spatial frequencies are separated.
            Actual cutoff will be rounded to the nearest pixel unit (1/px) equivalent.
        mode: str
            Mode of filtering. Possible values: 'highpass', 'lowpass', 'both', 'bandpass'.
        cutoff2: float
            Used only in mode='bandpass'. Specifies the lower cutoff frequency of the bandpass filter. Must be greater
            than cutoff.
        inplace: bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self. Inplace operation is not compatible with mode='both' argument, since two surfalize.Surface
            objects will be returned.

        Returns
        -------
        surface: surfalize.Surface
            Surface object.
        """
        if mode not in ('highpass', 'lowpass', 'both', 'bandpass'):
            raise ValueError("Invalid mode selected")
        if mode == 'both' and inplace:
            raise ValueError(
                "Mode 'both' does not support inplace operation since two Surface objects will be returned")

        if mode == 'bandpass':
            if cutoff2 is None:
                raise ValueError("cutoff2 must be provided.")
            if cutoff2 <= cutoff:
                raise ValueError("The value of cutoff2 must be greater than the value of cutoff.")
            cutoff_freq2 = 1 / cutoff2

        cutoff_freq = 1 / cutoff
        dft = np.fft.fftshift(np.fft.fft2(self.data))
        freq_y = np.fft.fftshift(np.fft.fftfreq(self.size.y, self.step_y))
        freq_x = np.fft.fftshift(np.fft.fftfreq(self.size.x, self.step_x))
        freq_x, freq_y = np.meshgrid(freq_x, freq_y)
        freq = np.sqrt(freq_x ** 2 + freq_y ** 2)
        filter_ = freq <= cutoff_freq
        if mode == 'lowpass':
            data_filtered = np.fft.ifft2(np.fft.ifftshift(dft * filter_)).real
            if inplace:
                self._set_data(data=data_filtered)
                return self
            return Surface(data_filtered, self.step_x, self.step_y)
        if mode == 'highpass':
            data_filtered = np.fft.ifft2(np.fft.ifftshift(dft * ~filter_)).real
            if inplace:
                self._set_data(data=data_filtered)
                return self
            return Surface(data_filtered, self.step_x, self.step_y)
        if mode == 'both':
            data_lowpass = np.fft.ifft2(np.fft.ifftshift(dft * filter_)).real
            data_highpass = np.fft.ifft2(np.fft.ifftshift(dft * ~filter_)).real
            return Surface(data_lowpass, self.step_x, self.step_y), Surface(data_highpass, self.step_x, self.step_y)
        if mode == 'bandpass':
            filter_lowpass = filter_
            filter_highpass = freq >= cutoff_freq2
            data_filtered = np.fft.ifft2(np.fft.ifftshift(dft * filter_lowpass * filter_highpass)).real
            if inplace:
                self._set_data(data=data_filtered)
                return self
            return Surface(data_filtered, self.step_x, self.step_y)
        
    def zoom(self, factor, inplace=False):
        """
        Magnifies the surface by the specified factor.

        Parameters
        ----------
        factor: float
            Factor by which the surface is magnified
        inplace: bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self

        Returns
        -------
        surface: surfalize.Surface
            Surface object.
        """
        y, x = self.size
        xn, yn = int(x / factor), int(y / factor)
        data = self.data[int((y - yn) / 2):yn + int((y - yn) / 2) + 1, int((x - xn) / 2):xn + int((x - xn) / 2) + 1]
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)

    def crop(self, box, inplace=False):
        """
        Crop the surface to the area specified by the box parameter.

        Parameters
        ----------
        box: tuple[float, float, float, float]
            The crop rectangle, as a (x0, x1, y0, y1) tuple.

        Returns
        -------
        surface: surfalize.Surface
            Surface object.
        """
        x0 = round(box[0] / self.step_x)
        x1 = round(box[1] / self.step_x)
        y1 = self.size.y - round(box[2] / self.step_y) - 1
        y0 = self.size.y - round(box[3] / self.step_y) - 1

        if x0 < 0 or y0 < 0 or x1 > self.size.x - 1 or y1 > self.size.y - 1:
            raise ValueError('Box is out of bounds!')

        data = self.data[y0:y1 + 1, x0:x1 + 1]
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)

    
    def align(self, inplace=False):
        """
        Computes the dominant orientation of the surface pattern and alignes the orientation with the horizontal
        or vertical axis.

        Parameters
        ----------
        inplace: bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self

        Returns
        -------
        surface: surfalize.Surface
            Surface object.
        """
        angle = self.orientation()
        return self.rotate(angle, inplace=inplace)

    @lru_cache
    def _get_fourier_peak_dx_dy(self):
        """
        Calculates the distance in x and y in spatial frequency length units. The zero peak is avoided by
        centering the data around the mean. This method is used by the period and orientation calculation.

        Returns
        -------
        (dx, dy): tuple[float,float]
            Distance between largest Fourier peaks in x (dx) and in y (dy)
        """
        # Get rid of the zero peak in the DFT for data that features a substantial offset in the z-direction
        # by centering the values around the mean
        data = self.data - self.data.mean()
        fft = np.abs(np.fft.fftshift(np.fft.fft2(data)))
        N, M = self.size
        # Calculate the frequency values for the x and y axes
        freq_x = np.fft.fftshift(np.fft.fftfreq(M, d=self.width_um / M))  # Frequency values for the x-axis
        freq_y = np.fft.fftshift(np.fft.fftfreq(N, d=self.height_um / N))  # Frequency values for the y-axis
        # Find the peaks in the magnitude spectrum
        peaks, properties = find_peaks(fft.flatten(), distance=10, prominence=10)
        # Find the prominence of the peaks
        prominences = properties['prominences']
        # Sort in descending order by computing sorting indices
        sorted_indices = np.argsort(prominences)[::-1]
        # Sort peaks in descending order
        peaks_sorted = peaks[sorted_indices]
        # Rearrange prominences based on the sorting of peaks
        prominences_sorted = prominences[sorted_indices]
        # Get peak coordinates in pixels
        peaks_y_px, peaks_x_px = np.unravel_index(peaks_sorted, fft.shape)
        # Transform into spatial frequencies in length units
        # If this is not done, the computed angle will be wrong since the frequency per pixel
        # resolution is different in x and y due to the different sampling length!
        peaks_x = freq_x[peaks_x_px]
        peaks_y = freq_y[peaks_y_px]
        # Create peak tuples for ease of use
        peak0 = (peaks_x[0], peaks_y[0])
        peak1 = (peaks_x[1], peaks_y[1])
        # Peak1 should always be to the right of peak0
        if peak0[0] > peak1[0]:
            peak0, peak1 = peak1, peak0

        dx = peak1[0] - peak0[0]
        dy = peak0[1] - peak1[1]

        return dx, dy

    CACHED_METODS.append(_get_fourier_peak_dx_dy)

    # Characterization #################################################################################################
   
    # Height parameters ################################################################################################
    
    @no_nonmeasured_points
    def Sa(self):
        return (np.abs(self.data - self.data.mean()).sum() / self.data.size)
    
    @no_nonmeasured_points
    def Sq(self):
        return np.sqrt(((self.data - self.data.mean()) ** 2).sum() / self.data.size).round(8)
    
    @no_nonmeasured_points
    def Sp(self):
        return (self.data - self.data.mean()).max()
    
    @no_nonmeasured_points
    def Sv(self):
        return np.abs((self.data - self.data.mean()).min())
    
    @no_nonmeasured_points
    def Sz(self):
        return self.Sp() + self.Sv()
    
    @no_nonmeasured_points
    def Ssk(self):
        return ((self.data - self.data.mean()) ** 3).sum() / self.data.size / self.Sq()**3
    
    @no_nonmeasured_points
    def Sku(self):
        return ((self.data - self.data.mean()) ** 4).sum() / self.data.size / self.Sq()**4
    
    # Hybrid parameters ################################################################################################
    
    def projected_area(self):
        return (self.width_um - self.step_x) * (self.height_um - self.step_y)
    
    @no_nonmeasured_points
    def surface_area(self, method='iso'):
        """
        Calculates the surface area of the surface. The method parameter can be either 'iso' or 'gwyddion'. The default
        method is the 'iso' method proposed by ISO 25178 and used by MountainsMap, whereby two triangles are
        spanned between four corner points. The 'gwyddion' method implements the approach used by the open-source
        software Gwyddion, whereby four triangles are spanned between four corner points and their calculated center
        point. The method is detailed here: http://gwyddion.net/documentation/user-guide-en/statistical-analysis.html.

        Parameters
        ----------
        method: str, Default 'iso'
            The method by which to calculate the surface area.
        Returns
        -------
        area: float
        """
        if not CYTHON_DEFINED:
            raise NotImplementedError("Surface area calculation is based on cython code. Compile cython code to run this"
                                      "method")
        return surface_area(self.data, self.step_x, self.step_y, method=method)
    
    @no_nonmeasured_points
    def Sdr(self, method='iso'):
        """
        Calculates Sdr. The method parameter can be either 'iso' or 'gwyddion'. The default method is the 'iso' method
        proposed by ISO 25178 and used by MountainsMap, whereby two triangles are spanned between four corner points.
        The 'gwyddion' method implements the approach used by the open-source software Gwyddion, whereby four triangles
        are spanned between four corner points and their calculated center point. The method is detailed here:
        http://gwyddion.net/documentation/user-guide-en/statistical-analysis.html.

        Parameters
        ----------
        method: str, Default 'iso'
            The method by which to calculate the surface area.
        Returns
        -------
        area: float
        """
        return (self.surface_area(method=method) / self.projected_area() -1) * 100
    
    @no_nonmeasured_points
    def Sdq(self):
        A = self.size.y * self.size.x
        diff_x = np.diff(self.data, axis=1) / self.step_x
        diff_y = np.diff(self.data, axis=0) / self.step_y
        return np.sqrt((np.sum(diff_x**2) + np.sum(diff_y**2)) / A)

    # Spatial parameters ###############################################################################################

    @lru_cache
    def _get_autocorrelation_function(self):
        return AutocorrelationFunction(self)

    CACHED_METODS.append(_get_autocorrelation_function)

    def Sal(self, s=0.2):
        """
        Calculates the autocorrelation length Sal. Sal represents the horizontal distance of the f_ACF(tx,ty)
        which has the fastest decay to a specified value s, with 0 < s < 1. s represents the fraction of the
        maximum value of the autocorrelation function. The default value for s is 0.2 according to ISO 25178-3.

        Parameters
        ----------
        s: float
            threshold value below which the data is considered to be uncorrelated. The
            point of fastest and slowest decay are calculated respective to the threshold
            value, to which the autocorrelation function decays. The threshold s is a fraction
            of the maximum value of the autocorrelation function.

        Returns
        -------
        Sal: float
            autocorrelation length.
        """
        return self._get_autocorrelation_function().Sal(s=s)

    def Str(self, s=0.2):
        """
        Calculates the texture aspect ratio Str. Str represents the ratio of the horizontal distance of the f_ACF(tx,ty)
        which has the fastest decay to a specified value s to the horizontal distance of the fACF(tx,ty) which has the
        slowest decay to s, with 0 < s < 1. s represents the fraction of the maximum value of the autocorrelation
        function. The default value for s is 0.2 according to ISO 25178-3.

        Parameters
        ----------
        s: float
            threshold value below which the data is considered to be uncorrelated. The
            point of fastest and slowest decay are calculated respective to the threshold
            value, to which the autocorrelation function decays. The threshold s is a fraction
            of the maximum value of the autocorrelation function.

        Returns
        -------
        Str: float
            texture aspect ratio.
        """
        return self._get_autocorrelation_function().Str(s=s)
    
    # Functional parameters ############################################################################################
    
    @lru_cache
    def _get_abbott_firestone_curve(self):
        return AbbottFirestoneCurve(self)

    CACHED_METODS.append(_get_abbott_firestone_curve)

    def Sk(self):
        """
        Calculates Sk in µm.

        Returns
        -------
        Sk: float
        """
        return self._get_abbott_firestone_curve().Sk()

    def Spk(self):
        """
        Calculates Spk in µm.

        Returns
        -------
        Spk: float
        """
        return self._get_abbott_firestone_curve().Spk()

    def Svk(self):
        """
        Calculates Svk in µm.

        Returns
        -------
        Svk: float
        """
        return self._get_abbott_firestone_curve().Svk()

    def Smr1(self):
        """
        Calculates Smr1 in %.

        Returns
        -------
        Smr1: float
        """
        return self._get_abbott_firestone_curve().Smr1()

    def Smr2(self):
        """
        Calculates Smr2 in %.

        Returns
        -------
        Smr2: float
        """
        return self._get_abbott_firestone_curve().Smr2()

    def Smr(self, c):
        """
        Calculates the ratio of the area of the material at a specified height c (in µm) to the evaluation area.

        Parameters
        ----------
        c: float
            height in µm.

        Returns
        -------
        areal material ratio: float
        """
        return self._get_abbott_firestone_curve().Smr(c)

    def Smc(self, mr):
        """
        Calculates the height (c) in µm for a given areal material ratio (mr).

        Parameters
        ----------
        mr: float
            areal material ratio in %.

        Returns
        -------
        height: float
        """
        return self._get_abbott_firestone_curve().Smc(mr)

    def Sxp(self, p=2.5, q=50):
        """
        Calculates the difference in height between the p and q material ratio. For Sxp, p and q are defined by the
        standard ISO 25178-3 to be 2.5% and 50%, respectively.

        Parameters
        ----------
        p: float
            material ratio p in % as defined by the standard ISO 25178-3
        q: float
            material ratio q in % as defined by the standard ISO 25178-3

        Returns
        -------
        Height difference: float
        """
        return self.Smc(p) - self.Smc(q)

    # Functional volume parameters ######################################################################################

    def Vmp(self, p=10):
        return self._get_abbott_firestone_curve().Vmp(p=p)

    def Vmc(self, p=10, q=80):
        return self._get_abbott_firestone_curve().Vmc(p=p, q=q)

    def Vvv(self, q=80):
        return self._get_abbott_firestone_curve().Vvv(q=q)

    def Vvc(self, p=10, q=80):
        return self._get_abbott_firestone_curve().Vvc(p=p, q=q)

    # Non-standard parameters ##########################################################################################
    
    @lru_cache
    @no_nonmeasured_points
    def period(self):
        logger.debug('period called.')
        dx, dy = self._get_fourier_peak_dx_dy()
        period = 2/np.hypot(dx, dy)
        return period

    CACHED_METODS.append(period)
    
    @lru_cache
    @no_nonmeasured_points
    def orientation(self):
        dx, dy = self._get_fourier_peak_dx_dy()
        #Account for special cases
        if dx == 0:
            orientation = 90
        elif dy == 0:
            orientation = 0
        else:
            orientation = np.rad2deg(np.arctan(dy/dx))
        return orientation

    CACHED_METODS.append(orientation)
    
    @no_nonmeasured_points
    def homogeneity(self):
        period = self.period()
        cell_length = int(period / self.height_um * self.size.y)
        ncells = int(self.size.y / cell_length) * int(self.size.x / cell_length)
        sa = np.zeros(ncells)
        ssk = np.zeros(ncells)
        sku = np.zeros(ncells)
        sdr = np.zeros(ncells)
        for i in range(int(self.size.y / cell_length)):
            for j in range(int(self.size.x / cell_length)):
                idx = i * int(self.size.x / cell_length) + j
                data = self.data[cell_length * i:cell_length * (i + 1), cell_length * j:cell_length * (j + 1)]
                cell_surface = Surface(data, self.step_x, self.step_y)
                sa[idx] = cell_surface.Sa()
                ssk[idx] = cell_surface.Ssk()
                sku[idx] = cell_surface.Sku()
                sdr[idx] = cell_surface.Sdr()
        sa = np.sort(sa.round(8))
        ssk = np.sort(np.abs(ssk).round(8))
        sku = np.sort(sku.round(8))
        sdr = np.sort(sdr.round(8))

        h = []
        for param in (sa, ssk, sku, sdr):
            if np.all(param == 0):
                h.append(1)
                continue
            x, step = np.linspace(0, 1, ncells, retstep=True)
            lorenz = np.cumsum(np.abs(param))
            lorenz = (lorenz - lorenz.min()) / lorenz.max()
            y = lorenz.min() + (lorenz.max() - lorenz.min()) * x
            total = np.trapz(y, dx=step)
            B = np.trapz(lorenz, dx=step)
            A = total - B
            gini = A / total
            h.append(1 - gini)
        return np.mean(h).round(4)

    @lru_cache
    @no_nonmeasured_points
    def depth(self, nprofiles=30, sampling_width=0.2, retstd=False, plot=False):
        logger.debug('Depth called.')
        size, length = self.size
        if nprofiles > size:
            raise ValueError(f'nprofiles cannot exceed the maximum available number of profiles of {size}')

        # Obtain the period estimate from the fourier transform in pixel units
        period_ft_um = self.period()
        # Calculate the number of intervals per profile
        nintervals = int(self.width_um / period_ft_um)
        # Allocate depth array with twice the length of the number of periods to accomodate both peaks and valleys
        # multiplied by the number of sampled profiles
        depths = np.zeros(nprofiles * nintervals)

        # Loop over each profile
        for i in range(nprofiles):
            line = self.data[int(size / nprofiles) * i]
            period_px = _period_from_profile(line)
            xp = np.arange(line.size)
            # Define initial guess for fit parameters
            p0=((line.max() - line.min())/2, period_px, 0, line.mean())
            # Fit the data to the general sine function
            popt, pcov = curve_fit(sinusoid, xp, line, p0=p0)
            # Extract the refined period estimate from the sine function period
            period_sin = popt[1]
            # Extract the lateral shift of the sine fit
            x0 = popt[2]

            depths_line = np.zeros(nintervals * 2)

            if plot and i == 4:
                fig, ax = plt.subplots(figsize=(16,4))
                ax.plot(xp, line, lw=1.5, c='k', alpha=0.7)
                ax.plot(xp, sinusoid(xp, *popt), c='orange', ls='--')
                ax.set_xlim(xp.min(), xp.max())

            # Loop over each interval
            for j in range(nintervals*2):
                idx = (0.25 + 0.5*j) * period_sin + x0        

                idx_min = int(idx) - int(period_sin * sampling_width/2)
                idx_max = int(idx) + int(period_sin * sampling_width/2)
                if idx_min < 0 or idx_max > length-1:
                    depths_line[j] = np.nan
                    continue
                depth_mean = line[idx_min:idx_max+1].mean()
                depth_median = np.median(line[idx_min:idx_max+1])
                depths_line[j] = depth_median
                # For plotting
                if plot and i == 4:          
                    rx = xp[idx_min:idx_max+1].min()
                    ry = line[idx_min:idx_max+1].min()
                    rw = xp[idx_max] - xp[idx_min+1]
                    rh = np.abs(line[idx_min:idx_max+1].min() - line[idx_min:idx_max+1].max())
                    rect = Rectangle((rx, ry), rw, rh, facecolor='tab:orange')
                    ax.plot([rx, rx+rw], [depth_mean, depth_mean], c='r')
                    ax.plot([rx, rx+rw], [depth_median, depth_median], c='g')
                    ax.add_patch(rect)   

            # Subtract peaks and valleys from eachother by slicing with 2 step
            depths[i*nintervals:(i+1)*nintervals] = np.abs(depths_line[0::2] - depths_line[1::2])

        if retstd:
            return np.nanmean(depths), np.nanstd(depths)
        return np.nanmean(depths)

    CACHED_METODS.append(depth)

    def aspect_ratio(self):
        """
        Calculates the aspect ratio of a periodic texture as the ratio of the structure depth and the structure period.

        Returns
        -------
        aspect_ratio: float
        """
        return self.depth() / self.period()

    def roughness_parameters(self, parameters=None):
        if parameters is None:
            parameters = self.AVAILABLE_PARAMETERS
        results = dict()
        for parameter in parameters:
            if parameter in self.AVAILABLE_PARAMETERS:
                results[parameter] = getattr(self, parameter)()
            else:
                raise ValueError(f'Parameter "{parameter}" is undefined.')
        return results

    # Plotting #########################################################################################################
    def abbott_curve(self, nbars=20):
        dist_bars, bins_bars = np.histogram(self.data, bins=nbars)
        dist_bars = np.flip(dist_bars)
        bins_bars = np.flip(bins_bars)

        nbins, bin_centers, cumsum = self._get_material_ratio_curve()

        fig, ax = plt.subplots()
        ax.set_xlabel('Material distribution (%)')
        ax.set_ylabel('z (µm)')
        ax2 = ax.twiny()
        ax2.set_xlabel('Material ratio (%)')
        ax.set_box_aspect(1)
        ax2.set_xlim(0, 100)
        ax.set_ylim(self.data.min(), self.data.max())

        ax.barh(bins_bars[:-1] + np.diff(bins_bars) / 2, dist_bars / dist_bars.cumsum().max() * 100,
                height=(self.data.max() - self.data.min()) / nbars, edgecolor='k', color='lightblue')
        ax2.plot(cumsum, bin_centers, c='r', clip_on=True)

        plt.show()
        
    def fourier_transform(self, log=True, hanning=False, subtract_mean=True, fxmax=None, fymax=None, cmap='inferno', adjust_colormap=True):
        """
        Plots the 2d Fourier transform of the surface. Optionally, a Hanning window can be applied to reduce to spectral leakage effects 
        that occur when analyzing a signal of finite sample length.

        Parameters
        ----------
        log: bool, Default True
            Shows the logarithm of the Fourier Transform to increase peak visibility.
        hanning: bool, Default False
            Applys a Hanning window to the data before the transform.
        subtract_mean: bool, Default False
            Subtracts the mean of the data before the transform to avoid the zero peak.
        fxmax: float, Default None
            Maximum frequency displayed in x. The plot will be cropped to -fxmax : fxmax.
        fymax: float, Default None
            Maximum frequency displayed in y. The plot will be cropped to -fymax : fymax.
        cmap: str, Default 'inferno'
            Matplotlib colormap with which to map the data.
        adjust_colormap: bool, Default True
            If True, the colormap starts at the mean and ends at 0.7 time the maximum of the data
            to increase peak visibility.
        Returns
        -------
        ax: matplotlib.axes
        """
        N, M = self.size
        data = self.data
        if subtract_mean:
            data = data - self.data.mean()

        if hanning:
            hann_window_y = np.hanning(N)
            hann_window_x = np.hanning(M)
            hann_window_2d = np.outer(hann_window_y, hann_window_x)
            data = data * hann_window_2d

        fft = np.abs(np.fft.fftshift(np.fft.fft2(data)))

        # Calculate the frequency values for the x and y axes
        freq_x = np.fft.fftshift(np.fft.fftfreq(M, d=self.width_um / M))  # Frequency values for the x-axis
        freq_y = np.fft.fftshift(np.fft.fftfreq(N, d=self.height_um / N))  # Frequency values for the y-axis

        if log:
            fft = np.log10(fft)
        ixmin = 0
        ixmax = M-1
        iymin = 0
        iymax = N-1

        if fxmax is not None:
            ixmax = argclosest(fxmax, freq_x)
            ixmin = M - ixmax
            fft = fft[:,ixmin:ixmax+1]

        if fymax is not None:
            iymax = argclosest(fymax, freq_y)
            iymin = N - iymax
            fft = fft[iymin:iymax+1]

        vmin = None
        vmax = None
        if adjust_colormap:
            vmin = fft.mean()
            vmax = 0.7 * fft.max()

        fig, ax = plt.subplots()
        ax.set_xlabel('Frequency [µm$^{-1}$]')
        ax.set_ylabel('Frequency [µm$^{-1}$]')
        extent = (freq_x[ixmin], freq_x[ixmax], freq_y[iymax], freq_y[iymin])

        ax.imshow(fft, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        return ax
    
    def show(self, cmap='jet', maskcolor='black'):
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(maskcolor)
        fig, ax = plt.subplots(dpi=150)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(self.data, cmap=cmap, extent=(0, self.width_um, 0, self.height_um))
        fig.colorbar(im, cax=cax, label='z [µm]')
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')
        if self._nonmeasured_points_exist:
            handles = [plt.plot([], [], marker='s', c=maskcolor, ls='')[0]]
            ax.legend(handles, ['non-measured points'], loc='lower right', fancybox=False, framealpha=1, fontsize=6)
        plt.show()
