# Standard imports
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from functools import lru_cache, wraps

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
from . fileloader import load_file
try:
    from .calculations import surface_area
    CYTHON_DEFINED = True
except ImportError:
    logger.warning('Could not import cythonized code. Surface area calculation unavailable.')
    CYTHON_DEFINED = False

# Define general sinusoid function for fitting
sinusoid = lambda x, a, p, xo, yo: a*np.sin((x-xo)/p*2*np.pi) + yo

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


class Profile:
    
    def __init__(self, height_data, step, length_um):
        self._data = height_data
        self._step = step
        self._length_um = length_um
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self._length_um:.2f} µm)'
    
    def _repr_png_(self):
        """
        Repr method for Jupyter notebooks. When Jupyter makes a call to repr, it checks first if a _repr_png_ is
        defined. If not, it falls back on __repr__.
        """
        self.show()
        
    def period(self):
        fft = np.abs(np.fft.fft(self._data))
        freq = np.fft.fftfreq(self._data.shape[0], d=self._step)
        peaks, properties = find_peaks(fft.flatten(), distance=10, prominence=10)
        # Find the prominence of the peaks
        prominences = properties['prominences']
        # Sort in descendin'g order by computing sorting indices
        sorted_indices = np.argsort(prominences)[::-1]
        # Sort peaks in descending order
        peaks_sorted = peaks[sorted_indices]
        # Rearrange prominences based on the sorting of peaks
        prominences_sorted = prominences[sorted_indices]
        period = 1/np.abs(freq[peaks_sorted[0]])
        return period
        
    def Ra(self):
        return np.abs(self._data - self._data.mean()).sum() / self._data.size
    
    def Rq(self):
        return np.sqrt(((self._data - self._data.mean()) ** 2).sum() / self._data.size)
    
    def Rp(self):
        return (self._data - self._data.mean()).max()
    
    def Rv(self):
        return np.abs((self._data - self._data.mean()).min())
    
    def Rz(self):
        return self.Sp() + self.Sv()
    
    def Rsk(self):
        return ((self._data - self._data.mean()) ** 3).sum() / self._data.size / self.Rq()**3
    
    def Rku(self):
        return ((self._data - self._data.mean()) ** 4).sum() / self._data.size / self.Rq()**4
    
    def depth(self, sampling_width=0.2, plot=False, retstd=False):
        period_px = int(self.period() / self._step)
        nintervals = int(self._data.shape[0] / period_px)
        xp = np.arange(self._data.shape[0])
        # Define initial guess for fit parameters
        p0=((self._data.max() - self._data.min())/2, period_px, 0, self._data.mean())
        # Fit the data to the general sine function
        popt, pcov = curve_fit(sinusoid, xp, self._data, p0=p0)
        # Extract the refined period estimate from the sine function period
        period_sin = popt[1]
        # Extract the lateral shift of the sine fit
        x0 = popt[2]

        depths_line = np.zeros(nintervals * 2)

        if plot:
            fig, ax = plt.subplots(figsize=(16,4))
            ax.plot(xp, self._data, lw=1.5, c='k', alpha=0.7)
            ax.plot(xp, sinusoid(xp, *popt), c='orange', ls='--')
            ax.set_xlim(xp.min(), xp.max())

        # Loop over each interval
        for i in range(nintervals*2):
            idx = (0.25 + 0.5*i) * period_sin + x0        

            idx_min = int(idx) - int(period_sin * sampling_width/2)
            idx_max = int(idx) + int(period_sin * sampling_width/2)
            if idx_min < 0 or idx_max > self._data.shape[0]-1:
                depths_line[i] = np.nan
                continue
            depth_mean = self._data[idx_min:idx_max+1].mean()
            depth_median = np.median(self._data[idx_min:idx_max+1])
            depths_line[i] = depth_median
            # For plotting
            if plot:          
                rx = xp[idx_min:idx_max+1].min()
                ry = self._data[idx_min:idx_max+1].min()
                rw = xp[idx_max] - xp[idx_min+1]
                rh = np.abs(self._data[idx_min:idx_max+1].min() - self._data[idx_min:idx_max+1].max())
                rect = plt.Rectangle((rx, ry), rw, rh, facecolor='tab:orange')
                ax.plot([rx, rx+rw], [depth_mean, depth_mean], c='r')
                ax.plot([rx, rx+rw], [depth_median, depth_median], c='g')
                ax.add_patch(rect)   

        # Subtract peaks and valleys from eachother by slicing with 2 step
        depths = np.abs(depths_line[0::2] - depths_line[1::2])

        if retstd:
            return np.nanmean(depths), np.nanstd(depths)
        return np.nanmean(depths)
    
    def show(self):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_xlim(0, self._length_um)
        ax.plot(np.linspace(0, self._length_um, self._data.size), self._data, c='k')
        plt.show()
        
           
def no_nonmeasured_points(function):
    @wraps(function)
    def wrapper_function(self, *args, **kwargs):
        if self._nonmeasured_points_exist:
            raise ValueError("Non-measured points must be filled before any other operation.")
        return function(self, *args, **kwargs)
    return wrapper_function
            
            
# TODO Image export
# TODO Oblique profiles
class Surface:
    
    AVAILABLE_PARAMETERS = ('Sa', 'Sq', 'Sp', 'Sv', 'Sz', 'Ssk', 'Sku', 'Sdr', 'period', 'homogeneity', 'depth')
    
    def __init__(self, height_data, step_x, step_y, width_um, height_um):
        self._data = height_data
        self._step_x = step_x
        self._step_y = step_y
        self._width_um = width_um
        self._height_um = height_um
        # True if non-measured points exist on the surface
        self._nonmeasured_points_exist = np.any(np.isnan(self._data))
        
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self._width_um:.2f} x {self._height_um:.2f} µm²)'
    
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
        if y > self._height_um:
            raise ValueError("y must not exceed height of surface.")
        
        if average_step is None:
            average_step_px = 1
        else:
            average_step_px = int(average_step / self._step_y)

        # vertical index of profile
        idx = int(y / self._height_um * self._data.shape[0])
        # first index from which a profile is taken for averaging
        idx_min = idx - int(average / 2) * average_step_px
        idx_min = 0 if idx_min < 0 else idx_min
        # last index from which a profile is taken for averaging
        idx_max = idx + int(average / 2) * average_step_px
        idx_max = self._data.shape[0] if idx_max > self._data.shape[0] else idx_max
        data = self._data[idx_min:idx_max+1:average_step_px].mean(axis=0)
        return Profile(data, self._step_x, self._width_um)
    
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
        if x > self._width_um:
            raise ValueError("x must not exceed height of surface.")
        
        if average_step is None:
            average_step_px = 1
        else:
            average_step_px = int(average_step / self._step_x)

        # vertical index of profile
        idx = int(x / self._width_um * self._data.shape[1])
        # first index from which a profile is taken for averaging
        idx_min = idx - int(average / 2) * average_step_px
        idx_min = 0 if idx_min < 0 else idx_min
        # last index from which a profile is taken for averaging
        idx_max = idx + int(average / 2) * average_step_px
        idx_max = self._data.shape[1] if idx_max > self._data.shape[1] else idx_max
        data = self._data[:,idx_min:idx_max+1:average_step_px].mean(axis=1)
        return Profile(data, self._step_y, self._height_um)
    
    def get_oblique_profile(self, x0, y0, x1, y1):
        x0px = int(x0 / self._width_um * self._data.shape[1])
        y0px = int(y0 / self._height_um * self._data.shape[0])
        x1px = int(x1 / self._width_um * self._data.shape[1])
        y1px = int(y1 / self._height_um * self._data.shape[0])

        if (not(0 <= x0px <= self._data.shape[1]) or not(0 <= y0px <= self._data.shape[0]) or 
            not(0 <= x1px <= self._data.shape[1]) or not(0 <= y1px <= self._data.shape[0])):
            raise ValueError("Start- and endpoint coordinates must lie within the surface.")

        dx = x1px - x0px
        dy = y1px - y0px

        size = int(np.hypot(dx, dy))

        m = dy/dx
        xp = np.linspace(x0px, x1px, size)
        yp = m * xp

        data = ndimage.map_coordinates(self._data, [yp, xp])

        length_um = np.hypot(dy * self._step_y, dx * self._step_x)
        step = length_um / size
        return Profile(data, step, length_um)

    # Operations #######################################################################################################

    def fill_nonmeasured(self, method='nearest', inplace=False):
        if not self._nonmeasured_points_exist:
            return self
        values = self._data.ravel()
        mask = ~np.isnan(values)

        grid_x, grid_y = np.meshgrid(np.arange(self._data.shape[1]), np.arange(self._data.shape[0]))
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        data_interpolated = griddata(points[mask], values[mask], (grid_x, grid_y), method=method)
        
        if inplace:
            self._data = data_interpolated
            self._nonmeasured_points_exist = False
            return self
        return Surface(data_interpolated, self._step_x, self._step_y, self._width_um, self._height_um)
    
    @no_nonmeasured_points
    def level(self, inplace=False):
        #self.period.cache_clear() # Clear the LRU cache of the period method
        x, y = np.meshgrid(np.arange(self._data.shape[1]), np.arange(self._data.shape[0]))
        # Flatten the x, y, and height_data arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        height_flat = self._data.flatten()
        # Create a design matrix A for linear regression
        A = np.column_stack((x_flat, y_flat, np.ones_like(x_flat)))
        # Use linear regression to fit a plane to the data
        coefficients, _, _, _ = lstsq(A, height_flat)
        # Extract the coefficients for the plane equation
        a, b, c = coefficients
        # Calculate the plane values for each point in the grid
        plane = a * x + b * y + c
        # Subtract the plane from the original height data to level it
        leveled_data = self._data - plane
        if inplace:
            self._data = leveled_data
            return self
        return Surface(leveled_data, self._step_x, self._step_y, self._width_um, self._height_um)
    
    @no_nonmeasured_points
    def rotate(self, angle, inplace=False):
        rotated = ndimage.rotate(self._data, angle, reshape=True)

        aspect_ratio = self._data.shape[0] / self._data.shape[1]
        rotated_aspect_ratio = rotated.shape[0] / rotated.shape[1]

        if aspect_ratio < 1:
            total_height = self._data.shape[0] / rotated_aspect_ratio
        else:
            total_height = self._data.shape[1]

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
        width_um = (self._width_um * pre_comp_cos + self._height_um * pre_comp_sin) * w / nx
        height_um = (self._width_um * pre_comp_sin + self._height_um * pre_comp_cos) * h / ny
        step_y = height_um / rotated_cropped.shape[0]
        step_x = width_um / rotated_cropped.shape[1]

        if inplace:
            self._data = rotated_cropped
            self._step_x = step_x
            self._step_y = step_y
            self._width_um = width_um
            self._height_um = height_um

        return Surface(rotated_cropped, step_x, step_y, width_um, height_um)
    
    @no_nonmeasured_points
    def filter(self, cutoff, *, mode, cutoff2=None, inplace=False):
        """
        Filters the surface by means of Fourier Transform. There a several possible modes of filtering:

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

        #self.period.cache_clear() # Clear the LRU cache of the period method
        if mode == 'both' and inplace:
            raise ValueError("Mode 'both' does not support inplace operation since two Surface objects will be returned")
        freq_x = np.fft.fftfreq(self._data.shape[1], d=self._step_x)
        freq_target = 1/cutoff
        cutoff1 = np.argmax(freq_x > freq_target)
        if np.abs(freq_target - freq_x[cutoff1]) > np.abs(freq_target - freq_x[cutoff1-1]):
            cutoff1 -= 1
        if mode == 'bandpass':
            if cutoff2 is None:
                raise ValueError("cutoff2 must be provided.")
            if cutoff2 <= cutoff:
                raise ValueError("The value of cutoff2 must be greater than the value of cutoff.")
            freq_target2 = 1/cutoff2
            cutoff2 = np.argmax(freq_x > freq_target2)
            if np.abs(freq_target2 - freq_x[cutoff2]) > np.abs(freq_target2 - freq_x[cutoff2-1]):
                cutoff2 -= 1
            
        fft = np.fft.fftshift(np.fft.fft2(self._data))
        rows, cols = self._data.shape
        
        
        if mode == 'bandpass':
            filter_highpass = np.ones((rows, cols))
            filter_highpass[rows//2-cutoff2:rows//2+cutoff2, cols//2-cutoff2:cols//2+cutoff2] = 0
            
            filter_lowpass = np.ones((rows, cols))
            filter_lowpass[rows//2-cutoff1:rows//2+cutoff1, cols//2-cutoff1:cols//2+cutoff1] = 0
            filter_lowpass = ~filter_lowpass.astype('bool')
            zfiltered_band = np.fft.ifft2(np.fft.ifftshift(fft * filter_highpass * filter_lowpass)).real
            if inplace:
                self._data = zfiltered_band
                return self
            return Surface(zfiltered_band, self._step_x, self._step_y, self._width_um, self._height_um)

        filter_highpass = np.ones((rows, cols))
        filter_highpass[rows//2-cutoff1:rows//2+cutoff1, cols//2-cutoff1:cols//2+cutoff1] = 0
        filter_lowpass = ~filter_highpass.astype('bool')
            
        zfiltered_high = np.fft.ifft2(np.fft.ifftshift(fft * filter_highpass)).real
        zfiltered_low = np.fft.ifft2(np.fft.ifftshift(fft * filter_lowpass)).real

        if mode == 'both':
            surface_high = Surface(zfiltered_high, self._step_x, self._step_y, self._width_um, self._height_um)
            surface_low = Surface(zfiltered_low, self._step_x, self._step_y, self._width_um, self._height_um)
            return surface_high, surface_low
        if mode == 'highpass':
            if inplace:
                self._data = zfiltered_high
                return self
            surface_high = Surface(zfiltered_high, self._step_x, self._step_y, self._width_um, self._height_um)
            return surface_high
        if mode == 'lowpass':
            if inplace:
                self._data = zfiltered_low
                return self
            surface_low = Surface(zfiltered_low, self._step_x, self._step_y, self._width_um, self._height_um)
            return surface_low
        
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
        y, x = self._data.shape
        xn, yn = int(x / factor), int(y / factor)
        data = self._data[int((x - xn) / 2):xn + int((x - xn) / 2) + 1, int((y - yn) / 2):yn + int((y - yn) / 2) + 1]
        width_um = self._width_um * xn/x
        height_um = self._height_um * yn/y
        if inplace:
            self._data = data
            self._width_um = width_um
            self._height_um = height_um
            return self
        return Surface(data, self._step_x, self._step_y, width_um, height_um)
    
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
        data = self._data - self._data.mean()
        fft = np.abs(np.fft.fftshift(np.fft.fft2(data)))
        N, M = self._data.shape
        # Calculate the frequency values for the x and y axes
        freq_x = np.fft.fftshift(np.fft.fftfreq(M, d=self._width_um/M))  # Frequency values for the x-axis
        freq_y = np.fft.fftshift(np.fft.fftfreq(N, d=self._height_um/N))  # Frequency values for the y-axis
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
        peaks_2d = np.unravel_index(peaks_sorted, fft.shape)
        
        # Does this really make sense? I think I mixed up x and y but it works which is weird
        dy = freq_x[peaks_2d[1][0]] - freq_x[peaks_2d[1][1]]
        dx = freq_y[peaks_2d[0][0]] - freq_y[peaks_2d[0][1]]
        
        return dx, dy

    # Characterization #################################################################################################

    #@lru_cache
    @no_nonmeasured_points
    def period(self):
        dx, dy = self._get_fourier_peak_dx_dy()
        period = 2/np.hypot(dx, dy)
        return period
    
    @no_nonmeasured_points
    def orientation(self):
        dx, dy = self._get_fourier_peak_dx_dy()
        angle = np.rad2deg(np.arctan2(np.abs(dx), np.abs(dy)))
        
        #TODO: This needs to be fixed!
        if dy < 0:
            angle *= -1
        return angle
    
    def projected_area(self):
        return (self._width_um - self._step_x) * (self._height_um - self._step_y)
    
    @no_nonmeasured_points
    def surface_area(self):
        if not CYTHON_DEFINED:
            raise NotImplementedError("Surface area calculation is based on cython code. Compile cython code to run this"
                                      "method")
        return surface_area(self._data, self._step_x, self._step_y)
    
    @no_nonmeasured_points
    def Sa(self):
        return (np.abs(self._data - self._data.mean()).sum() / self._data.size)
    
    @no_nonmeasured_points
    def Sq(self):
        return np.sqrt(((self._data - self._data.mean()) ** 2).sum() / self._data.size).round(8)
    
    @no_nonmeasured_points
    def Sp(self):
        return (self._data - self._data.mean()).max()
    
    @no_nonmeasured_points
    def Sv(self):
        return np.abs((self._data - self._data.mean()).min())
    
    @no_nonmeasured_points
    def Sz(self):
        return self.Sp() + self.Sv()
    
    @no_nonmeasured_points
    def Ssk(self):
        return ((self._data - self._data.mean()) ** 3).sum() / self._data.size / self.Sq()**3
    
    @no_nonmeasured_points
    def Sku(self):
        return ((self._data - self._data.mean()) ** 4).sum() / self._data.size / self.Sq()**4
    
    @no_nonmeasured_points
    def Sdr(self):
        return (self.surface_area() / self.projected_area() -1) * 100
    
    # EXPERIMENTAL -> Does not work correctly
    @no_nonmeasured_points
    def _Sdq(self):
        A = self._data.shape[0] * self._data.shape[1]
        diff_x = np.diff(self._data, axis=1, append=0) / self._step_x
        diff_y = np.diff(self._data, axis=0, append=0) / self._step_y
        return np.sqrt(np.sum(diff_x**2 + diff_y**2) / A)
    
    @no_nonmeasured_points
    def homogeneity(self):
        period = self.period()
        cell_length = int(period / self._height_um * self._data.shape[0])
        ncells = int(self._data.shape[0] / cell_length) * int(self._data.shape[1] / cell_length)
        sa = np.zeros(ncells)
        ssk = np.zeros(ncells)
        sku = np.zeros(ncells)
        sdr = np.zeros(ncells)
        for i in range(int(self._data.shape[0] / cell_length)):
            for j in range(int(self._data.shape[1] / cell_length)):
                idx = i * int(self._data.shape[1] / cell_length) + j
                data = self._data[cell_length * i:cell_length * (i + 1), cell_length * j:cell_length * (j + 1)]
                cell_surface = Surface(data, self._step_x, self._step_y, cell_length * self._step_x,
                                       cell_length * self._step_y)
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

    @no_nonmeasured_points
    def depth(self, nprofiles=30, sampling_width=0.2, retstd=False, plot=False):
        size, length = self._data.shape
        if nprofiles > size:
            raise ValueError(f'nprofiles cannot exceed the maximum available number of profiles of {size}')

        # Obtain the period estimate from the fourier transform in pixel units
        period_ft_um = self.period()
        # Calculate the number of intervals per profile
        nintervals = int(self._width_um/period_ft_um)
        # Allocate depth array with twice the length of the number of periods to accomodate both peaks and valleys
        # multiplied by the number of sampled profiles
        depths = np.zeros(nprofiles * nintervals)

        # Loop over each profile
        for i in range(nprofiles):
            line = self._data[int(size/nprofiles) * i]
            period_px = _period_from_profile(line)
            xp = np.arange(line.size)
            # Define initial guess for fit parameters
            p0=((line.max() - line.min())/2, period_px, 0, line.mean())
            # Fit the data to the general sine function
            popt, pcov = curve_fit(f, xp, line, p0=p0)
            # Extract the refined period estimate from the sine function period
            period_sin = popt[1]
            # Extract the lateral shift of the sine fit
            x0 = popt[2]

            depths_line = np.zeros(nintervals * 2)

            if plot and i == 4:
                fig, ax = plt.subplots(figsize=(16,4))
                ax.plot(xp, line, lw=1.5, c='k', alpha=0.7)
                ax.plot(xp, f(xp, *popt), c='orange', ls='--')
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
    @no_nonmeasured_points
    def abbott_curve(self):
        zmin = self._data.min()
        zmax = self._data.max()
        hist, bins = np.histogram(self._data, bins=40)
        step = np.abs(bins[0] - bins[1])

        hist = hist / self._data.size * 100
        cumulated = np.cumsum(np.histogram(self._data, bins=500)[0])
        cumulated = cumulated / cumulated.max() * 100

        fig, ax = plt.subplots()
        ax2 = ax.twiny()
        ax.set_box_aspect(1)
        ax.barh(np.arange(bins[0] + step, bins[-1] + step, step), hist, height=step*0.8)
        ax2.plot(cumulated, np.linspace(zmax, zmin, cumulated.size), c='r')

        ax.set_ylim(zmin, zmax)
        ax.set_ylabel('z (µm)')
        ax2.set_xlim(0, 100)

        ax.set_xlabel('Material distribution (%)')
        ax2.set_xlabel('Material ratio (%)')

        plt.show()
    
    def show(self, cmap='jet'):
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad('k')
        fig, ax = plt.subplots(dpi=150)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(self._data, cmap=cmap, extent=(0, self._width_um, 0, self._height_um))
        fig.colorbar(im, cax=cax, label='z [µm]')
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')
        if self._nonmeasured_points_exist:
            handles = [plt.plot([], [], marker='s', c='k', ls='')[0]]
            ax.legend(handles, ['non-measured points'], loc='lower right', fancybox=False, framealpha=1, fontsize=6)
        plt.show()