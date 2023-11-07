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

try:
    # Optional import
    from tqdm import tqdm
except ImportError:
    logger.info('tqdm not found, no progessbars will be shown.')
    # If tqdm is not defined, replace with dummy function
    tqdm = lambda x, *args, **kwargs: x

# Custom imports
from . fileloader import load_file
try:
    from .calculations import surface_area
    CYTHON_DEFINED = True
except ImportError:
    logger.warning('Could not import cythonized code. Surface area calculation unavailable.')
    CYTHON_DEFINED = False

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

#TODO batch image export
class Batch:
    
    def __init__(self, filepaths):
        """
        Initializies the Batch object from filepaths to topography files.

        Parameters
        ----------
        filepaths: list of filepaths
        """
        self._filepaths = [Path(file) for file in filepaths]
        self._load_surfaces()
        
    def _load_surfaces(self):
        self._surfaces = dict()
        for file in tqdm(self._filepaths, desc='Loading files'):
            self._surfaces[file] = Surface.load(file)
            
    def fill_nonmeasured(self, mode='nearest'):
        for surface in tqdm(self._surfaces.values(), desc='Filling non-measured points'):
            surface.fill_nonmeasured(mode=mode, inplace=True)
        return self
            
    def level(self):
        for surface in tqdm(self._surfaces.values(), desc='Leveling'):
            surface.level(inplace=True)
        return self
            
    def filter(self, cutoff, *, mode, cutoff2=None, inplace=False):
        for surface in tqdm(self._surfaces.values(), desc='Filtering'):
            surface.filter(cutoff, mode=mode, cutoff2=None, inplace=True)
        return self

    def roughness_parameters(self, parameters=None):
        if parameters is None:
            parameters = list(Surface.AVAILABLE_PARAMETERS)
        df = pd.DataFrame({'filepath': [file.name for file in self._filepaths]})
        df = df.set_index('filepath')
        df[list(parameters)] = np.nan
        for file, surface in tqdm(self._surfaces.items(), desc='Calculating parameters'):
            results = surface.roughness_parameters(parameters)
            for k, v in results.items():
                df.loc[file.name][k] = v
        return df.reset_index()


class Profile:
    
    def __init__(self, height_data, step, length_um):
        self._data = height_data
        self._step = step
        self._length_um = length_um
        
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
    
    def show(self):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.linspace(0, self._length_um, self._data.size), self._data, c='k')
        plt.show()
        
           
def no_nonmeasured_points(function):
    @wraps(function)
    def wrapper_function(self, *args, **kwargs):
        if self._nonmeasured_points_exist:
            raise ValueError("Non-measured points must be filled before any other operation.")
        return function(self, *args, **kwargs)
    return wrapper_function
            
            
# TODO Rotation function -> Detect orientation using FFT
# TODO Depth function
# TODO Profile function, Average profile function
# TODO Image export
# TODO Potential profile class -> Profile roughness parameters
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
        self.show()
    
    @classmethod
    def load(cls, filepath):
        return cls(*load_file(filepath))
    
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
        self.period.cache_clear() # Clear the LRU cache of the period method
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
    def filter(self, cutoff, *, mode, cutoff2=None, inplace=False):
        """
        Filters the surface by means of Fourier Transform.
        """
        self.period.cache_clear() # Clear the LRU cache of the period method
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
    
    @lru_cache
    @no_nonmeasured_points
    def period(self):
        # Get rid of the zero peak in the DFT for data that features a substantial offset in the z-direction by centering
        # the values around the mean
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
        period = 2/np.hypot(freq_x[peaks_2d[1][0]] - freq_x[peaks_2d[1][1]],
                            freq_y[peaks_2d[0][0]] - freq_y[peaks_2d[0][1]])
        return period
    
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
        return np.abs(self._data - self._data.mean()).sum() / self._data.size
    
    @no_nonmeasured_points
    def Sq(self):
        return np.sqrt(((self._data - self._data.mean()) ** 2).sum() / self._data.size)
    
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
        logger.warning("Homogeneity calculation is possibly wrong!")
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
                data = self._data[cell_length * i:cell_length*(i+1), cell_length * j:cell_length*(j+1)]
                cell_surface = Surface(data, self._step_x, self._step_y, cell_length * self._step_x, cell_length * self._step_y)
                sa[idx] = cell_surface.Sa()
                ssk[idx] = cell_surface.Ssk()
                sku[idx] = cell_surface.Sku()
                sdr[idx] = cell_surface.Sdr()
        sa = np.sort(sa)
        ssk = np.sort(np.abs(ssk))
        sku = np.sort(sku)
        sdr = np.sort(sdr)

        h = []
        for param in (sa, ssk, sku, sdr):
            x, step = np.linspace(0, 1, ncells, retstep=True)
            lorenz = np.cumsum(np.abs(param))
            lorenz = (lorenz - lorenz.min()) / lorenz.max()
            y = lorenz.min() + (lorenz.max() - lorenz.min()) * x
            total = np.trapz(y, dx=step) 
            B = np.trapz(lorenz, dx=step)
            A = total - B
            gini = A / total
            h.append(1 - gini)
        return np.mean(h)
    
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
    
    @no_nonmeasured_points
    def depth(self, nprofiles=30, sampling_width=0.2, retstd=False, plot=False):
        f = lambda x, a, p, xo, yo: a*np.sin((x-xo)/p*2*np.pi) + yo

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
    
    def show(self):
        fig, ax = plt.subplots(dpi=150)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(self._data, cmap='jet', extent=(0, self._width_um, 0, self._height_um))
        fig.colorbar(im, cax=cax, label='z [µm]')
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')                 
        plt.show()