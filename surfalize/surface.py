# Standard imports
import logging
logger = logging.getLogger(__name__)
from functools import wraps
from collections import namedtuple

# Scipy stack
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import lstsq
from scipy.interpolate import griddata
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans

# Custom imports
from .file import load_file, write_file
from .utils import is_list_like, register_returnlabels
from .cache import CachedInstance, cache
from .mathutils import Sinusoid, argclosest, interp1d
from .autocorrelation import AutocorrelationFunction
from .abbottfirestone import AbbottFirestoneCurve
from .profile import Profile
from .filter import GaussianFilter
from .image import Image


size = namedtuple('Size', ['y', 'x'])
           
def no_nonmeasured_points(function):
    """
    Decorator that raises an Exception if the method is called on a surface object that contains non-measured points.
    This decorator should be used for any method that does not compute correctly if nan values are present in the array.

    Parameters
    ----------
    function : function
        Function to be decorated.

    Returns
    -------
    Wrapped function
    """
    @wraps(function)
    def wrapper_function(self, *args, **kwargs):
        if self.has_missing_points:
            raise ValueError("Non-measured points must be filled before any other operation.")
        return function(self, *args, **kwargs)
    return wrapper_function


class Surface(CachedInstance):
    """
    Representation a 2D-topography characterised by a 2d array of height data and an associated stepsize in x and y.
    x hereby denotes the horizontal axis, which corresponds to the second array dimension, while y denotes the vertical
    axis corresponding to the first array dimension. Generally, the code is written with equal stepsize in both axes
    in mind. Undexpected behavior may occur if the stepsize is not equal in both axes. This has to be tested in the
    future.

    The class implements methods to calculate roughness parameters defined in ISO 25178 as well as custom parameters
    for surfaces that exhibit 1D-periodic textures. Moreover, it implements methods for data processing and correction.

    Overview over available ISO-25178 roughness parameters:

    - Height parameters: Sa, Sq, Sz, Sv, Sp, Ssk, Sku
    - Hybrid parameters: Sdr, Sdq
    - Functional parameters: Sk, Svk, Spk, Smr1, Smr2, Sxp, Smr(c), Smc(mr)
    - Functional volume parameters: Vmc, Vmp, Vvc, Vvv
    - Spatial parameters: Sal, Str

    Periodic parameters:
    - Spatial period: computed from Frourier transform
    - Structure depth: computed from n profiles
    - Structure aspect ratio: computed from depth and period
    - Structure homogeneity: computed using Gini coefficient
    - Structure orientation: Angle of the dominant texture towards the vertical axis

    Overview of data operations:
    - Zeroing: Setting lowest height value to zero
    - Centering: Centering height values around the mean
    - Cropping: Cropping to specified borders
    - Zooming: Magnification by factor around the center
    - Rotating: Rotating surface by angle
    - Aligning: Aligning the surface to the dominant texture direction
    - Levelling: Leveling by least squares plane
    - Filtering: Applying lowpass, highpass or bandpass filters
    - Removing outliers: Remove outliers by mean of median filters
    - Thresholding: Thresholding based on areal material ratio
    - Filling non-measured points: Interpolating non-measured points

    Plotting:
    - Surface data
    - Abbott-Firestone curve
    - Fourier Transform

    Parameters
    ----------
    height_data : ndarray
        A 2d numpy array containing the height data
    step_x : float
        Interval between two datapoints in x-axis (horizontal axis, second array dimension)
    step_y : float
        Interval between two datapoints in y-axis (vertical axis, first array dimension)

    Examples
    --------
    Constructing a surface from a 2d array.

    >>> step_x = step_y = 0.1 # mm
    >>> size_x = 200
    >>> size_y = 100
    >>> period = 5
    >>> y, x = np.mgrid[0:size_y:step_y, 0:size_x:step_x]
    >>> height_data = np.sin(x / period * 2 * np.pi)
    >>> surface = Surface(height_data, step_x, step_y)

    Use the load class method to load a topography from a file.

    >>> filepath = r'path\\to\\surface.plu'
    >>> surface = Surface.load(filepath)
    """
    ISO_PARAMETERS = ('Sa', 'Sq', 'Sp', 'Sv', 'Sz', 'Ssk', 'Sku', 'Sdr', 'Sdq', 'Sal', 'Str', 'Sk', 'Spk', 'Svk',
                            'Smr1', 'Smr2', 'Sxp', 'Vmp', 'Vmc', 'Vvv', 'Vvc')
    AVAILABLE_PARAMETERS = ('Sa', 'Sq', 'Sp', 'Sv', 'Sz', 'Ssk', 'Sku', 'Sdr', 'Sdq', 'Sal', 'Str', 'Sk', 'Spk', 'Svk',
                            'Smr1', 'Smr2', 'Sxp', 'Vmp', 'Vmc', 'Vvv', 'Vvc', 'period', 'depth', 'aspect_ratio',
                            'homogeneity', 'stepheight', 'cavity_volume')
    
    def __init__(self, height_data, step_x, step_y, metadata=None, image_layers=None):
        super().__init__() # Initialize cached instance
        self.data = height_data
        self.step_x = step_x
        self.step_y = step_y
        self.metadata = metadata if metadata is not None else {}
        self.image_layers = image_layers if image_layers is not None else {}

        self.width_um = (height_data.shape[1] - 1) * step_x
        self.height_um = (height_data.shape[0] - 1) * step_y

    @property
    def size(self):
        """
        Returns the size of the height data array in pixels as a namedtuple of the form (y, x).
        The elements can be accessed either through indexing or dot notation.

        Returns
        -------
        size : namedtuple(y, x)

        Examples
        --------
        >>> surface.size
        (y=768, x=1024)

        >>> surface.size[0]
        768

        >>> surface.size.y
        768
        """
        return size(*self.data.shape)
            
    def _set_data(self, data=None, step_x=None, step_y=None):
        """
        Overwrites the data of the surface. Used to modify surfaces inplace and recalculate the width_um and height_um
        attributes as well as clear the cache on all lru_cached methods. This method should be used by any method
        that modifys the surface object data inplace.

        Parameters
        ----------
        data : ndarray
        A 2d numpy array containing the height data.
        step_x : float
            Interval between two datapoints in x-axis (horizontal axis, second array dimension).
        step_y : float
            Interval between two datapoints in y-axis (vertical axis, first array dimension).

        Returns
        -------
        None
        """
        if data is not None:
            self.data = data
        if step_x is not None:
            self.step_x = step_x
        if step_y is not None:
            self.step_y = step_y
        self.width_um = (self.size.x - 1) * self.step_x
        self.height_um = (self.size.y - 1) * self.step_y
        self.clear_cache() # Calls method from parent class
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.width_um:.2f} x {self.height_um:.2f} µm²)'
    
    def _repr_png_(self):
        """
        Repr method for Jupyter notebooks. When Jupyter makes a call to repr, it checks first if a _repr_png_ is
        defined. If not, it falls back on __repr__.
        """
        self.show()

    def _arithmetic_operation(self, other, func):
        """
        Generic template method to be used for arithmetic dunder methods.

        Parameters
        ----------
        other : float | Surface
            other operand.
        func : function
            arithmetic function to be applied.

        Returns
        -------
        Surface
        """
        if isinstance(other, Surface):
            if self.step_x != other.step_x or self.step_y != other.step_y or self.size != other.size:
                raise ValueError('Surface objects must have same dimensions and stepsize.')
            return Surface(func(self.data, other.data), self.step_x, self.step_y)
        elif isinstance(other, (int, float)):
            return Surface(func(self.data, other), self.step_x, self.step_y)
        raise ValueError(f'Adding of {type(other)} not supported.')
    def __add__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a+b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a-b)

    __rsub__ = __sub__

    def __mul__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a*b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a/b)

    def __eq__(self, other):
        if not isinstance(other, Surface):
            return False
        if self.step_x != other.step_x or self.step_y != other.step_y or self.size != other.size:
            return False
        if np.any(self.data - other.data > 1e-10):
            return False
        return True

    def __hash__(self):
        return hash((self.step_x, self.step_y, self.size.x, self.size.y, self.data.mean(), self.data.std()))

    @property
    def has_missing_points(self):
        """
        Returns true if surface contains non-measured points.

        Returns
        -------
        bool
        """
        return np.any(np.isnan(self.data))

    @classmethod
    def load(cls, filepath, encoding='utf-8', read_image_layers=False):
        """
        Classmethod to load a topography from a file.

        Parameters
        ----------
        filepath : str | pathlib.Path
            Filepath pointing to the topography file.
        encoding : str, Default utf-8
            Encoding of characters in the file. Defaults to utf-8.
        read_image_layers : bool, Default False
            If true, reads all available image layers in the file and saves them in Surface.image_layers dict

        Returns
        -------
        surface : surfalize.Surface
        """
        raw_surface = load_file(filepath, encoding=encoding, read_image_layers=read_image_layers)
        return cls.from_raw_surface(raw_surface)

    @classmethod
    def from_raw_surface(cls, raw_surface):
        image_layers = {k: Image(v) for k, v in raw_surface.image_layers.items()}
        return cls(raw_surface.data, raw_surface.step_x, raw_surface.step_y, metadata=raw_surface.metadata,
                   image_layers=image_layers)

    def save(self, filepath, encoding='utf-8', **kwargs):
        """
        Saves the surface to a supported file format. The kwargs are specific to individual file formats.

        Parameters
        ----------
        filepath : str | pathlib.Path
            Filepath pointing to the topography file.
        encoding : str, Default utf-8
            Encoding of characters in the file. Defaults to utf-8.

        Optional Parameters
        -------------------
        binary : bool
            Specifies whether to save in the binary version of the format of the ascii version.

        Returns
        -------
        None
        """
        write_file(filepath, self, encoding=encoding, **kwargs)

    def get_image_layer_names(self):
        """
        Returns a list of the names of available image layers.

        Returns
        -------
        List[str]
        """
        return list(self.image_layers.keys())
        
    def get_horizontal_profile(self, y, average=1, average_step=None):
        """
        Extracts a horizontal profile from the surface with optional averaging over parallel profiles.
        Profiles on the edge might be averaged over fewer profiles.

        Parameters
        ----------
        y : float
            vertical (height) value in µm from where the profile is extracted. The value is rounded to the closest data
            point.
        average : int
            number of profiles over which to average. Defaults to 1. Profiles will be extracted above and below the
            position designated by y.
        average_step : float, default None
            distance in µm between parallel profiles used for averaging. The value is rounded to the closest integer
            multiple of the pixel resolution. If the value is None, a distance of 1 px will be assumed.

        Returns
        -------
        profile : surfalize.Profile
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
         x : float
             laterial (width) value in µm from where the profile is extracted. The value is rounded to the closest data
             point.
         average : int
             number of profiles over which to average. Defaults to 1. Profiles will be extracted above and below the
             position designated by x.
         average_step : float, default None
             distance in µm between parallel profiles used for averaging. The value is rounded to the closest integer
             multiple of the pixel resolution. If the value is None, a distance of 1 px will be assumed.

         Returns
         -------
         profile : surfalize.Profile
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

    #TODO: implement averaging
    def get_oblique_profile(self, x0, y0, x1, y1):
        """
         Extracts an oblique profile from the surface.

         Parameters
         ----------
         x0 : float
            starting point of the profile in x.
         y0 : float
            starting point of the profile in y.
         x1 : float
            end point of the profile in x.
         y1 : float
            end point of the profile in y.

         Raises
         ------
         ValueError
            If the points lie outside the definition area.

         Returns
         -------
         profile : surfalize.Profile
         """
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
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self. 

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
        data = self.data - np.nanmean(self.data)
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)
    
    def zero(self, inplace=False):
        """
        Sets the minimum height of the surface to zero.

        Parameters
        ----------
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self. 

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
        data = self.data - np.nanmin(self.data)
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
        n : float, default 3
            Number of standard deviations outside of which values are considered outliers if method is 'mean'. If the
            method is 'median', n represents the number of medians distances of the data to its median value.
        method : {'mean', 'median'}, default 'mean'
            Method by which to perform the outlier detection. The default method is mean, which removes outliers outside
            an interval of n standard deviations from the mean. The method 'median' removes outliers outside n median
            distances of the data to its median.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        surface : surfalize.Surface
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
        threshold : float or tuple[float, float], default 0.5
            Percentage threshold value of the material ratio. If threshold is a tuple, the first value represents the
            upper threshold and the second value represents the lower threshold. For example, threshold=0.5 removes the
            uppermost and lowermost 0.5% from the areal material ratio curve. The achieve the same result when
            specifiying the upper and lower threshold explicitly, the tuple passed ton threshold must be (0.5, 0.5)
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
        y = np.sort(self.data[~np.isnan(self.data)])[::-1]
        x = np.arange(1, y.size + 1, 1) / y.size
        if is_list_like(threshold):
            threshold_upper, threshold_lower = threshold
        else:
            threshold_upper, threshold_lower = threshold, threshold
        if threshold_lower + threshold_upper >= 100:
            raise ValueError("Combined threshold is larger than 100%.")
        idx0 = argclosest(threshold_upper / 100, x)
        idx1 = argclosest(1 - threshold_lower / 100, x)
        data = self.data.copy()
        data[(data > y[idx0]) | (data < y[idx1])] = np.nan
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)

    def fill_nonmeasured(self, method='nearest', inplace=False):
        """
        Fills the non-measured points by interpolation.

        Parameters
        ----------
        method : {‘linear’, ‘nearest’, ‘cubic’}, default 'nearest'
            Method by which to perform the interpolation. See scipy.interpolate.griddata for details.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
        if not self.has_missing_points:
            return self
        values = self.data.ravel()
        mask = ~np.isnan(values)

        grid_x, grid_y = np.meshgrid(np.arange(self.size.x), np.arange(self.size.y))
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        data_interpolated = griddata(points[mask], values[mask], (grid_x, grid_y), method=method)
        
        if inplace:
            self._set_data(data=data_interpolated)
            return self
        return Surface(data_interpolated, self.step_x, self.step_y)

    def level(self, return_trend=False, inplace=False):
        """
        Levels the surface by subtraction of a least squares fit plane.

        Parameters
        ----------
        return_trend: bool, default False
            return the trend as a Surface object alongside the detrended surface if True.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        Surface
        """
        return self.detrend_polynomial(degree=1, inplace=inplace, return_trend=return_trend)

    def detrend_polynomial(self, degree=1, inplace=False, return_trend=False):
        """
        Detrend a 2d array of height data using a polynomial surface

        Parameters
        ----------
        degree : int, default 1
            Polynomial degree.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self.
        return_trend: bool, default False
            return the trend as a Surface object alongside the detrended surface if True.

        Returns
        -------
        Surface or tuple of Surfaces
        """
        rows, cols = self.size
        y, x = np.mgrid[:rows, :cols]

        # Normalize coordinates to [-1, 1] range
        x = (x - x.mean()) / x.max()
        y = (y - y.mean()) / y.max()

        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = self.data.flatten()

        # Create design matrix with cross-terms
        A = np.ones((len(x_flat), 1))
        for i in range(1, degree + 1):
            for j in range(i + 1):
                A = np.column_stack((A, (x_flat ** (i - j)) * (y_flat ** j)))

        # Fit polynomial using SVD for improved numerical stability
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)

        # Calculate trend
        trend = np.dot(A, coeffs).reshape(self.size)

        # Subtract trend from data
        detrended = self.data - trend

        if inplace:
            self._set_data(data=detrended)
            return_surface = self
        else:
            return_surface = Surface(detrended, self.step_x, self.step_y)

        if return_trend:
            return return_surface, Surface(trend, self.step_x, self.step_y)
        return return_surface

    
    @no_nonmeasured_points
    def rotate(self, angle, inplace=False):
        """
        Rotates the surface counterclockwise by the specified angle and crops it to largest possible rectangle with
        the same aspect ratio as the original surface that does not contain any invalid points.

        Parameters
        ----------
        angle : float
            Angle in degrees.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
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
    def filter(self, filter_type, cutoff, cutoff2=None, inplace=False, endeffect_mode='reflect'):
        """
        Filters the surface by applying a Gaussian filter.

        There a several types of filtering:

        - 'highpass': computes spatial frequencies above the specified cutoff value
        - 'lowpass': computes spatial frequencies below the specified cutoff value
        - 'both': computes and returns both the highpass and lowpass filtered surfaces
        - 'bandpass': computes frequencies below the specified cutoff value and above the value specified for cutoff2

        The surface object's data can be changed inplace by specifying 'inplace=True' for 'highpass', 'lowpass' and
        'bandpass' mode. For mode='both', inplace=True will raise a ValueError.

        Parameters
        ----------
        filter_type : str
            Mode of filtering. Possible values: 'highpass', 'lowpass', 'both', 'bandpass'.
        cutoff : float
            Cutoff wavelength in µm at which the high and low spatial frequencies are separated.
            Actual cutoff will be rounded to the nearest pixel unit (1/px) equivalent.
        cutoff2 : float | None, default None
            Used only in mode='bandpass'. Specifies the larger cutoff wavelength of the bandpass filter. Must be greater
            than cutoff.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self. Inplace operation is not compatible with mode='both' argument, since two surfalize.Surface
            objects will be returned.
        endeffect_mode : {reflect, constant, nearest, mirror, wrap}, default reflect
            The parameter determines how the endeffects of the filter at the boundaries of the data are managed.
            For details, see the documentation of scipy.ndimage.gaussian_filter.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
        if filter_type not in ('highpass', 'lowpass', 'both', 'bandpass'):
            raise ValueError("Invalid mode selected")
        if filter_type == 'both' and inplace:
            raise ValueError(
                "Mode 'both' does not support inplace operation since two Surface objects will be returned")

        if filter_type == 'bandpass':
            if cutoff2 is None:
                raise ValueError("cutoff2 must be provided.")
            if cutoff2 <= cutoff:
                raise ValueError("The value of cutoff2 must be greater than the value of cutoff.")

            lowpass_filter = GaussianFilter(filter_type='lowpass', cutoff=cutoff, endeffect_mode=endeffect_mode)
            highpass_filter = GaussianFilter(filter_type='highpass', cutoff=cutoff2, endeffect_mode=endeffect_mode)
            return highpass_filter(lowpass_filter(self, inplace=inplace), inplace=inplace)

        if filter_type == 'lowpass':
            lowpass_filter = GaussianFilter(filter_type='lowpass', cutoff=cutoff, endeffect_mode=endeffect_mode)
            return lowpass_filter(self, inplace=inplace)

        if filter_type == 'highpass':
            highpass_filter = GaussianFilter(filter_type='highpass', cutoff=cutoff, endeffect_mode=endeffect_mode)
            return highpass_filter(self, inplace=inplace)

        # If filter_type == 'both' is only remaining option
        highpass_filter = GaussianFilter(filter_type='highpass', cutoff=cutoff, endeffect_mode=endeffect_mode)
        lowpass_filter = GaussianFilter(filter_type='lowpass', cutoff=cutoff, endeffect_mode=endeffect_mode)
        return highpass_filter(self, inplace=False), lowpass_filter(self, inplace=False)

    def zoom(self, factor, inplace=False):
        """
        Magnifies the surface by the specified factor.

        Parameters
        ----------
        factor : float
            Factor by which the surface is magnified
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
        y, x = self.size
        xn, yn = int(x / factor), int(y / factor)
        data = self.data[int((y - yn) / 2):yn + int((y - yn) / 2) + 1, int((x - xn) / 2):xn + int((x - xn) / 2) + 1]
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)

    def crop(self, box, in_units=True, inplace=False):
        """
        Crop the surface to the area specified by the box parameter.

        Parameters
        ----------
        box : tuple[float, float, float, float]
            The crop rectangle, as a (x0, x1, y0, y1) tuple.
        in_units : bool, default True
            If true, the box is interpreted as physical units (µm). If false, the box is interpreted in pixel values.

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
        if in_units:
            x0 = round(box[0] / self.step_x)
            x1 = round(box[1] / self.step_x)
            y1 = self.size.y - round(box[2] / self.step_y) - 1
            y0 = self.size.y - round(box[3] / self.step_y) - 1
        else:
            x0, x1, y0, y1 = box

        if x0 < 0 or y0 < 0 or x1 > self.size.x - 1 or y1 > self.size.y - 1:
            raise ValueError('Box is out of bounds!')

        data = self.data[y0:y1 + 1, x0:x1 + 1]
        if inplace:
            self._set_data(data=data)
            return self
        return Surface(data, self.step_x, self.step_y)

    
    def align(self, axis='y', method='fft_refined', inplace=False):
        """
        Computes the dominant orientation of the surface pattern and alignes the orientation with the horizontal
        or vertical axis.

        Parameters
        ----------
        axis : {'x', 'y'}, default 'y'
            The axis with which to align the texture with.
        method : {'fft_refined', 'fft'}
            Method by which to calculate the orientation. Default is 'fft_refined'. See Surface.orientation for more
            details.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
        if axis not in ('x', 'y'):
            raise ValueError('Invalid axis specified.')
        angle = self.orientation(method=method)
        if axis == 'x':
            angle += 90
        return self.rotate(-angle, inplace=inplace)

    @cache
    def _get_fourier_peak_dx_dy(self):
        """
        Calculates the distance in x and y in spatial frequency length units. The zero peak is avoided by
        centering the data around the mean. This method is used by the period and orientation calculation.

        Returns
        -------
        (dx, dy) : tuple[float,float]
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
        # Sort in descending order by computing sorting indices
        idx_y, idx_x = np.unravel_index(np.argsort(fft.flatten())[::-1], fft.shape)
        # Transform into spatial frequencies in length units
        # If this is not done, the computed angle will be wrong since the frequency per pixel
        # resolution is different in x and y due to the different sampling length!
        peaks_x = freq_x[idx_x]
        peaks_y = freq_y[idx_y]
        # Create peak tuples for ease of use
        peak0 = (peaks_x[0], peaks_y[0])
        peak1 = (peaks_x[1], peaks_y[1])
        # Peak1 should always be to the right of peak0
        if peak0[0] > peak1[0]:
            peak0, peak1 = peak1, peak0

        dx = peak1[0] - peak0[0]
        dy = peak0[1] - peak1[1]

        return dx, dy

    # Stepheight #######################################################################################################

    @cache
    def _stepheight_get_mask(self):
        """
        Uses k-means algorithm to segment the upper and lower surface of a rectangular ablation cavity.
        Returns a numpy array mask which is true for points that belong to the upper surface level.

        Returns
        -------
        np.array[bool]
        """
        flattened_data = self.data.flatten().reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(flattened_data)
        cluster_labels = kmeans.labels_
        mask = cluster_labels.reshape(self.size).astype('bool')
        if self.data[~mask].mean() > self.data[mask].mean():
            mask = ~mask
        return mask

    @cache
    def stepheight_level(self, inplace=False):
        """
        Levels the surface only based on the datapoints from the upper level surface in a rectangular ablation cavity.
        This function is intended to be used when the measurement contains two approximately flat surfaces on two
        different levels.

        Parameters
        ----------
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self

        Returns
        -------
        surface : surfalize.Surface
            Surface object.
        """
        mask = self._stepheight_get_mask()
        x, y = np.meshgrid(np.arange(self.size.x), np.arange(self.size.y))
        x_flat = x[mask]
        y_flat = y[mask]
        height_flat = self.data[mask]
        A = np.column_stack((x_flat, y_flat, np.ones_like(x_flat)))
        # Use linear regression to fit a plane to the data
        coefficients, _, _, _ = lstsq(A, height_flat)
        # Extract the coefficients for the plane equation
        a, b, c = coefficients
        # Calculate the plane values for each point in the grid
        plane = a * x + b * y + c
        plane = plane - plane.mean()
        leveled_data = self.data - plane
        if inplace:
            self._set_data(data=leveled_data)
            return self
        surface = Surface(leveled_data, self.step_x, self.step_y)
        # This is not an ideal solution, but I can't think of a better one without major refactoring
        # If the leveling is not done inplace, we need to somehow transfer the computed mask to the new objhect
        # We are manually creating a cache entry for the new surface object so that we don't have to recompute the mask
        surface.create_cache_entry(surface._stepheight_get_mask, mask, tuple(), dict())
        return surface

    @cache
    def _stepheight_get_upper_lower_median(self):
        """
        Calculates the median value of the upper and lower surfaces in a stepheight calculation for a rectangular
        ablation cavity.

        Returns
        -------
        upper_median, lower_median : (float, flaot)
        """
        mask = self._stepheight_get_mask()
        upper_median = np.median(self.data[mask])
        lower_median = np.median(self.data[~mask])
        return upper_median, lower_median

    @cache
    def stepheight(self):
        """
        Calculates the stepheight of two-level ablation experiment.

        Returns
        -------
        stepheight : float
        """
        upper_median, lower_median = self._stepheight_get_upper_lower_median()
        step_height = upper_median - lower_median
        return step_height

    def cavity_volume(self, threshold=0.50):
        """
        Calculates the cavity volume of a flat surface containing an ablation crater with a leveled bottom plane.

        Parameters
        ----------
        threshold : float, default 0.5
            Percentage threshold value for the cutoff between the upper and lower levels used to determine the area
            inside which the volume is calculated.

        Returns
        -------
        volume : float
        """
        upper_median, lower_median = self._stepheight_get_upper_lower_median()
        stepheight = self.stepheight()
        mask_volume = self.data < upper_median - threshold * (stepheight)
        volume = (upper_median - self.data[mask_volume]).sum() * self.step_x * self.step_y
        return volume

    # Characterization #################################################################################################
   
    # Height parameters ################################################################################################

    @cache
    @no_nonmeasured_points
    def height_parameters(self):
        """
        Calculates the roughness parameters from the height parameter family.
        Returns a dictionary of the height parameters.

        Returns
        -------
        dict[str: float]
        """
        mean = self.data.mean()
        centered_data = self.data - mean
        abs_centered_data = np.abs(centered_data)
        centered_data_sq = abs_centered_data ** 2

        size = self.data.size
        sa = np.sum(abs_centered_data) / size
        sq = np.sqrt(np.sum(centered_data_sq) / size)
        sv = np.abs(centered_data.min())
        sp = centered_data.max()
        sz = sp + sv
        ssk = np.sum(centered_data_sq * centered_data) / size / sq ** 3
        sku = np.sum(centered_data_sq ** 2) / size / sq ** 4
        return {'Sa': sa, 'Sq': sq, 'Sv': sv, 'Sp': sp, 'Sz': sz, 'Ssk': ssk, 'Sku': sku}

    def Sa(self):
        """
        Calcualtes the arithmetic mean height Sa according to ISO 25178-2.

        Returns
        -------
        Sa : float
        """
        return self.height_parameters()['Sa']

    def Sq(self):
        """
        Calcualtes the root mean square height Sq according to ISO 25178-2.

        Returns
        -------
        Sq : float
        """
        return self.height_parameters()['Sq']

    def Sp(self):
        """
        Calcualtes the maximum peak height Sp according to ISO 25178-2.

        Returns
        -------
        Sp : float
        """
        return self.height_parameters()['Sp']

    def Sv(self):
        """
        Calcualtes the maximum pit height Sv according to ISO 25178-2.

        Returns
        -------
        Sv : float
        """
        return self.height_parameters()['Sv']

    def Sz(self):
        """
        Calcualtes the skewness Ssk according to ISO 25178-2.

        Returns
        -------
        Ssk : float
        """
        return self.height_parameters()['Sz']

    def Ssk(self):
        """
        Calcualtes the skewness Ssk according to ISO 25178-2. It is the quotient of the mean cube value of the ordinate
        values and the cube of Sq within a definition area.

        Returns
        -------
        Ssk : float
        """
        return self.height_parameters()['Ssk']

    def Sku(self):
        """
        Calcualtes the kurtosis Sku  according to ISO 25178-2. It is the quotient of the mean quartic value of the
        ordinate values and the fourth power of Sq within a definition area.

        Returns
        -------
        Sku : float
        """
        return self.height_parameters()['Sku']
    
    # Hybrid parameters ################################################################################################

    @cache
    def projected_area(self):
        """
        Calculates the projected surface area.

        Returns
        -------
        projected area : float
        """
        return (self.width_um - self.step_x) * (self.height_um - self.step_y)
    
    @no_nonmeasured_points
    @cache
    def surface_area(self):
        """
        Calculates the surface area of the surface according to the method proposed by ISO 25178 and used by
        MountainsMap, whereby two triangles are spanned between four corner points.

        Returns
        -------
        area : float
        """
        # Calculate differences
        dz_y = np.diff(self.data, axis=0)
        dz_x = np.diff(self.data, axis=1)

        # Calculate cross products
        cross_x = self.step_y * dz_y[:, :-1]
        cross_y = self.step_x * dz_x[:-1, :]
        cross_z = self.step_x * self.step_y

        # Calculate areas using vectorized operations
        areas = 0.5 * np.sqrt(cross_x ** 2 + cross_y ** 2 + cross_z ** 2)

        # Sum up all areas
        total_area = 2 * np.sum(areas)  # Multiply by 2 for both triangles in each quad

        return total_area

    def Sdr(self):
        """
        Calculates the developed interfacial area ratio according to ISO 25178-2.

        Returns
        -------
        area : float
        """
        return (self.surface_area() / self.projected_area() -1) * 100
    
    @no_nonmeasured_points
    @cache
    def Sdq(self):
        """
        Calculates the root mean square gradient Sdq according to ISO 25178-2.

        Returns
        -------
        Sdq : float
        """
        A = self.size.y * self.size.x
        diff_x = np.diff(self.data, axis=1) / self.step_x
        diff_y = np.diff(self.data, axis=0) / self.step_y
        return np.sqrt((np.sum(diff_x**2) + np.sum(diff_y**2)) / A)

    # Spatial parameters ###############################################################################################

    @cache
    def get_autocorrelation_function(self):
        """
        Instantiates and returns an AutocorrelationFunction object. LRU cache is used to return the same object with
        every function call.

        Returns
        -------
        AutocorrelationFunction
        """
        return AutocorrelationFunction(self)

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
        return self.get_autocorrelation_function().Sal(s=s)

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
        return self.get_autocorrelation_function().Str(s=s)
    
    # Functional parameters ############################################################################################
    
    @cache
    def get_abbott_firestone_curve(self):
        """
        Instantiates and returns an AbbottFirestoneCurve object. LRU cache is used to return the same object with
        every function call.

        Returns
        -------
        AbbottFirestoneCurve
        """
        return AbbottFirestoneCurve(self)

    def Sk(self):
        """
        Calculates Sk in µm.

        Returns
        -------
        Sk : float
        """
        return self.get_abbott_firestone_curve().Sk()

    def Spk(self):
        """
        Calculates Spk in µm.

        Returns
        -------
        Spk : float
        """
        return self.get_abbott_firestone_curve().Spk()

    def Svk(self):
        """
        Calculates Svk in µm.

        Returns
        -------
        Svk : float
        """
        return self.get_abbott_firestone_curve().Svk()

    def Smr1(self):
        """
        Calculates Smr1 in %.

        Returns
        -------
        Smr1 : float
        """
        return self.get_abbott_firestone_curve().Smr1()

    def Smr2(self):
        """
        Calculates Smr2 in %.

        Returns
        -------
        Smr2 : float
        """
        return self.get_abbott_firestone_curve().Smr2()

    def Smr(self, c):
        """
        Calculates the ratio of the area of the material at a specified height c (in µm) to the evaluation area.

        Parameters
        ----------
        c : float
            height in µm.

        Returns
        -------
        areal material ratio : float
        """
        return self.get_abbott_firestone_curve().Smr(c)

    def Smc(self, mr):
        """
        Calculates the height (c) in µm for a given areal material ratio (mr).

        Parameters
        ----------
        mr : float
            areal material ratio in %.

        Returns
        -------
        height : float
        """
        return self.get_abbott_firestone_curve().Smc(mr)

    def Sxp(self, p=2.5, q=50):
        """
        Calculates the difference in height between the p and q material ratio. For Sxp, p and q are defined by the
        standard ISO 25178-3 to be 2.5% and 50%, respectively.

        Parameters
        ----------
        p : float
            material ratio p in % as defined by the standard ISO 25178-3
        q : float
            material ratio q in % as defined by the standard ISO 25178-3

        Returns
        -------
        Height difference: float
        """
        return self.Smc(p) - self.Smc(q)

    # Functional volume parameters ######################################################################################

    def Vmp(self, p=10):
        """
        Calculates the peak material volume at p. The default value of p is 10% according to ISO-25178-3.

        Parameters
        ----------
        p : float, default 10.
            areal material ratio in %.

        Returns
        -------
        Vmp : float
        """
        return self.get_abbott_firestone_curve().Vmp(p=p)

    def Vmc(self, p=10, q=80):
        """
        Calculates the difference in material volume between the p and q material ratio. The default value of p and q
        are is 10% and 80%, respectively, according to ISO-25178-3.

        Parameters
        ----------
        p : float, default 10.
            areal material ratio in %.
        q : float, default 80.
            areal material ratio in %.

        Returns
        -------
        Vmc : float
        """
        return self.get_abbott_firestone_curve().Vmc(p=p, q=q)

    def Vvv(self, q=80):
        """
        Calculates the dale volume at p material ratio. The default value of p is 80% according to ISO-25178-3.

        Parameters
        ----------
        p : float, default 80.
            areal material ratio in %.

        Returns
        -------
        Vvv : float
        """
        return self.get_abbott_firestone_curve().Vvv(q=q)

    def Vvc(self, p=10, q=80):
        """
        Calculates the difference in void volume between p and q material ratio. The default value of p and q
        are is 10% and 80%, respectively, according to ISO-25178-3.

        Parameters
        ----------
        p : float, default 10.
            areal material ratio in %.
        q : float, default 80.
            areal material ratio in %.

        Returns
        -------
        Vvc : float
        """
        return self.get_abbott_firestone_curve().Vvc(p=p, q=q)

    # Non-standard parameters ##########################################################################################
    
    @cache
    @no_nonmeasured_points
    def period(self) -> float:
        """
        Calculates the 1d spatial period based on the Fourier transform. This can yield unexcepted results if the
        surface contains peaks at lower spatial frequencies than the frequency of the periodic structure to be
        evaluated. It is advised to perform appropriate filtering and leveling to remove waviness before invoking this
        method.

        Returns
        -------
        period : float
        """
        dx, dy = self._get_fourier_peak_dx_dy()
        return 2/np.hypot(dx, dy)

    @cache
    @no_nonmeasured_points
    def period_x_y(self) -> tuple[float, float]:
        """
        Calculates the spatial period along the x and y axes based on the Fourier transform.

        Returns
        -------
        (periodx, periody) : tuple[float, float]
        """
        dx, dy = self._get_fourier_peak_dx_dy()
        periodx = np.inf if dx == 0 else np.abs(2/dx)
        periody = np.inf if dy == 0 else np.abs(2/dy)
        return periodx, periody

    def _orientation_fft(self) -> float:
        """
        Computes the orientation angle of the dominant texture towards the vertical axis from the peaks of the Fourier
        transform.

        Returns
        -------
        angle : float
            Angle of the dominant texture to the vertical axis
        """
        dx, dy = self._get_fourier_peak_dx_dy()
        # Account for special cases
        if dx == np.inf or dx == 0:
            return 90
        if dy == np.inf or dy == 0:
            return 0
        return np.rad2deg(np.arctan(dy / dx))

    def _orientation_refined(self) -> float:
        """
        Computes the orientation angle of the dominant texture towards the vertical axis using a refined algorithm.
        This method is more costly than the FFT-based method, but offers significantly better angular resolution.

        Returns
        -------
        angle : float
            Angle of the dominant texture to the vertical axis
        """
        SAMPLE_RATE_FACTOR = 0.1
        period_x, period_y = self.period_x_y()

        if period_y > period_x:
            # We sample at 0.25 * period because we need to at least obey the Nyquist theorem
            period_px = int(period_x / self.step_x)
            dist_px = int(period_px * SAMPLE_RATE_FACTOR)
            nprofiles = int(self.size.y / dist_px)
            get_profile = lambda idx: self.data[idx]
        else:
            period_px = int(period_y / self.step_y)
            dist_px = int(period_px * SAMPLE_RATE_FACTOR)
            nprofiles = int(self.size.x / dist_px)
            get_profile = lambda idx: self.data[:, idx]

        xfp = np.zeros(nprofiles)
        for i in range(nprofiles):
            profile = get_profile(i * dist_px)
            # We make an initial guess for the fit
            p0 = ((profile.max() - profile.min()) / 2, period_px, 0, profile.mean())
            sinusoid = Sinusoid.from_fit(np.arange(profile.size), profile, p0=p0)
            # This computes the position of the first peak of the sinusoid
            xfp[i] = sinusoid.first_extremum()

        # Now we need to get rid of the points where the first peak jumps by one period
        diff = np.diff(xfp)
        # We compute the absolute difference to the median slope
        # We assume that we have more values that correspond to the correct slope than outliers where the peak
        # jumps back by one period. For this to be true, we need to sample at least a couple points between each
        # periodic interval, which is determined by SAMPLE_RATE_FACTOR
        diff_norm = np.abs((diff - np.median(diff)))
        diff_norm = diff_norm / np.abs(diff_norm).max()
        # Here we remove all values that deviate more than 5% from the median slope
        THRESHOLD = 0.05
        d = diff[diff_norm < THRESHOLD].mean()

        # Compute the angle
        angle = np.rad2deg(np.arctan(d / dist_px))
        # If the texture is more aligned with the horizontal axis, we need to correct the angle
        if period_x > period_y:
            angle = np.sign(angle) * 90 - angle
        return angle
    
    @cache
    @no_nonmeasured_points
    def orientation(self, method: str = 'fft_refined') -> float:
        """
        Computes the orientation angle of the dominant texture to the vertical axis in degree. The fft method
        estimates the angle from the peak positions in the 2d Fourier transform. However, the angular resolution for
        low frequencies is quite poor and therefore deviations of up to multiple degree should be expected depending on
        the angle. The fft_refined method refines the estimate from the Fourier transform by sampling profiles along
        the texture, fitting the profiles with a sinusoid and computing the drift of the position of the first peak.
        From this drift, the angle can be obtained with much better precision. The tradeoff is longer computing time.

        Parameters
        ----------
        method : {'fft_refined', 'fft'}
            Method by which to calculate the orientation. Default is 'fft_refined'.

        Returns
        -------
        orientation : float
            Angle of dominant texture to vertical axis in degree.
        """
        if method == 'fft_refined':
            return self._orientation_refined()
        elif method == 'fft':
            return self._orientation_fft()
        raise ValueError('Invalid method specified.')
    
    @no_nonmeasured_points
    @cache
    def homogeneity(self, parameters: tuple[str] = ('Sa', 'Sku', 'Sdr'), period: float = None) -> float:
        """
        Calculates the homogeneity of a periodic surface through Gini coefficient analysis. It returns 1 - Gini, which
        is distributed on in the range between 0 and 1, where 0 represents minimum and 1 represents maximum homogeneity.
        The homogeneity factor is calculated for each roughness parameter specified in 'parameters' and the mean value
        is returned. The surface is divided into square unit cells with a side length equivalent to the period, for
        which each parameter is evaluated.

        Parameters
        ----------
        parameters : tuple[str], optional
            Roughness parameters that are evaluated for their homogeneity distribution. Defaults to ['Sa', 'Sku', Sdr'].
        period : None | float, optional
            The period which is used to devide the surface into unit cells. If None, the period is automatically
            computed from the fourier transform.

        Returns
        -------
        Homogeneity : float
            Value between 0 and 1.

        Notes
        -----
        The algoritm used by this function was proposed by Lechthaler et al. [1]_ and parctically applied by Soldera et
        al. [2]_. Note that only surface rougness parameters which do not yield negative number qualify for the Gini
        analysis  (e.g. the skewness 'Ssk' is not a valid input).

        References
        ----------
        .. [1] Lechthaler, B., Pauly, C. & Mücklich, F. Objective homogeneity quantification of a periodic surface using
           the Gini coefficient. Sci Rep 10, 14516 (2020). https://doi.org/10.1038/s41598-020-70758-9

        .. [2] Soldera, S., Reichel, C., Kuisat, F., Lasagni, A. F. Topography Analysis and Homogeneity Quantification
           of Laser-Patterned Periodic Surface Structures. JLMN 17, 81 (2022). https://doi.org/0.2961/jlmn.2022.02.2002
        """
        DISALLOWED_PARAMETERS = {'Ssk', 'depth', 'homogeneity', 'orientation', 'aspect_ratio', 'period'}
        if params := set(parameters) & DISALLOWED_PARAMETERS:
            raise ValueError('Parameter{} {} {} not allowed for homogeneity calculation.'.format(
                's' if len(params) > 1 else '', ", ".join(params), "are" if len(params) > 1 else "is")
            )

        if period is None:
            period = self.period()

        cell_length_x = int(period / self.step_x)
        cell_length_y = int(period / self.step_y)
        ncells_x = int(self.size.x / cell_length_x)
        ncells_y = int(self.size.y / cell_length_y)
        ncells = ncells_x * ncells_y
        results = np.zeros((len(parameters), ncells))
        for i in range(ncells_y):
            for j in range(ncells_x):
                idx = i * int(ncells_x) + j
                data = self.data[cell_length_y * i:cell_length_y * (i + 1), cell_length_x * j:cell_length_x * (j + 1)]
                cell_surface = Surface(data, self.step_x, self.step_y)
                for k, parameter in enumerate(parameters):
                    results[k, idx] = getattr(cell_surface, parameter)()

        results = np.sort(results.round(8), axis=1)
        h = []
        for i in range(len(parameters)):
            lorenz = np.zeros(ncells + 1)
            lorenz[1:] = np.cumsum(results[i]) / np.sum(results[i])
            x, step = np.linspace(0, 1, lorenz.size, retstep=True)
            y = lorenz.min() + (lorenz.max() - lorenz.min()) * x
            B = np.trapz(lorenz, dx=step)
            A = 0.5 - B
            gini = A / 0.5
            h.append(1 - gini)
        return np.mean(h).round(4)

    @register_returnlabels(('mean', 'std'))
    @cache
    @no_nonmeasured_points
    def depth(self, nprofiles: int = 30, sampling_width: float = 0.2, plot: int = None) -> tuple[float, float]:
        """
        Calculates the peak-to-valley depth of a periodically grooved surface texture. It samples a specified number
        of equally spaced apart profiles from the surface and fits them with a sinusoid. It then evaluates the actual
        profile data in a specified interval around the minima and maxima of the sinusoid and computes their median
        value to reduce the influence of outliers. It then computes the depth by taking the absoulte distance between
        two adjacent maxima and minima. The overall depth is then calculated as the mean of all peak-to-valley depths
        over all sampled profiles.

        Parameters
        ----------
        nprofiles : int, default 30
            Number of profiles to sample from the surface.
        sampling_width : float, default 0.2
            Sampling width around the extrema of the sinusoid as a fraction of the spatial period.
        plot : None | list-like[int], default None
            List of number of profiles to plot.

        Returns
        -------
        Mean depth and standard deviation : tuple[float, float].
        """
        # Check if alignment is more vertical or horizontal
        aligned_vertically = True if -45 < self.orientation(method='fft') < 45 else False
        size = self.size
        if aligned_vertically and nprofiles > size.y:
            raise ValueError(f'nprofiles cannot exceed the maximum available number of profiles of {size.y}')
        if not aligned_vertically and nprofiles > size.x:
            raise ValueError(f'nprofiles cannot exceed the maximum available number of profiles of {size.x}')
        # Obtain the period estimate from the fourier transform in pixel units
        periodx, periody = self.period_x_y()
        # If the texture is more vertically aligned, we take the period in x, else in y
        if aligned_vertically:
            # Calculate the number of intervals per profile
            nintervals = int(self.width_um / periodx)
            period_px = periodx / self.step_x
            profile_dist_px = int(size.y / (nprofiles-1))
        else:
            nintervals = int(self.height_um / periody)
            period_px = periody / self.step_y
            profile_dist_px = int(size.x / (nprofiles-1))

        # Allocate depth array with twice the length of the number of periods to accommodate both peaks and valleys
        # multiplied by the number of sampled profiles
        depths = np.zeros(nprofiles * nintervals)

        # Loop over each profile
        for i in range(nprofiles):
            if aligned_vertically:
                line = self.data[profile_dist_px * i]
            else:
                line = self.data[:, profile_dist_px * i]
            xp = np.arange(line.size)
            # Define initial guess for fit parameters
            p0=((line.max() - line.min())/2, period_px, 0, line.mean())
            # Fit the data to the general sine function
            sinusoid = Sinusoid.from_fit(xp, line, p0=p0)
            first_extremum = sinusoid.first_extremum()
            # Allocate depth array for line
            depths_line = np.zeros(nintervals * 2)

            if plot and i in plot:
                fig, ax = plt.subplots(figsize=(16,4))
                ax.plot(xp, line, lw=1.5, c='k', alpha=0.7)
                ax.plot(xp, sinusoid(xp), c='orange', ls='--')
                ax.set_xlim(xp.min(), xp.max())

            # Loop over each interval
            for j in range(nintervals*2):
                #idx = (0.25 + 0.5*j) * sinusoid.period + sinusoid.x0
                idx = (0.5 * j) * sinusoid.period + first_extremum

                idx_min = int(idx) - int(sinusoid.period * sampling_width/2)
                idx_max = int(idx) + int(sinusoid.period * sampling_width/2)
                if idx_min < 0 or idx_max > line.size-1:
                    depths_line[j] = np.nan
                    continue
                depth_mean = line[idx_min:idx_max+1].mean()
                depth_median = np.median(line[idx_min:idx_max+1])
                depths_line[j] = depth_median
                # For plotting
                if plot and i in plot:
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

        return np.nanmean(depths), np.nanstd(depths)

    @cache
    def aspect_ratio(self) -> float:
        """
        Calculates the aspect ratio of a periodic texture as the ratio of the structure depth and the structure period.

        Returns
        -------
        aspect_ratio : float
        """
        return self.depth()[0] / self.period()

    def roughness_parameters(self, parameters: list[str] = None) -> dict[str: float]:
        """
        Computes multiple roughness parameters at once and returns them in a dictionary.

        Examples
        --------

        >>> surface.roughness_parameters(['Sa', 'Sq', 'Sz'])
        {'Sa': 1.23, 'Sq': 1.87, 'Sz': 2.51}

        Parameters
        ----------
        parameters : list-like[str], default None
            List-like object of parameters to evaluate. If None, all available parameters are evaluated.

        Returns
        -------
        parameters : dict[str: float]
        """
        if parameters is None:
            parameters = self.ISO_PARAMETERS
        results = dict()
        for parameter in parameters:
            if parameter in self.AVAILABLE_PARAMETERS:
                results[parameter] = getattr(self, parameter)()
            else:
                raise ValueError(f'Parameter "{parameter}" is undefined.')
        return results

    # Plotting #########################################################################################################
    def plot_abbott_curve(self, nbars: int = 20):
        """
        Plots the Abbott-Firestone curve.

        Parameters
        ----------
        nbars : int
            Number of bars to display for the material density

        Returns
        -------
        None
        """
        abbott_curve = self.get_abbott_firestone_curve()
        return abbott_curve.plot(nbars=nbars)

    def plot_functional_parameter_study(self):
        """
        Plots the Abbott-Firestone curve.

        Parameters
        ----------
        nbars : int
            Number of bars to display for the material density

        Returns
        -------
        None
        """
        abbott_curve = self.get_abbott_firestone_curve()
        return abbott_curve.visual_parameter_study()

    def plot_autocorrelation(self, ax=None, cmap='jet', show_cbar=True):
        acf = self.get_autocorrelation_function()
        return acf.plot_autocorrelation(ax=ax, cmap=cmap, show_cbar=show_cbar)
        
    def plot_fourier_transform(self, log=True, hanning=False, subtract_mean=True, fxmax=None, fymax=None,
                               cmap='inferno', adjust_colormap=True):
        """
        Plots the 2d Fourier transform of the surface. Optionally, a Hanning window can be applied to reduce to spectral
        leakage effects that occur when analyzing a signal of finite sample length.

        Parameters
        ----------
        log : bool, Default True
            Shows the logarithm of the Fourier Transform to increase peak visibility.
        hanning : bool, Default False
            Applys a Hanning window to the data before the transform.
        subtract_mean : bool, Default False
            Subtracts the mean of the data before the transform to avoid the zero peak.
        fxmax : float, Default None
            Maximum frequency displayed in x. The plot will be cropped to -fxmax : fxmax.
        fymax : float, Default None
            Maximum frequency displayed in y. The plot will be cropped to -fymax : fymax.
        cmap : str, Default 'inferno'
            Matplotlib colormap with which to map the data.
        adjust_colormap : bool, Default True
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
            # We add a small offset to avoid ln(0)
            fft = np.log10(fft+ 1e-10)
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

    def plot_2d(self, cmap='jet', maskcolor='black', layer='Topography', ax=None, vmin=None, vmax=None,
                show_cbar=None):
        """
        Creates a 2D-plot of the surface using matplotlib.

        Parameters
        ----------
        cmap : str | mpl.cmap, default 'jet'
            Colormap to apply on the topography layer. Argument has no effect if an image layer is selected.
        maskcolor : str, default 'Black'
            Color for masked values.
        layer : str, default Topography
            Indicate the layer to plot, by default the topography layer is shown. Alternatively, the label of an image
            layer can be indicated.
        ax : matplotlib axis, default None
            If specified, the plot will be drawn the specified axis.
        vmin : float, default None
            Minimum value of the colormap, passed to imshow.
        vmax : float, default None
            Maximum value of the colormap, passed to imshow.
        show_cbar : bool | None, default None
            Determines whether to show a colorbar. If the value is None, the colorbar is shown only for topographies
            and omitted for image data.

        Returns
        -------
        ax.
        """
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(maskcolor)
        if ax is None:
            fig, ax = plt.subplots(dpi=150)
        else:
            fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if layer == 'Topography':
            data = self.data
            if show_cbar is None:
                show_cbar = True
        elif layer in self.image_layers.keys():
            data = self.image_layers[layer].data
            if show_cbar is None:
                show_cbar = False
            if data.ndim == 3:
                cmap = None
            elif data.ndim == 2:
                cmap = 'gray'
        else:
            raise ValueError(f'Layer {layer} does not exist.')
        im = ax.imshow(data, cmap=cmap, extent=(0, self.width_um, 0, self.height_um), vmin=vmin, vmax=vmax)
        if show_cbar:
            fig.colorbar(im, cax=cax, label='z [µm]')
        else:
            cax.axis('off')
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')
        if layer == 'Topography' and self.has_missing_points:
            handles = [plt.plot([], [], marker='s', c=maskcolor, ls='')[0]]
            ax.legend(handles, ['non-measured points'], loc='lower right', fancybox=False, framealpha=1, fontsize=6)
        return ax
    
    def show(self, cmap='jet', maskcolor='black', layer='Topography', ax=None):
        """
        Shows a 2D-plot of the surface using matplotlib.

        Parameters
        ----------
        cmap : str | mpl.cmap, default 'jet'
            Colormap to apply on the topography layer. Argument has no effect if an image layer is selected.
        maskcolor : str, default 'Black'
            Color for masked values.
        layer : str, default Topography
            Indicate the layer to plot, by default the topography layer is shown. Alternatively, the label of an image
            layer can be indicated.
        ax : matplotlib axis, default None
            If specified, the plot will be drawn the specified axis.

        Returns
        -------
        None.
        """
        self.plot_2d(cmap=cmap, maskcolor=maskcolor, layer=layer, ax=ax)
        plt.show()
