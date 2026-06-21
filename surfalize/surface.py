# Standard imports
import logging

from .plotting import plot_3d

logger = logging.getLogger(__name__)
import warnings
from collections import namedtuple

# Scipy stack
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import lstsq
from scipy.interpolate import griddata
import scipy.ndimage as ndimage

# Custom imports
from .file import FileHandler
from .utils import approximately_equal
from .cache import cache
from .mathutils import Sinusoid, Cylinder, trapezoid, otsu_threshold
from .exceptions import FittingError
from .autocorrelation import AutocorrelationFunction
from .feature import FeatureParameters
from .fourier import FourierTransform
from .base import BaseTopography, batch_method, no_nonmeasured_points
from .profile import Profile
from .image import Image
from .mask import Mask


size = namedtuple('Size', ['y', 'x'])

class Surface(BaseTopography):
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
    - Feature parameters: Spd, Svd, Spc, Svc, S5p, S5v, S10z

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
    ISO_PARAMETERS = ('Sa', 'Sq', 'Sp', 'Sv', 'Sz', 'Ssk', 'Sku', 'Sdr', 'Sdq', 'Sal', 'Str', 'Ssw', 'Sk', 'Spk',
                            'Svk', 'Spkx', 'Svkx', 'Sak1', 'Sak2', 'Smr1', 'Smr2', 'Sxp', 'Sdc', 'Vmp', 'Vmc', 'Vvv',
                            'Vvc', 'Spd', 'Svd', 'Spc', 'Svc', 'S5p', 'S5v', 'S10z')
    # Non-standard parameters that are not defined by ISO 25178 but can still be evaluated
    NON_ISO_PARAMETERS = ('period', 'depth', 'aspect_ratio', 'homogeneity', 'stepheight', 'cavity_volume')
    AVAILABLE_PARAMETERS = ISO_PARAMETERS + NON_ISO_PARAMETERS
    
    def __init__(self, height_data, step_x, step_y, metadata=None, image_layers=None, mask=None):
        super().__init__() # Initialize cached instance
        if not approximately_equal(step_x, step_y):
            warnings.warn(
                'The surface has different pixel size in x and y. Some methods might result in incorrect values.'
            )

        self.step_x = step_x
        self.step_y = step_y
        # The mask is created before the data is assigned so that the data setter can consult it. It is empty until a
        # region is masked, so no array is allocated here.
        self.mask = Mask(self)
        # Assigning through the data property stores the array, recomputes width_um/height_um and clears the cache
        self.data = height_data
        if mask is not None:
            self.mask.set(mask, inplace=True)

        self.metadata = metadata if metadata is not None else {}
        self.image_layers = image_layers if image_layers is not None else {}

    def __repr__(self):
        return f'{self.__class__.__name__}({self.width_um:.2f} x {self.height_um:.2f} µm²)'

    @property
    def data(self):
        """
        The 2d height data array. The returned array is a read-only view: mutating it in place
        (e.g. ``surface.data[i, j] = x``) is disallowed because it would not invalidate the cached
        roughness parameters. To edit values in place use ``surface[i, j] = x``, which clears the cache.
        Assigning a new array (``surface.data = new_array``) recomputes width_um/height_um and clears the cache.
        """
        view = self._data.view()
        view.flags.writeable = False
        return view

    @data.setter
    def data(self, value):
        self._data = value
        self.width_um = (value.shape[1] - 1) * self.step_x
        self.height_um = (value.shape[0] - 1) * self.step_y
        # A mask of a different shape becomes meaningless when the data is replaced, so it is reset.
        mask = getattr(self, 'mask', None)
        if mask is not None and not mask.is_empty and mask._array.shape != value.shape:
            warnings.warn('The mask was cleared because the new data has a different shape.')
            mask.clear(inplace=True)
        self.clear_cache()

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
            
    def _set_data(self, data=None, step_x=None, step_y=None, mask=None):
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
        mask : ndarray[bool] | None
            New mask array. Must be provided by shape-changing operations (crop, zoom, rotate) so that the mask stays
            consistent with the new data shape. If None, the existing mask is kept.

        Returns
        -------
        None
        """
        if data is not None:
            # Clear the mask before assigning new data so the shape-mismatch guard in the data setter does not warn and
            # discard a mask that the caller is about to replace anyway.
            if mask is not None:
                self.mask.clear(inplace=True)
            self.data = data
        if mask is not None:
            self.mask.set(mask, inplace=True)
        if step_x is not None:
            self.step_x = step_x
        if step_y is not None:
            self.step_y = step_y
        self.width_um = (self.size.x - 1) * self.step_x
        self.height_um = (self.size.y - 1) * self.step_y
        self.clear_cache() # Calls method from parent class

    def _with_data(self, data, mask=None):
        if mask is None and not self.mask.is_empty:
            mask = self.mask.to_array()
        return Surface(data, self.step_x, self.step_y, metadata=self.metadata, image_layers=self.image_layers,
                       mask=mask)

    def copy(self):
        """
        Returns a deep copy of the surface. The height data and the mask are copied, while metadata and image layers
        are shallow-copied.

        Returns
        -------
        surface : surfalize.Surface
        """
        mask = None if self.mask.is_empty else self.mask.to_array().copy()
        return Surface(self._data.copy(), self.step_x, self.step_y, metadata=dict(self.metadata),
                       image_layers=dict(self.image_layers), mask=mask)

    @property
    def has_masked_points(self):
        """
        Returns true if the surface contains masked points.

        Returns
        -------
        bool
        """
        return not self.mask.is_empty

    @property
    def _invalid(self):
        """
        Boolean array marking points excluded from analysis: non-measured points (NaN) and masked points.

        Returns
        -------
        ndarray[bool]
        """
        if self.mask.is_empty:
            return np.isnan(self._data)
        return np.isnan(self._data) | self.mask.to_array()

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
            mask = None
            if not self.mask.is_empty or not other.mask.is_empty:
                mask = self.mask.to_array() | other.mask.to_array()
            return Surface(func(self.data, other.data), self.step_x, self.step_y, mask=mask)
        elif isinstance(other, (int, float)):
            return self._with_data(func(self.data, other))
        raise ValueError(f'Adding of {type(other)} not supported.')
    def __add__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a+b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a-b)

    def __rsub__(self, other):
        return self._arithmetic_operation(other, lambda a, b: b-a)

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
        # Non-measured points (NaN) must be located at the same positions in both surfaces. Comparing them directly
        # would not work, since any comparison involving NaN evaluates to False and would wrongly mask a difference.
        nan_mask = np.isnan(self.data)
        if not np.array_equal(nan_mask, np.isnan(other.data)):
            return False
        if np.any(np.abs(self.data[~nan_mask] - other.data[~nan_mask]) > 1e-10):
            return False
        return True

    def __hash__(self):
        return hash((self.step_x, self.step_y, self.size.x, self.size.y, self.data.mean(), self.data.std()))

    def __getitem__(self, item):
        step_y = self.step_y
        step_x = self.step_x
        if isinstance(item, slice):
            if item.step is not None:
                step_y = item.step * self.step_y
        if isinstance(item, tuple):
            if isinstance(item[0], slice) and item[0].step is not None:
                step_y = item[0].step * self.step_y
            if isinstance(item[1], slice) and item[1].step is not None:
                step_x = item[1].step * self.step_x
        mask = None if self.mask.is_empty else self.mask.to_array().__getitem__(item)
        return Surface(self._data.__getitem__(item), step_x, step_y, mask=mask)

    @classmethod
    def load(cls, path_or_buffer, format=None, encoding='auto', read_image_layers=False):
        """
        Classmethod to load a topography from a file.

        Parameters
        ----------
        path_or_buffer : str | pathlib.Path | buffer
            Filepath pointing to the topography file or buffer.
        format : str | None
            File format in which file should be read. If the file is provided as a path and does not contain a suffix,
            the format must be specified here. If both a suffix and format are given, the format overrides the suffix.
            If the surface is read from a buffer, the format value must be specified.
        encoding : str, Default auto
            Encoding of characters in the file. If set to 'auto', the encoding is inferred automatically. For file
            formats with fixed encoding (such as ASCII formats), this parameter has no effect. The default value is
            'auto'.
        read_image_layers : bool, Default False
            If true, reads all available image layers in the file and saves them in Surface.image_layers dict

        Returns
        -------
        surface : surfalize.Surface
        """
        raw_surface = FileHandler(path_or_buffer, format_=format).read(encoding=encoding,
                                                                      read_image_layers=read_image_layers)
        return cls.from_raw_surface(raw_surface)

    @classmethod
    def from_raw_surface(cls, raw_surface):
        """
        Classmethod that instantiates a `Surface` object from a `RawSurface` object returned by the file readers.

        Parameters
        ----------
        raw_surface: surfalize.file.common.RawSurface
            Raw surface object.

        Returns
        -------
        surfalize.Surface
        """
        image_layers = {k: Image(v) for k, v in raw_surface.image_layers.items()}
        return cls(raw_surface.data, raw_surface.step_x, raw_surface.step_y, metadata=raw_surface.metadata,
                   image_layers=image_layers)

    def save(self, path_or_buffer, format=None, encoding='utf-8', **kwargs):
        """
        Saves the surface to a supported file format. The kwargs are specific to individual file formats.

        Parameters
        ----------
        path_or_buffer : str | pathlib.Path | buffer
            Filepath pointing to the topography file or buffer.
        format : str | None
            File format in which file should be saved. If the file is provided as a path and does not contain a suffix,
            the format must be specified here. If both a suffix and format are given, the format overrides the suffix.
            If the surface is saved to a buffer, the format value must be specified.
        encoding : str, Default utf-8
            Encoding of characters in the file. Defaults to utf-8.

        Optional Parameters
        -------------------
        binary : bool
            Only for SDF format. Specifies whether to save in the binary version of the format of the ascii version.
        comment : str
            Only for SUR format. Specifies a comment to add to the file header.
        compressed : bool
            Only for SUR format. Specifies whether to use the compressed format. Default is False.
        compression: {'none', 'zlib', 'lzma'}
            Only for SFLZ format. Specifies the type of compression, either none, zlib or lzma.

        Returns
        -------
        None
        """
        FileHandler(path_or_buffer, format_=format).write(self, encoding=encoding, **kwargs)

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
    @batch_method('operation')
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

    @batch_method('operation', fixed={'inplace': True, 'return_trend': False})
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

    @batch_method('operation', fixed={'inplace': True, 'return_trend': False})
    def detrend_polynomial(self, degree=1, inplace=False, return_trend=False):
        """
        Detrend a 2d array of height data using a polynomial surface, handling NaN values

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

        # Exclude invalid points (non-measured and masked) from the fit
        valid = ~self._invalid.flatten()
        x_valid = x_flat[valid]
        y_valid = y_flat[valid]
        z_valid = z_flat[valid]

        # Create design matrix with cross-terms
        A = np.ones((len(x_valid), 1))
        for i in range(1, degree + 1):
            for j in range(i + 1):
                A = np.column_stack((A, (x_valid ** (i - j)) * (y_valid ** j)))

        # Fit polynomial using SVD for improved numerical stability
        coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)

        # Calculate trend for all points
        A_full = np.ones((len(x_flat), 1))
        for i in range(1, degree + 1):
            for j in range(i + 1):
                A_full = np.column_stack((A_full, (x_flat ** (i - j)) * (y_flat ** j)))
        trend = np.dot(A_full, coeffs).reshape(self.size)

        # Subtract trend from data, preserving NaN values
        detrended = np.where(np.isnan(self.data), np.nan, self.data - trend)

        if inplace:
            self._set_data(data=detrended)
            return_surface = self
        else:
            return_surface = self._with_data(detrended)

        if return_trend:
            return return_surface, Surface(trend, self.step_x, self.step_y)
        return return_surface

    @batch_method('operation', fixed={'inplace': True, 'return_trend': False})
    def remove_cylinder(self, radius=None, axis='auto', inplace=False, return_trend=False):
        """
        Removes a cylindrical form from the surface. This is intended for surfaces measured on a cylindrical part, where
        the dominant form is the curvature of the cylinder. A cylinder is fitted to the height data in 3d space by
        treating the position and orientation of the rotation axis (and optionally the radius) as free parameters, and
        the cylinder form is subtracted from the height data.

        The cylinder is fitted by minimizing the squared radial deviations of the measured points from the cylinder
        surface. The form is then removed by subtracting, at each lateral position, the z-height of the fitted cylinder
        directly below or above the measured point, so that the returned surface retains the original lateral grid and
        positive features remain positive.

        If the radius of the cylinder is known (e.g. from the part diameter), it should be passed via the ``radius``
        argument. In that case only the position and orientation of the axis are optimized, which is more robust. If no
        radius is provided, it is estimated from the data and optimized alongside the axis.

        Parameters
        ----------
        radius : float | None, default None
            Known radius of the cylinder in the same lateral units as the surface (typically µm). If provided, the
            radius is held fixed during the fit. If None, the radius is estimated and optimized.
        axis : {'auto', 'x', 'y'}, default 'auto'
            Approximate orientation of the cylinder's rotation axis in the lateral plane, used as the starting point for
            the fit. With 'auto', the axis orientation is inferred from the direction of lowest curvature. The fit
            refines the orientation from this starting point, so small tilts of the axis relative to the chosen
            direction are accounted for.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self.
        return_trend : bool, default False
            Return the trend (the fitted cylinder form) as a Surface object alongside the detrended surface if True.

        Returns
        -------
        Surface or tuple of Surfaces
        """
        if axis not in ('auto', 'x', 'y'):
            raise ValueError("Invalid axis specified. Must be one of 'auto', 'x' or 'y'.")

        y_idx, x_idx = np.mgrid[:self.size.y, :self.size.x]
        X = x_idx * self.step_x
        Y = y_idx * self.step_y
        Z = self.data

        # Reduce the surface to mean profiles along each lateral axis and fit a parabola to estimate the curvature.
        # The cylinder axis is oriented along the direction of lowest curvature (smallest sagitta).
        x_coords = np.arange(self.size.x) * self.step_x
        y_coords = np.arange(self.size.y) * self.step_y
        profile_x = np.nanmean(Z, axis=0)
        profile_y = np.nanmean(Z, axis=1)
        poly_x = np.polyfit(x_coords, profile_x, 2)
        poly_y = np.polyfit(y_coords, profile_y, 2)
        sagitta_x = np.abs(poly_x[0]) * (x_coords[-1] - x_coords[0]) ** 2
        sagitta_y = np.abs(poly_y[0]) * (y_coords[-1] - y_coords[0]) ** 2

        if axis == 'auto':
            axis = 'x' if sagitta_y >= sagitta_x else 'y'

        # The curvature occurs perpendicular to the axis. Use the parabola along the curved direction to estimate the
        # radius and the position/height of the axis.
        if axis == 'x':
            a, b, _ = poly_y
            curved_coords = y_coords
        else:
            a, b, _ = poly_x
            curved_coords = x_coords
        if a == 0:
            raise FittingError('Could not estimate the cylinder curvature from the data. The surface appears flat '
                               'along the curved direction.')
        vertex = -b / (2 * a)
        vertex_height = np.polyval((a, b, poly_y[2] if axis == 'x' else poly_x[2]), vertex)
        radius_guess = radius if radius is not None else 1 / (2 * np.abs(a))
        # For a convex (bulging) surface a < 0 and the axis lies below the apex, for a concave surface a > 0 above it.
        axis_height = vertex_height - radius_guess if a < 0 else vertex_height + radius_guess

        if axis == 'x':
            guess = Cylinder(point=(0, vertex, axis_height), direction=(1, 0, 0), radius=radius_guess)
        else:
            guess = Cylinder(point=(vertex, 0, axis_height), direction=(0, 1, 0), radius=radius_guess)

        valid = ~self._invalid
        points = np.column_stack((X[valid], Y[valid], Z[valid]))
        cylinder = Cylinder.from_fit(points, guess, radius=radius)

        # Build the trend by intersecting vertical lines with the fitted cylinder at every lateral position.
        trend = cylinder.intersect_vertical(X, Y, Z)
        detrended = np.where(np.isnan(Z), np.nan, Z - trend)

        if inplace:
            self._set_data(data=detrended)
            return_surface = self
        else:
            return_surface = self._with_data(detrended)

        if return_trend:
            return return_surface, Surface(trend, self.step_x, self.step_y)
        return return_surface

    @batch_method('operation')
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

    @batch_method('operation')
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
        sy = slice(int((y - yn) / 2), yn + int((y - yn) / 2) + 1)
        sx = slice(int((x - xn) / 2), xn + int((x - xn) / 2) + 1)
        data = self.data[sy, sx]
        mask = None if self.mask.is_empty else self.mask.to_array()[sy, sx]
        if inplace:
            self._set_data(data=data, mask=mask)
            return self
        return Surface(data, self.step_x, self.step_y, mask=mask)

    @batch_method('operation')
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
        mask = None if self.mask.is_empty else self.mask.to_array()[y0:y1 + 1, x0:x1 + 1]
        if inplace:
            self._set_data(data=data, mask=mask)
            return self
        return Surface(data, self.step_x, self.step_y, mask=mask)

    @batch_method('operation')
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

    # Stepheight #######################################################################################################

    @cache
    def _stepheight_get_mask(self):
        """
        Uses Otsu's method to segment the upper and lower surface of a rectangular ablation cavity by thresholding
        the height values into two classes. Returns a numpy array mask which is true for points that belong to the
        upper surface level.

        Returns
        -------
        np.array[bool]
        """
        threshold = otsu_threshold(self.data)
        mask = self.data > threshold
        if self.data[~mask].mean() > self.data[mask].mean():
            mask = ~mask
        return mask

    @batch_method('operation')
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
        ablation cavity. This function is intended to be used when the measurement contains two approximately flat surfaces on two
        different levels.

        Returns
        -------
        upper_median, lower_median : (float, flaot)
        """
        mask = self._stepheight_get_mask()
        upper_median = np.median(self.data[mask])
        lower_median = np.median(self.data[~mask])
        return upper_median, lower_median

    @batch_method('parameter')
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

    @batch_method('parameter')
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
    def height_parameters(self):
        """
        Calculates the roughness parameters from the height parameter family.
        Returns a dictionary of the height parameters. Masked points are excluded from the calculation.

        Returns
        -------
        dict[str: float]
        """
        if self.has_missing_points:
            raise ValueError("Non-measured points must be filled before any other operation.")
        # Use only valid (non-masked) height values. With no mask present this is equivalent to the full data.
        centered_data = self._valid_values()
        mean = centered_data.mean()
        centered_data = centered_data - mean
        abs_centered_data = np.abs(centered_data)
        centered_data_sq = abs_centered_data ** 2

        size = centered_data.size
        sa = np.sum(abs_centered_data) / size
        sq = np.sqrt(np.sum(centered_data_sq) / size)
        sv = np.abs(centered_data.min())
        sp = centered_data.max()
        sz = sp + sv
        ssk = np.sum(centered_data_sq * centered_data) / size / sq ** 3
        sku = np.sum(centered_data_sq ** 2) / size / sq ** 4
        return {'Sa': sa, 'Sq': sq, 'Sv': sv, 'Sp': sp, 'Sz': sz, 'Ssk': ssk, 'Sku': sku}

    @batch_method('parameter')
    def Sa(self):
        """
        Calcualtes the arithmetic mean height Sa according to ISO 25178-2.

        Returns
        -------
        Sa : float
        """
        return self.height_parameters()['Sa']

    @batch_method('parameter')
    def Sq(self):
        """
        Calcualtes the root mean square height Sq according to ISO 25178-2.

        Returns
        -------
        Sq : float
        """
        return self.height_parameters()['Sq']

    @batch_method('parameter')
    def Sp(self):
        """
        Calcualtes the maximum peak height Sp according to ISO 25178-2.

        Returns
        -------
        Sp : float
        """
        return self.height_parameters()['Sp']

    @batch_method('parameter')
    def Sv(self):
        """
        Calcualtes the maximum pit height Sv according to ISO 25178-2.

        Returns
        -------
        Sv : float
        """
        return self.height_parameters()['Sv']

    @batch_method('parameter')
    def Sz(self):
        """
        Calcualtes the skewness Ssk according to ISO 25178-2.

        Returns
        -------
        Ssk : float
        """
        return self.height_parameters()['Sz']

    @batch_method('parameter')
    def Ssk(self):
        """
        Calcualtes the skewness Ssk according to ISO 25178-2. It is the quotient of the mean cube value of the ordinate
        values and the cube of Sq within a definition area.

        Returns
        -------
        Ssk : float
        """
        return self.height_parameters()['Ssk']

    @batch_method('parameter')
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
    @batch_method('parameter')
    @cache
    def projected_area(self):
        """
        Calculates the projected surface area.

        Returns
        -------
        projected area : float
        """
        return (self.width_um - self.step_x) * (self.height_um - self.step_y)

    @batch_method('parameter')
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

    @batch_method('parameter')
    def Sdr(self):
        """
        Calculates the developed interfacial area ratio according to ISO 25178-2.

        Returns
        -------
        area : float
        """
        return (self.surface_area() / self.projected_area() -1) * 100

    @batch_method('parameter')
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
    @batch_method('parameter')
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

    @batch_method('parameter')
    @cache
    def get_fourier_transform(self):
        """
        Instantiates and returns a FourierTransform object. LRU cache is used to return the same object with
        every function call.

        Returns
        -------
        FourierTransform
        """
        return FourierTransform(self)

    @batch_method('parameter')
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

    @batch_method('parameter')
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

    @batch_method('parameter')
    def Sk(self):
        """
        Calculates Sk in µm.

        Returns
        -------
        Sk : float
        """
        return self.get_abbott_firestone_curve().k()

    @batch_method('parameter')
    def Spk(self):
        """
        Calculates Spk in µm.

        Returns
        -------
        Spk : float
        """
        return self.get_abbott_firestone_curve().pk()

    @batch_method('parameter')
    def Svk(self):
        """
        Calculates Svk in µm.

        Returns
        -------
        Svk : float
        """
        return self.get_abbott_firestone_curve().vk()

    @batch_method('parameter')
    def Spkx(self):
        """
        Calculates the maximum peak height Spkx in µm according to ISO 25178-2, i.e. the height of the highest point
        above the core surface before the reduction process.

        Returns
        -------
        Spkx : float
        """
        return self.get_abbott_firestone_curve().pkx()

    @batch_method('parameter')
    def Svkx(self):
        """
        Calculates the maximum pit depth Svkx in µm according to ISO 25178-2, i.e. the depth of the deepest point
        below the core surface before the reduction process.

        Returns
        -------
        Svkx : float
        """
        return self.get_abbott_firestone_curve().vkx()

    @batch_method('parameter')
    def Sak1(self):
        """
        Calculates the area of the hills Sak1 in %·µm according to ISO 25178-2, the area of the triangle obtained
        during the reduction process of the protruding hills.

        Returns
        -------
        Sak1 : float
        """
        return self.get_abbott_firestone_curve().ak1()

    @batch_method('parameter')
    def Sak2(self):
        """
        Calculates the area of the dales Sak2 in %·µm according to ISO 25178-2, the area of the triangle obtained
        during the reduction process of the protruding dales.

        Returns
        -------
        Sak2 : float
        """
        return self.get_abbott_firestone_curve().ak2()

    @batch_method('parameter')
    def Smr1(self):
        """
        Calculates Smr1 in %.

        Returns
        -------
        Smr1 : float
        """
        return self.get_abbott_firestone_curve().mr1()

    @batch_method('parameter')
    def Smr2(self):
        """
        Calculates Smr2 in %.

        Returns
        -------
        Smr2 : float
        """
        return self.get_abbott_firestone_curve().mr2()

    @batch_method('parameter')
    def Smrk1(self):
        """
        Calculates Smrk1 in %, the material ratio of the hills according to ISO 25178-2:2021. This is the parameter
        formerly named Smr1.

        Returns
        -------
        Smrk1 : float
        """
        return self.Smr1()

    @batch_method('parameter')
    def Smrk2(self):
        """
        Calculates Smrk2 in %, the material ratio of the dales according to ISO 25178-2:2021. This is the parameter
        formerly named Smr2.

        Returns
        -------
        Smrk2 : float
        """
        return self.Smr2()

    @batch_method('parameter')
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
        return self.get_abbott_firestone_curve().mr(c)

    @batch_method('parameter')
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
        return self.get_abbott_firestone_curve().mc(mr)

    @batch_method('parameter')
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

    @batch_method('parameter')
    def Sdc(self, p=2.5, q=50):
        """
        Calculates the material ratio height difference Sdc according to ISO 25178-2, the difference in height between
        the p and q material ratio with p < q. This generalizes Sxp; the default values of p and q match Sxp.

        Parameters
        ----------
        p : float, default 2.5
            material ratio p in %.
        q : float, default 50
            material ratio q in %.

        Returns
        -------
        Height difference : float
        """
        return self.get_abbott_firestone_curve().dc(p, q)

    # Feature parameters ###############################################################################################

    @batch_method('parameter')
    @cache
    @no_nonmeasured_points
    def get_feature_parameters(self):
        """
        Instantiates and returns a FeatureParameters object, which performs the watershed segmentation and Wolf pruning
        underlying the ISO 25178-2 feature parameters. The cache returns the same object on every call, so that the
        segmentation is shared between all feature parameters that use the same pruning value.

        Returns
        -------
        FeatureParameters
        """
        return FeatureParameters(self)

    @batch_method('parameter')
    @no_nonmeasured_points
    def Spd(self, pruning=5, exclude_edge=True):
        """
        Calculates the density of peaks Spd according to ISO 25178-2, the number of significant hills per unit area
        (in 1/µm²). Significant hills are obtained by watershed segmentation followed by Wolf pruning.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Sz. The default value of 5 % follows ISO 25178-3.
        exclude_edge : bool, default True
            If True, hills whose region touches the border of the evaluation area are excluded, since such features are
            incomplete. This matches the default behaviour of commercial software such as MountainsMap.

        Returns
        -------
        Spd : float
        """
        return self.get_feature_parameters().Spd(pruning=pruning, exclude_edge=exclude_edge)

    @batch_method('parameter')
    @no_nonmeasured_points
    def Svd(self, pruning=5, exclude_edge=True):
        """
        Calculates the density of pits Svd according to ISO 25178-2, the number of significant dales per unit area
        (in 1/µm²). Significant dales are obtained by watershed segmentation followed by Wolf pruning.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Sz. The default value of 5 % follows ISO 25178-3.
        exclude_edge : bool, default True
            If True, dales whose region touches the border of the evaluation area are excluded, since such features are
            incomplete. This matches the default behaviour of commercial software such as MountainsMap.

        Returns
        -------
        Svd : float
        """
        return self.get_feature_parameters().Svd(pruning=pruning, exclude_edge=exclude_edge)

    @batch_method('parameter')
    @no_nonmeasured_points
    def Spc(self, pruning=5, exclude_edge=True):
        """
        Calculates the arithmetic mean peak curvature Spc according to ISO 25178-2 (in 1/µm), the mean of the local
        mean curvature at the peaks of the significant hills.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Sz. The default value of 5 % follows ISO 25178-3.
        exclude_edge : bool, default True
            If True, hills whose region touches the border of the evaluation area are excluded from the feature set.

        Returns
        -------
        Spc : float
        """
        return self.get_feature_parameters().Spc(pruning=pruning, exclude_edge=exclude_edge)

    @batch_method('parameter')
    @no_nonmeasured_points
    def Svc(self, pruning=5, exclude_edge=True):
        """
        Calculates the arithmetic mean pit curvature Svc according to ISO 25178-2 (in 1/µm), the mean of the local
        mean curvature at the pits of the significant dales. Pits are concave, so Svc is negative.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Sz. The default value of 5 % follows ISO 25178-3.
        exclude_edge : bool, default True
            If True, dales whose region touches the border of the evaluation area are excluded from the feature set.

        Returns
        -------
        Svc : float
        """
        return self.get_feature_parameters().Svc(pruning=pruning, exclude_edge=exclude_edge)

    @batch_method('parameter')
    @no_nonmeasured_points
    def S5p(self, pruning=5, exclude_edge=True):
        """
        Calculates the five-point peak height S5p according to ISO 25178-2 (in µm), the mean of the heights of the five
        highest significant peaks referenced to the mean plane. If fewer than five significant peaks are found, the mean
        is taken over those available.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Sz. The default value of 5 % follows ISO 25178-3.
        exclude_edge : bool, default True
            If True, hills whose region touches the border of the evaluation area are excluded from the feature set.

        Returns
        -------
        S5p : float
        """
        return self.get_feature_parameters().S5p(pruning=pruning, exclude_edge=exclude_edge)

    @batch_method('parameter')
    @no_nonmeasured_points
    def S5v(self, pruning=5, exclude_edge=True):
        """
        Calculates the five-point pit depth S5v according to ISO 25178-2 (in µm), the mean of the depths of the five
        deepest significant pits referenced to the mean plane. If fewer than five significant pits are found, the mean
        is taken over those available.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Sz. The default value of 5 % follows ISO 25178-3.
        exclude_edge : bool, default True
            If True, dales whose region touches the border of the evaluation area are excluded from the feature set.

        Returns
        -------
        S5v : float
        """
        return self.get_feature_parameters().S5v(pruning=pruning, exclude_edge=exclude_edge)

    @batch_method('parameter')
    @no_nonmeasured_points
    def S10z(self, pruning=5, exclude_edge=True):
        """
        Calculates the ten-point height S10z according to ISO 25178-2 (in µm), the sum of the five-point peak height
        S5p and the five-point pit depth S5v.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Sz. The default value of 5 % follows ISO 25178-3.
        exclude_edge : bool, default True
            If True, features whose region touches the border of the evaluation area are excluded from the feature set.

        Returns
        -------
        S10z : float
        """
        return self.get_feature_parameters().S10z(pruning=pruning, exclude_edge=exclude_edge)

    def plot_feature_segmentation(self, kind='dale', pruning=5, exclude_edge=True, ax=None, cmap='jet', save_to=None):
        """
        Plots the watershed segmentation of the surface into significant motifs (hills or dales) used by the feature
        parameters, together with the motif boundaries (ridge/course lines) and critical points (pits/peaks).

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
        save_to : str | pathlib.Path | None
            Path to where the plot should be saved.

        Returns
        -------
        plt.Figure, plt.Axes
        """
        fig, ax = self.get_feature_parameters().plot_segmentation(kind=kind, pruning=pruning,
                                                                  exclude_edge=exclude_edge, ax=ax, cmap=cmap)
        if save_to:
            fig.savefig(save_to, dpi=300, bbox_inches='tight')
        return fig, ax

    # Misc parameters ##################################################################################################

    def Std(self, angle_step=0.5):
        """
        Calculates the texture direction parameter, which is the angle at which the angular power spectrum is the
        largest. It represents the lay of the surface texture.

        Parameters
        ----------
        angle_step : float
            Angular resolution of the power spectrum in degree. Defaults to 0.5

        Returns
        -------
        float
        """
        return self.get_fourier_transform().Std(angle_step=angle_step)

    @batch_method('parameter')
    @no_nonmeasured_points
    def Ssw(self) -> float:
        """
        Calculates the dominant spatial wavelength Ssw according to ISO 25178-2, the wavelength which corresponds to
        the largest absolute value of the Fourier transform of the ordinate values.

        Returns
        -------
        Ssw : float
        """
        return self.get_fourier_transform().dominant_wavelength()

    # Non-standard parameters ##########################################################################################

    @batch_method('parameter')
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
        return self.get_fourier_transform().period()

    @cache
    @no_nonmeasured_points
    def period_x_y(self) -> tuple[float, float]:
        """
        Calculates the spatial period along the x and y axes based on the Fourier transform.

        Returns
        -------
        (periodx, periody) : tuple[float, float]
        """
        return self.get_fourier_transform().period_x_y()

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

    @batch_method('parameter')
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
            return self.get_fourier_transform().orientation()
        raise ValueError('Invalid method specified.')

    @batch_method('parameter')
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
            B = trapezoid(lorenz, dx=step)
            A = 0.5 - B
            gini = A / 0.5
            h.append(1 - gini)
        return np.mean(h).round(4)

    @batch_method('parameter', return_labels=('mean', 'std'))
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

    @batch_method('parameter')
    @cache
    def aspect_ratio(self) -> float:
        """
        Calculates the aspect ratio of a periodic texture as the ratio of the structure depth and the structure period.

        Returns
        -------
        aspect_ratio : float
        """
        return self.depth()[0] / self.period()

    # Plotting #########################################################################################################
    def plot_autocorrelation(self, ax=None, cmap='jet', show_cbar=True, save_to=None):
        """
        Plots the Autocorrelation function.

        Parameters
        ----------
        ax : matplotlib axis, default None
            If specified, the plot will be drawn the specified axis.
        cmap : str | mpl.cmap, default 'jet'
            Colormap to apply on the topography layer. Argument has no effect if an image layer is selected.
        show_cbar : bool | None, default None
            Determines whether to show a colorbar. If the value is None, the colorbar is shown only for topographies
            and omitted for image data.
        save_to : str | pathlib.Path | None
            Path to where the plot should be saved.

        Returns
        -------
        plt.Figure, plt.Axes
        """
        if self.has_missing_points:
            raise ValueError("Non-measured points must be filled before any other operation.")

        acf = self.get_autocorrelation_function()
        fig, ax = acf.plot_autocorrelation(ax=ax, cmap=cmap, show_cbar=show_cbar)
        if save_to:
            fig.savefig(save_to, dpi=300, bbox_inches='tight')
        return fig, ax
        
    def plot_fourier_transform(self, ax=None, log=True, hanning=False, subtract_mean=True, fxmax=None, fymax=None,
                               cmap='inferno', adjust_colormap=True, save_to=None):
        """
        Plots the 2d Fourier transform of the surface. Optionally, a Hanning window can be applied to reduce to spectral
        leakage effects that occur when analyzing a signal of finite sample length.

        Parameters
        ----------
        ax : matplotlib axis, default None
            If specified, the plot will be drawn the specified axis.
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
        save_to : str | pathlib.Path | None
            Path to where the plot should be saved.

        Returns
        -------
        plt.Figure, plt.Axes
        """
        fig, ax = self.get_fourier_transform().plot(
            ax=ax, log=log, hanning=hanning, subtract_mean=subtract_mean, fxmax=fxmax, fymax=fymax,
            cmap=cmap, adjust_colormap=adjust_colormap
        )
        if save_to:
            fig.savefig(save_to, dpi=300, bbox_inches='tight')
        return fig, ax

    def plot_angular_power_spectrum(self, ax=None, angle_step=1):
        return self.get_fourier_transform().plot_angular_power_spectrum(ax=ax, angle_step=angle_step)

    def plot_2d(self, cmap='jet', maskcolor='black', layer='Topography', ax=None, vmin=None, vmax=None,
                show_cbar=None, save_to=None, masked_color='red', masked_alpha=0.5):
        """
        Creates a 2D-plot of the surface using matplotlib.

        Parameters
        ----------
        cmap : str | mpl.cmap, default 'jet'
            Colormap to apply on the topography layer. Argument has no effect if an image layer is selected.
        maskcolor : str, default 'Black'
            Color for non-measured points.
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
        save_to : str | pathlib.Path | None
            Path to where the plot should be saved.
        masked_color : str, default 'red'
            Color of the translucent overlay drawn over masked points on the topography layer.
        masked_alpha : float, default 0.5
            Opacity of the masked-point overlay, between 0 (invisible) and 1 (opaque).

        Returns
        -------
        plt.Figure, plt.Axes
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
        if layer == 'Topography' and self.has_masked_points:
            # Translucent overlay so the underlying height still shows through the masked region.
            overlay = np.zeros((self.size.y, self.size.x, 4))
            overlay[self.mask.to_array()] = to_rgba(masked_color, masked_alpha)
            ax.imshow(overlay, extent=(0, self.width_um, 0, self.height_um), interpolation='nearest')
        if layer == 'Topography' and (self.has_missing_points or self.has_masked_points):
            handles, labels = [], []
            if self.has_missing_points:
                handles.append(plt.plot([], [], marker='s', c=maskcolor, ls='')[0])
                labels.append('non-measured points')
            if self.has_masked_points:
                handles.append(plt.plot([], [], marker='s', c=to_rgba(masked_color, masked_alpha), ls='')[0])
                labels.append('masked points')
            ax.legend(handles, labels, loc='lower right', fancybox=False, framealpha=1, fontsize=6)
        if save_to:
            fig.savefig(save_to, dpi=300, bbox_inches='tight')
        return fig, ax

    def plot_3d(self, vertical_angle=50, horizontal_angle=0, zoom=1, cmap='jet', colorbar=True, show_grid=True,
                light=0.3, light_position=None, crop_white=True, cbar_pad=50, cbar_height=0.5, scale=1,
                level_of_detail=100, save_to=None, interactive=False, window_title='surfalize',
                perspective_projection=True):
        """
        Renders a surface object in 3d using pyvista.

        Parameters
        ----------
        vertical_angle : float
            Angle of the camera in the vertical plane in degree. Defaults to 50.
        horizontal_angle : float
            Angle of the camera in the horizontal plane in degree. Defaults to 0.
        zoom : float
            Zoom factor of the surface render. Defaults to 1. Decreasing the value will zoom out the render.
        cmap : str
            Matplotlib colormap name. Defaults to jet.
        colorbar : bool
            Whether to show a colorbar. Defaults to True.
        show_grid : bool
            Whether to show a grid. Defaults to True.
        light : float
            Intensity of the light from 0 to 1. Defaults to 1.
        light_position : tuple[float, float, float]
            Position of the light source. Defaults to the position of the camera.
        crop_white : bool
            Whether to crop out white image borders in the horizontal axis. Defaults to True.
        cbar_pad : int
            Additional padding of the colorbar from the 3d render in pixels. Defaults to 50.
        cbar_height : float
            Height of the colorbar as a fraction of the image height.
        scale : float
            Vertical scaling factor of the topography. Defaults to 1. Currently, there are issues with the grid rendering
            for scale values other than 1 due to the current pyvista implementation.
        level_of_detail : float
            Level of detail in % by which the topography is downsampled for the 3d plot. A value of 50 will downsample the
            number of points in each axis by a factor of 2. Defaults to 100.
        save_to : str | pathlib.Path | None
            Path to where the plot should be saved.
        interactive : bool
            Specifies whether the plot should be shown in an interactive window. Does not currently work for jupyter.
            Defaults to False.
        window_title : str
            The window title to show in interactive mode. Defaults to 'surfalize'.
        perspective_projection : bool
            Whether to use perspective or parallel projection. Default is True.

        Returns
        -------
        PIL.Image
        """
        if interactive and save_to:
            raise ValueError('Argument "save_to" can only be set for static plots. '
                             'For interactive plots, use the widget save button.')
        image = plot_3d(
            self,
            vertical_angle=vertical_angle,
            horizontal_angle=horizontal_angle,
            zoom=zoom,
            cmap=cmap,
            colorbar=colorbar,
            show_grid=show_grid,
            light=light,
            light_position=light_position,
            crop_white=crop_white,
            cbar_pad=cbar_pad,
            cbar_height=cbar_height,
            scale=scale,
            level_of_detail=level_of_detail,
            interactive=interactive,
            window_title=window_title,
            perspective_projection=perspective_projection
        )
        if save_to:
            image.save(save_to)
        return image

    def show(self, cmap='jet', maskcolor='black', layer='Topography', ax=None, masked_color='red', masked_alpha=0.5):
        """
        Shows a 2D-plot of the surface using matplotlib.

        Parameters
        ----------
        cmap : str | mpl.cmap, default 'jet'
            Colormap to apply on the topography layer. Argument has no effect if an image layer is selected.
        maskcolor : str, default 'Black'
            Color for non-measured points.
        layer : str, default Topography
            Indicate the layer to plot, by default the topography layer is shown. Alternatively, the label of an image
            layer can be indicated.
        ax : matplotlib axis, default None
            If specified, the plot will be drawn the specified axis.
        masked_color : str, default 'red'
            Color of the translucent overlay drawn over masked points on the topography layer.
        masked_alpha : float, default 0.5
            Opacity of the masked-point overlay, between 0 (invisible) and 1 (opaque).

        Returns
        -------
        None.
        """
        self.plot_2d(cmap=cmap, maskcolor=maskcolor, layer=layer, ax=ax, masked_color=masked_color,
                     masked_alpha=masked_alpha)
        plt.show()
