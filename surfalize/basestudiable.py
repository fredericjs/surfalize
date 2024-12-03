from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata

from surfalize import Profile
from surfalize.cache import CachedInstance
from surfalize.filter import GaussianFilter
from surfalize.mathutils import argclosest
from surfalize.utils import is_list_like

size = namedtuple('Size', ['y', 'x'])


class BaseStudiable(CachedInstance):

    def __init__(self, data, step_x, step_y, offset_x=0, offset_y=0, metadata=None, image_layers=None):
        self.data = data
        self.step_x = step_x
        self.step_y = step_y
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.metadata = metadata if metadata is not None else {}
        self.image_layers = image_layers if image_layers is not None else {}

        self.width = data.shape[1] * step_x
        self.height = data.shape[0] * step_y

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

    @property
    def physical_dimensions(self):
        return size(self.height, self.width)

    def _set_data(self, data=None, step_x=None, step_y=None, offset_x=None, offset_y=None):
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
        if offset_x is not None:
            self.offset_x = offset_x
        if offset_y is not None:
            self.offset_y = offset_y
        self.width = self.size.x * self.step_x
        self.height = self.size.y * self.step_y
        self.clear_cache()  # Calls method from parent class
    #
    # def _copy_or_modify(self, data=None, step_x=None, step_y=None, offset_x=None, offset_y=None, copy=True):
    #     if copy:
    #         data = self.data if data is None else data
    #         step_x = self.step_x if step_x is None else step
    #         return self.__class__(self.data, self.step_x)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, item):
        self.data[index] = item

    def __repr__(self):
        return f'{self.__class__.__name__}({self.width:.2f} x {self.height:.2f} µm²)'

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
        if isinstance(other, BaseStudiable):
            if self.step_x != other.step_x or self.step_y != other.step_y or self.size != other.size:
                raise ValueError('Surface objects must have same dimensions and stepsize.')
            return self.__class__(func(self.data, other.data), self.step_x, self.step_y)
        elif isinstance(other, (int, float)):
            return self.__class__(func(self.data, other), self.step_x, self.step_y)
        elif isinstance(other, np.ndarray):
            if self.data.shape == other.shape:
                return self.__class__(func(self.data, other), self.step_x, self.step_y, self.offset_x, self.offset_y)
            raise ValueError (f'Operands have incompatible shapes {self.data.shape} and {other.shape}.')
        raise ValueError(f'Adding of {type(other)} not supported.')

    def __add__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a - b)

    __rsub__ = __sub__

    def __mul__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._arithmetic_operation(other, lambda a, b: a / b)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.step_x != other.step_x or self.step_y != other.step_y or self.size != other.size:
            return False
        if np.any(self.data - other.data > 1e-10):
            return False
        return True

    def __hash__(self):
        return hash((self.step_x, self.step_y, self.size.x, self.size.y, self.offset_x, self.offset_y, self.data.mean(),
                     self.data.std()))

    @property
    def has_missing_points(self):
        """
        Returns true if surface contains non-measured points.

        Returns
        -------
        bool
        """
        return np.any(np.isnan(self.data))

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
        if y - self.offset_y > self.height or y - self.offset_y < 0:
            raise ValueError(f"y must not exceed dimensions of surface which lie between {self.offset_y} and {self.offset_y + self.height}.")

        if average_step is None:
            average_step_px = 1
        else:
            average_step_px = int(average_step / self.step_y)

        # vertical index of profile
        idx = int((y - self.offset_y) / self.height * self.size.y)
        # first index from which a profile is taken for averaging
        idx_min = idx - int(average / 2) * average_step_px
        idx_min = 0 if idx_min < 0 else idx_min
        # last index from which a profile is taken for averaging
        idx_max = idx + int(average / 2) * average_step_px
        idx_max = self.size.y if idx_max > self.size.y else idx_max
        data = self.data[idx_min:idx_max + 1:average_step_px].mean(axis=0)
        return Profile(data, self.step_x, self.width)

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
        if x - self.offset_x > self.width or x - self.offset_x < 0:
            raise ValueError(
                f"x must not exceed dimensions of surface which lie between {self.offset_x} and {self.offset_x + self.width}.")

        if average_step is None:
            average_step_px = 1
        else:
            average_step_px = int(average_step / self.step_x)

        # vertical index of profile
        idx = int((x - self.offset_x) / self.width * self.size.x)
        # first index from which a profile is taken for averaging
        idx_min = idx - int(average / 2) * average_step_px
        idx_min = 0 if idx_min < 0 else idx_min
        # last index from which a profile is taken for averaging
        idx_max = idx + int(average / 2) * average_step_px
        idx_max = self.size.x if idx_max > self.size.x else idx_max
        data = self.data[:, idx_min:idx_max + 1:average_step_px].mean(axis=1)
        return Profile(data, self.step_y, self.height)

    # TODO: implement averaging
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
        x0px = int((x0 - self.offset_x) / self.width * self.size.x)
        y0px = int((y0 - self.offset_y)  / self.height * self.size.y)
        x1px = int((x1 - self.offset_x)  / self.width * self.size.x)
        y1px = int((y1 - self.offset_y)  / self.height * self.size.y)

        if (not (0 <= x0px <= self.size.x) or not (0 <= y0px <= self.size.y) or
                not (0 <= x1px <= self.size.x) or not (0 <= y1px <= self.size.y)):
            raise ValueError("Start- and endpoint coordinates must lie within the surface.")

        dx = x1px - x0px
        dy = y1px - y0px

        size = int(np.hypot(dx, dy))

        m = dy / dx
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
        return self.__class__(data, self.step_x, self.step_y)

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
        return self.__class__(data, self.step_x, self.step_y)

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
        return self.__class__(data, self.step_x, self.step_y)

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
        return self.__class__(data, self.step_x, self.step_y)

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
        return self.__class__(data_interpolated, self.step_x, self.step_y)

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

        # Create mask for non-NaN values
        mask = ~np.isnan(z_flat)
        x_valid = x_flat[mask]
        y_valid = y_flat[mask]
        z_valid = z_flat[mask]

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
            return_surface = self.__class__(detrended, self.step_x, self.step_y)

        if return_trend:
            return return_surface, self.__class__(trend, self.step_x, self.step_y)
        return return_surface

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
        ymin = int((ny - h) / 2) + 1
        ymax = int(ny - (ny - h) / 2) - 1
        xmin = int((nx - w) / 2) + 1
        xmax = int(nx - (nx - w) / 2) - 1

        rotated_cropped = rotated[ymin:ymax + 1, xmin:xmax + 1]
        width_um = (self.width * pre_comp_cos + self.height * pre_comp_sin) * w / nx
        height_um = (self.width * pre_comp_sin + self.height * pre_comp_cos) * h / ny
        step_y = height_um / rotated_cropped.shape[0]
        step_x = width_um / rotated_cropped.shape[1]

        if inplace:
            self._set_data(data=rotated_cropped, step_x=step_x, step_y=step_y)
            return self

        return self.__class__(rotated_cropped, step_x, step_y)

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
        if factor < 1:
            raise ValueError("Factor cannot assume values < 1!")
        y, x = self.size
        xn, yn = int(x / factor), int(y / factor)
        data = self.data[int((y - yn) / 2):yn + int((y - yn) / 2) + 1, int((x - xn) / 2):xn + int((x - xn) / 2) + 1]
        offset_x = self.offset_x + int(x - xn) /2 * self.step_x
        offset_y = self.offset_x + int(y - yn) /2 * self.step_y
        if inplace:
            self._set_data(data=data, offset_x=offset_x, offset_y=offset_y)
            return self
        return self.__class__(data, self.step_x, self.step_y, offset_x, offset_y)

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
            x0 = round((box[0] - self.offset_x) / self.step_x)
            x1 = round((box[1] - self.offset_x) / self.step_x)
            y1 = self.size.y - round((box[2] - self.offset_y) / self.step_y) - 1
            y0 = self.size.y - round((box[3] - self.offset_y) / self.step_y) - 1
            offset_x = box[0]
            offset_y = box[2]
        else:
            x0, x1, y0, y1 = box
            offset_x = self.offset_x + x0 * self.step_x
            offset_y = self.offset_y + (self.size.y - y0) * self.step_y

        if x0 < 0 or y0 < 0 or x1 > self.size.x - 1 or y1 > self.size.y - 1:
            raise ValueError('Box is out of bounds!')

        data = self.data[y0:y1 + 1, x0:x1 + 1]

        if inplace:
            self._set_data(data=data, offset_x=offset_x, offset_y=offset_y)
            return self
        return self.__class__(data, self.step_x, self.step_y, offset_x, offset_y)

    def show(self):
        extent = (self.offset_x, self.offset_x + self.width, self.offset_y, self.offset_y + self.height)
        plt.imshow(self.data, extent=extent, cmap='jet', origin='lower')