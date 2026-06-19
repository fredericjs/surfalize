# Standard imports
import inspect
import textwrap
from functools import wraps

# Scipy stack
import numpy as np

# Custom imports
from .utils import is_list_like
from .cache import CachedInstance, cache
from .mathutils import argclosest
from .abbottfirestone import AbbottFirestoneCurve
from .filter import GaussianFilter


def no_nonmeasured_points(function):
    """
    Decorator that raises an Exception if the method is called on a surface or profile object that contains
    non-measured points. This decorator should be used for any method that does not compute correctly if nan values
    are present in the array.

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
        if self.has_masked_points:
            raise ValueError("This operation is not supported on masked surfaces. Clear the mask first.")
        return function(self, *args, **kwargs)
    return wrapper_function

# Mutable default arg should be no issue here since we don't mutate it. Hopefully I won't change implementation in the
# future and forget about this^^
def batch_method(type_, return_labels=None, batch_doc=None, fixed={'inplace': True}):
    """
    Decorator to mark Surface methods for batch processing.

    Parameters
    ----------
    type_ : str
        Type of batch method ('operation' or 'parameter')
    return_labels : tuple, optional
        Labels for multiple return values (only used for parameters)
    batch_doc : str, optional
        Additional batch-specific documentation.
    fixed : dict[str: Any]
        Keyword arguments that must have a specific value when calling the method from the Batch class. By default, the
        inplace argument is set to True for the Batch method, since returning a copy of the surface object does not make
        sense. Parameters with fixed values are removed from the function signature and docstring of the Batch method.

    Returns
    -------
    Wrapped function
    """

    def decorator(method):
        sig = inspect.signature(method)
        param_names = set(sig.parameters.keys())

        # Filter fixed parameters to only include those in the method signature
        valid_fixed = {k: v for k, v in fixed.items() if k in param_names}

        method._batch_type = type_
        method._fixed = valid_fixed
        if return_labels is not None:
            method.return_labels = return_labels
        if batch_doc is not None:
            method._batch_doc = batch_doc

        # Create the batch availability note
        fixed_params = ', '.join(f'`{param}`' for param in fixed.keys())
        batch_note = '\n.. note:: \n\n\tThis method is available in the Batch class.'
        if valid_fixed:
            batch_note += f' The following parameters are removed in the batch version: {fixed_params}.\n'
        else:
            batch_note += '\n'

        # Append the note to the method's docstring
        if method.__doc__ is None:
            method.__doc__ = batch_note
        else:
            method.__doc__ = textwrap.dedent(method.__doc__) + batch_note

        return method

    return decorator


class BaseTopography(CachedInstance):
    """
    Common base class of Surface and Profile that implements all operations and roughness parameters which are
    agnostic to the dimensionality of the underlying height data. Methods defined here operate solely on the height
    data array and therefore apply identically to profiles (1d) and surfaces (2d).

    Subclasses must implement the methods '_set_data', which overwrites the data inplace and invalidates the cache,
    and '_with_data', which constructs a new instance of the subclass with new height data but otherwise identical
    attributes.
    """

    def _set_data(self, data=None):
        """
        Overwrites the data of the topography inplace, recalculates dependent attributes and clears the cache.
        Must be implemented by the subclass.
        """
        raise NotImplementedError

    def _with_data(self, data):
        """
        Constructs a new instance of the subclass from height data, inheriting the stepsize of this instance.
        Must be implemented by the subclass.
        """
        raise NotImplementedError

    def _repr_png_(self):
        """
        Repr method for Jupyter notebooks. When Jupyter makes a call to repr, it checks first if a _repr_png_ is
        defined. If not, it falls back on __repr__.
        """
        self.show()

    def __setitem__(self, key, value):
        """
        Blessed in-place edit path. Mutates the underlying height data directly and invalidates the cache, since the
        data property itself returns a read-only view to prevent silently stale cached parameters.
        """
        self._data[key] = value
        self.clear_cache()

    @property
    def has_missing_points(self):
        """
        Returns true if the topography contains non-measured points.

        Returns
        -------
        bool
        """
        return np.any(np.isnan(self.data))

    @property
    def has_masked_points(self):
        """
        Returns true if the topography contains masked points. The base implementation always returns False; subclasses
        that support masking (such as `Surface`) override this.

        Returns
        -------
        bool
        """
        return False

    @property
    def _invalid(self):
        """
        Boolean array marking points that are excluded from analysis, i.e. non-measured points. Subclasses that support
        masking extend this to also include masked points.

        Returns
        -------
        ndarray[bool]
        """
        return np.isnan(self._data)

    def _analysis_data(self):
        """
        Returns the height data with all invalid points (non-measured and masked) set to NaN. This is the array that
        reduction-style parameters (which ignore NaN) should operate on. If no point is invalid, the original data is
        returned without copying.

        Returns
        -------
        ndarray
        """
        invalid = self._invalid
        if not invalid.any():
            return self.data
        return np.where(invalid, np.nan, self._data)

    def _valid_values(self):
        """
        Returns a 1d array of the height values at all valid (non-invalid) points.

        Returns
        -------
        ndarray
        """
        return self._data[~self._invalid]

    def min(self):
        """
        Computes the minimum height value, ignoring invalid points.

        Returns
        -------
        float
        """
        return np.nanmin(self._analysis_data())

    def max(self):
        """
        Computes the maximum height value, ignoring invalid points.

        Returns
        -------
        float
        """
        return np.nanmax(self._analysis_data())

    def mean(self):
        """
        Computes the mean height value, ignoring invalid points.

        Returns
        -------
        float
        """
        return np.nanmean(self._analysis_data())

    def median(self):
        """
        Computes the median height value, ignoring invalid points.

        Returns
        -------
        float
        """
        return np.nanmedian(self._analysis_data())

    def std(self):
        """
        Computes the standard deviation of the height values, ignoring invalid points.

        Returns
        -------
        float
        """
        return np.nanstd(self._analysis_data())

    # Operations #######################################################################################################

    @batch_method('operation')
    def center(self, inplace=False):
        """
        Centers the data around its mean value. The height will be distributed equally around 0.

        Parameters
        ----------
        inplace : bool, default False
            If False, create and return new object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        Surface | Profile
        """
        data = self.data - np.nanmean(self._analysis_data())
        if inplace:
            self._set_data(data=data)
            return self
        return self._with_data(data)

    @batch_method('operation')
    def zero(self, inplace=False):
        """
        Sets the minimum height to zero.

        Parameters
        ----------
        inplace : bool, default False
            If False, create and return new object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        Surface | Profile
        """
        data = self.data - np.nanmin(self._analysis_data())
        if inplace:
            self._set_data(data=data)
            return self
        return self._with_data(data)

    @batch_method('operation')
    def invert(self, inplace=False):
        """
        Inverts the topography, creating a negative.

        Parameters
        ----------
        inplace : bool, default False
            If False, create and return new object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        Surface | Profile
        """
        data = self.min() + self.max() - self.data
        if inplace:
            self._set_data(data=data)
            return self
        return self._with_data(data)

    @batch_method('operation')
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
            If False, create and return new object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        Surface | Profile
        """
        ad = self._analysis_data()
        data = self.data.copy()
        if method == 'mean':
            data[np.abs(ad - np.nanmean(ad)) > n * np.nanstd(ad)] = np.nan
        elif method == 'median':
            dist = np.abs(ad - np.nanmedian(ad))
            data[dist > n * np.nanmedian(dist)] = np.nan
        else:
            raise ValueError("Invalid methode.")
        if inplace:
            self._set_data(data=data)
            return self
        return self._with_data(data)

    @batch_method('operation')
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
            uppermost and lowermost 0.5% from the material ratio curve. The achieve the same result when
            specifiying the upper and lower threshold explicitly, the tuple passed ton threshold must be (0.5, 0.5)
        inplace : bool, default False
            If False, create and return new object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        Surface | Profile
        """
        y = np.sort(self._valid_values())[::-1]
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
        return self._with_data(data)

    @batch_method('operation')
    @no_nonmeasured_points
    def filter(self, filter_type, cutoff, cutoff2=None, inplace=False, endeffect_mode='reflect'):
        """
        Filters the topography by applying a Gaussian filter.

        There a several types of filtering:

        - 'highpass': computes spatial frequencies above the specified cutoff value
        - 'lowpass': computes spatial frequencies below the specified cutoff value
        - 'both': computes and returns both the highpass and lowpass filtered topographies
        - 'bandpass': computes frequencies below the specified cutoff value and above the value specified for cutoff2

        The object's data can be changed inplace by specifying 'inplace=True' for 'highpass', 'lowpass' and
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
            If False, create and return new object with processed data. If True, changes data inplace and
            return self. Inplace operation is not compatible with mode='both' argument, since two objects will be
            returned.
        endeffect_mode : {reflect, constant, nearest, mirror, wrap}, default reflect
            The parameter determines how the endeffects of the filter at the boundaries of the data are managed.
            For details, see the documentation of scipy.ndimage.gaussian_filter.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

        Returns
        -------
        Surface | Profile
        """
        if filter_type not in ('highpass', 'lowpass', 'both', 'bandpass'):
            raise ValueError("Invalid mode selected")
        if filter_type == 'both' and inplace:
            raise ValueError(
                "Mode 'both' does not support inplace operation since two objects will be returned")

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

    # Functional parameters ############################################################################################

    @batch_method('parameter')
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

    # Functional volume parameters ######################################################################################

    @batch_method('parameter')
    def Vmp(self, p=10):
        """
        Calculates the peak material volume at p. The default value of p is 10% according to ISO-25178-3.

        Parameters
        ----------
        p : float, default 10.
            material ratio in %.

        Returns
        -------
        Vmp : float
        """
        return self.get_abbott_firestone_curve().vmp(p=p)

    @batch_method('parameter')
    def Vmc(self, p=10, q=80):
        """
        Calculates the difference in material volume between the p and q material ratio. The default value of p and q
        are is 10% and 80%, respectively, according to ISO-25178-3.

        Parameters
        ----------
        p : float, default 10.
            material ratio in %.
        q : float, default 80.
            material ratio in %.

        Returns
        -------
        Vmc : float
        """
        return self.get_abbott_firestone_curve().vmc(p=p, q=q)

    @batch_method('parameter')
    def Vvv(self, q=80):
        """
        Calculates the dale volume at p material ratio. The default value of p is 80% according to ISO-25178-3.

        Parameters
        ----------
        p : float, default 80.
            material ratio in %.

        Returns
        -------
        Vvv : float
        """
        return self.get_abbott_firestone_curve().vvv(q=q)

    @batch_method('parameter')
    def Vvc(self, p=10, q=80):
        """
        Calculates the difference in void volume between p and q material ratio. The default value of p and q
        are is 10% and 80%, respectively, according to ISO-25178-3.

        Parameters
        ----------
        p : float, default 10.
            material ratio in %.
        q : float, default 80.
            material ratio in %.

        Returns
        -------
        Vvc : float
        """
        return self.get_abbott_firestone_curve().vvc(p=p, q=q)

    @batch_method('parameter')
    def Vm(self, p):
        """
        Calculates the material volume per unit area at a given material ratio p (Vm(p)) according to ISO 25178-2.
        Vmp and Vmc are special cases of this general parameter.

        Parameters
        ----------
        p : float
            material ratio in %.

        Returns
        -------
        Vm : float
        """
        return self.get_abbott_firestone_curve().Vm(p)

    @batch_method('parameter')
    def Vv(self, p):
        """
        Calculates the void volume per unit area at a given material ratio p (Vv(p)) according to ISO 25178-2.
        Vvv and Vvc are special cases of this general parameter.

        Parameters
        ----------
        p : float
            material ratio in %.

        Returns
        -------
        Vv : float
        """
        return self.get_abbott_firestone_curve().Vv(p)

    def roughness_parameters(self, parameters=None):
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

    def plot_abbott_curve(self, nbars=20, save_to=None):
        """
        Plots the Abbott-Firestone curve.

        Parameters
        ----------
        nbars : int
            Number of bars to display for the material density
        save_to : str | pathlib.Path | None
            Path to where the plot should be saved.

        Returns
        -------
        plt.Figure, tuple[plt.Axes]
        """
        abbott_curve = self.get_abbott_firestone_curve()
        fig, axs = abbott_curve.plot(nbars=nbars)
        if save_to:
            fig.savefig(save_to, dpi=300, bbox_inches='tight')
        return fig, axs

    def plot_functional_parameter_study(self, save_to=None):
        """
        Plots a visual study of the functional parameters derived from the Abbott-Firestone curve.

        Parameters
        ----------
        save_to : str | pathlib.Path | None
            Path to where the plot should be saved.

        Returns
        -------
        plt.Figure, plt.Axes
        """
        abbott_curve = self.get_abbott_firestone_curve()
        fig, ax = abbott_curve.visual_parameter_study()
        if save_to:
            fig.savefig(save_to, dpi=300, bbox_inches='tight')
        return fig, ax
