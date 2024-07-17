import numpy as np
import scipy.ndimage as ndimage

class GaussianFilter:
    """
    Constructs a Gaussian filter that can be applied on a topography using filter.apply or the __call__ syntax.

    Parameters
    ----------
    cutoff : float
        Cutoff wavelength.
    filter_type : {'lowpass', 'highpass'}
        Type of filter to apply. For highpass, simply subtracts the lowpass filtered data from the original data.
    endeffect_mode : {reflect, constant, nearest, mirror, wrap}, default reflect
            The parameter determines how the endeffects of the filter at the boundaries of the data are managed.
            For details, see the documentation of scipy.ndimage.gaussian_filter.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

    Examples
    --------
    >>> lowpass_filter = GaussianFilter(1, 'lowpass')
    >>> filtered_surface = lowpass_filter(original_surface)
    """
    def __init__(self, cutoff, filter_type, endeffect_mode='reflect'):
        self._cutoff = cutoff
        if filter_type not in ['lowpass', 'highpass']:
            raise ValueError('f"{filter_type}" is not a valid filter type.')
        self._filter_type = filter_type
        self._endeffect_mode = endeffect_mode

    @staticmethod
    def sigma(cutoff):
        """
        Calculates the standard deviation of the Gaussian kernel from the cutoff value, considering that the cutoff
        wavelength should specify the wavelength where the amplitude transmission is reduced to 50%.

        Parameters
        ----------
        cutoff : float
            Cutoff wavelength.

        Returns
        -------
        sigma : float

        Notes
        -----
        This equation results from solving for the standard deviation when setting the generic Gaussian kernel to the
        Gaussian kernel defined in the norm.
        """
        return cutoff / np.pi * np.sqrt(np.log(2) / 2)

    def __call__(self, surface, inplace=False):
        """
        Applied the filter to a Surface object

        Parameters
        ----------
        surface : Surface
            The surface object on which to apply the filter.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self. Inplace operation is not compatible with mode='both' argument, since two surfalize.Surface
            objects will be returned.

        Returns
        -------
        filtered_surface : Surface
        """
        return self.apply(surface, inplace=inplace)

    def apply(self, surface, inplace=False):
        """
        Applied the filter to a Surface object.

        Parameters
        ----------
        surface : Surface
            The surface object on which to apply the filter.
        inplace : bool, default False
            If False, create and return new Surface object with processed data. If True, changes data inplace and
            return self. Inplace operation is not compatible with mode='both' argument, since two surfalize.Surface
            objects will be returned.

        Returns
        -------
        filtered_surface : Surface
        """
        cutoff_x_px = self._cutoff / surface.step_x
        cutoff_y_px = self._cutoff / surface.step_y
        sigma_x = self.sigma(cutoff_x_px)
        sigma_y = self.sigma(cutoff_y_px)
        data = ndimage.gaussian_filter(surface.data, (sigma_y, sigma_y), mode=self._endeffect_mode)
        if self._filter_type == 'highpass':
            data = surface.data - data
        if inplace:
            surface._set_data(data=data)
            return surface
        # We use surface.__class__ to obtain the class without needing to import it
        # This mitigates a circular import conflict
        return surface.__class__(data, surface.step_x, surface.step_y)