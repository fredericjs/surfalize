import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from .base import BaseTopography, no_nonmeasured_points
from .cache import cache
from .mathutils import get_period_fft_1d

class Profile(BaseTopography):
    """
    Representation of a 1D-profile characterised by a 1d array of height data and an associated stepsize along the
    lateral axis.

    The class implements methods to calculate profile roughness parameters defined in ISO 4287/ISO 21920 as well as
    the functional parameters defined in ISO 13565-2. Moreover, it implements methods for data processing and
    correction that are analogous to their Surface counterparts.

    Overview over available roughness parameters:

    - Height parameters: Ra, Rq, Rp, Rv, Rz, Rt, Rsk, Rku
    - Hybrid parameters: Rdq
    - Functional parameters: Rk, Rpk, Rvk, Rmr1, Rmr2, Rxp, Rmr(c), Rmc(mr)
    - Functional volume parameters: Vmp, Vmc, Vvv, Vvc

    Contrary to areal parameters, which are always evaluated over the entire definition area, several profile
    parameters (Rp, Rv, Rz) are defined by ISO 4287 on a sampling length and averaged over the number of sampling
    lengths contained in the evaluation length (typically five). These methods therefore expose an 'n_sections'
    argument that controls the number of sampling lengths the evaluation length is divided into, which defaults to 5
    according to ISO 4287/ISO 4288. To evaluate these parameters on the entire evaluation length instead, specify
    n_sections=1.

    Overview of data operations:

    - Zeroing: Setting lowest height value to zero
    - Centering: Centering height values around the mean
    - Cropping: Cropping to a specified interval
    - Zooming: Magnification by factor around the center
    - Levelling: Leveling by least squares line
    - Detrending: Removal of polynomial trends
    - Filtering: Applying lowpass, highpass or bandpass filters
    - Removing outliers: Remove outliers by mean of median filters
    - Thresholding: Thresholding based on material ratio
    - Filling non-measured points: Interpolating non-measured points

    Parameters
    ----------
    height_data : ndarray
        A 1d numpy array containing the height data.
    step : float
        Interval between two datapoints along the lateral axis.
    length_um : float | None, default None
        Length of the profile in µm. If None, the length is calculated from the number of datapoints and the stepsize.
    """
    ISO_PARAMETERS = ('Ra', 'Rq', 'Rp', 'Rv', 'Rz', 'Rsk', 'Rku', 'Rdq', 'Rk', 'Rpk', 'Rvk', 'Rmr1', 'Rmr2', 'Rxp',
                      'Vmp', 'Vmc', 'Vvv', 'Vvc')
    AVAILABLE_PARAMETERS = ISO_PARAMETERS + ('Rt', 'period')
    # Number of sampling lengths in the evaluation length according to ISO 4287/4288
    DEFAULT_N_SECTIONS = 5

    def __init__(self, height_data, step, length_um=None):
        super().__init__()  # Initialize cached instance
        self.data = height_data
        self.step = step
        if length_um is None:
            length_um = (height_data.shape[0] - 1) * step
        self.length_um = length_um

    def __repr__(self):
        return f'{self.__class__.__name__}({self.length_um:.2f} µm)'

    @property
    def size(self):
        """
        Returns the number of datapoints of the profile.

        Returns
        -------
        int
        """
        return self.data.shape[0]

    def _set_data(self, data=None, step=None):
        """
        Overwrites the data of the profile. Used to modify profiles inplace and recalculate the length_um attribute
        as well as clear the cache on all cached methods. This method should be used by any method that modifies the
        profile object data inplace.

        Parameters
        ----------
        data : ndarray
            A 1d numpy array containing the height data.
        step : float
            Interval between two datapoints along the lateral axis.

        Returns
        -------
        None
        """
        if data is not None:
            self.data = data
        if step is not None:
            self.step = step
        self.length_um = (self.size - 1) * self.step
        self.clear_cache()  # Calls method from parent class

    def _with_data(self, data):
        return Profile(data, self.step)

    # Operations #######################################################################################################

    def fill_nonmeasured(self, method='nearest', inplace=False):
        """
        Fills the non-measured points by interpolation.

        Parameters
        ----------
        method : {‘linear’, ‘nearest’, ‘cubic’}, default 'nearest'
            Method by which to perform the interpolation. See scipy.interpolate.griddata for details.
        inplace : bool, default False
            If False, create and return new Profile object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        profile : surfalize.Profile
        """
        if not self.has_missing_points:
            return self
        indices = np.arange(self.size)
        mask = ~np.isnan(self.data)
        data_interpolated = griddata(indices[mask], self.data[mask], indices, method=method)

        if inplace:
            self._set_data(data=data_interpolated)
            return self
        return self._with_data(data_interpolated)

    def level(self, return_trend=False, inplace=False):
        """
        Levels the profile by subtraction of a least squares fit line.

        Parameters
        ----------
        return_trend : bool, default False
            return the trend as a Profile object alongside the detrended profile if True.
        inplace : bool, default False
            If False, create and return new Profile object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        Profile
        """
        return self.detrend_polynomial(degree=1, inplace=inplace, return_trend=return_trend)

    def detrend_polynomial(self, degree=1, inplace=False, return_trend=False):
        """
        Detrend a 1d array of height data using a polynomial, handling NaN values.

        Parameters
        ----------
        degree : int, default 1
            Polynomial degree.
        inplace : bool, default False
            If False, create and return new Profile object with processed data. If True, changes data inplace and
            return self.
        return_trend : bool, default False
            return the trend as a Profile object alongside the detrended profile if True.

        Returns
        -------
        Profile or tuple of Profiles
        """
        # Normalize coordinates to [-1, 1] range for numerical stability
        x = np.linspace(-1, 1, self.size)
        mask = ~np.isnan(self.data)
        coeffs = np.polynomial.polynomial.polyfit(x[mask], self.data[mask], degree)
        trend = np.polynomial.polynomial.polyval(x, coeffs)

        # Subtract trend from data, preserving NaN values
        detrended = np.where(np.isnan(self.data), np.nan, self.data - trend)

        if inplace:
            self._set_data(data=detrended)
            return_profile = self
        else:
            return_profile = self._with_data(detrended)

        if return_trend:
            return return_profile, self._with_data(trend)
        return return_profile

    def crop(self, interval, in_units=True, inplace=False):
        """
        Crop the profile to the interval specified by the interval parameter.

        Parameters
        ----------
        interval : tuple[float, float]
            The crop interval, as a (x0, x1) tuple.
        in_units : bool, default True
            If true, the interval is interpreted as physical units (µm). If false, the interval is interpreted in
            datapoint indices.
        inplace : bool, default False
            If False, create and return new Profile object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        profile : surfalize.Profile
        """
        if in_units:
            x0 = round(interval[0] / self.step)
            x1 = round(interval[1] / self.step)
        else:
            x0, x1 = interval

        if x0 < 0 or x1 > self.size - 1 or x0 >= x1:
            raise ValueError('Interval is out of bounds!')

        data = self.data[x0:x1 + 1]
        if inplace:
            self._set_data(data=data)
            return self
        return self._with_data(data)

    def zoom(self, factor, inplace=False):
        """
        Magnifies the profile by the specified factor around its center.

        Parameters
        ----------
        factor : float
            Factor by which the profile is magnified.
        inplace : bool, default False
            If False, create and return new Profile object with processed data. If True, changes data inplace and
            return self.

        Returns
        -------
        profile : surfalize.Profile
        """
        n = self.size
        nn = int(n / factor)
        data = self.data[int((n - nn) / 2):nn + int((n - nn) / 2) + 1]
        if inplace:
            self._set_data(data=data)
            return self
        return self._with_data(data)

    # Characterization #################################################################################################

    @cache
    @no_nonmeasured_points
    def period(self):
        """
        Calculates the dominant spatial period of the profile based on the Fourier transform.

        Returns
        -------
        period : float
        """
        x = np.arange(self.size) * self.step
        return get_period_fft_1d(x, self.data)

    def _sectioned_data(self, n_sections):
        """
        Centers the height data around the mean line and divides it into n approximately equally sized sections,
        corresponding to the sampling lengths defined by ISO 4287. The height values remain referenced to the mean
        line of the entire evaluation length.

        Parameters
        ----------
        n_sections : int
            Number of sections (sampling lengths) the profile is divided into.

        Returns
        -------
        list[ndarray]
        """
        if n_sections < 1:
            raise ValueError('The number of sections must be at least 1.')
        if n_sections > self.size:
            raise ValueError('The number of sections cannot exceed the number of datapoints.')
        return np.array_split(self.data - self.data.mean(), n_sections)

    # Height parameters ################################################################################################

    @cache
    @no_nonmeasured_points
    def Ra(self):
        """
        Calculates the arithmetic mean deviation Ra over the evaluation length.

        Returns
        -------
        Ra : float
        """
        return np.abs(self.data - self.data.mean()).sum() / self.data.size

    @cache
    @no_nonmeasured_points
    def Rq(self):
        """
        Calculates the root mean square deviation Rq over the evaluation length.

        Returns
        -------
        Rq : float
        """
        return np.sqrt(((self.data - self.data.mean()) ** 2).sum() / self.data.size)

    @cache
    @no_nonmeasured_points
    def Rp(self, n_sections=DEFAULT_N_SECTIONS):
        """
        Calculates the maximum peak height Rp. According to ISO 4287, Rp is defined on a sampling length and averaged
        over the number of sampling lengths contained in the evaluation length. The number of sampling lengths is
        specified by the n_sections argument and defaults to 5 according to ISO 4287/4288. Specify n_sections=1 to
        evaluate Rp on the entire evaluation length.

        Parameters
        ----------
        n_sections : int, default 5
            Number of sampling lengths the evaluation length is divided into.

        Returns
        -------
        Rp : float
        """
        return np.mean([section.max() for section in self._sectioned_data(n_sections)])

    @cache
    @no_nonmeasured_points
    def Rv(self, n_sections=DEFAULT_N_SECTIONS):
        """
        Calculates the maximum pit depth Rv. According to ISO 4287, Rv is defined on a sampling length and averaged
        over the number of sampling lengths contained in the evaluation length. The number of sampling lengths is
        specified by the n_sections argument and defaults to 5 according to ISO 4287/4288. Specify n_sections=1 to
        evaluate Rv on the entire evaluation length.

        Parameters
        ----------
        n_sections : int, default 5
            Number of sampling lengths the evaluation length is divided into.

        Returns
        -------
        Rv : float
        """
        return np.mean([np.abs(section.min()) for section in self._sectioned_data(n_sections)])

    @cache
    @no_nonmeasured_points
    def Rz(self, n_sections=DEFAULT_N_SECTIONS):
        """
        Calculates the maximum height Rz. According to ISO 4287, Rz is defined on a sampling length as the sum of the
        maximum peak height and the maximum pit depth and averaged over the number of sampling lengths contained in
        the evaluation length. The number of sampling lengths is specified by the n_sections argument and defaults to
        5 according to ISO 4287/4288. Specify n_sections=1 to evaluate Rz on the entire evaluation length, which is
        equivalent to the total height Rt.

        Parameters
        ----------
        n_sections : int, default 5
            Number of sampling lengths the evaluation length is divided into.

        Returns
        -------
        Rz : float
        """
        return np.mean([section.max() - section.min() for section in self._sectioned_data(n_sections)])

    @cache
    @no_nonmeasured_points
    def Rt(self):
        """
        Calculates the total height Rt as the sum of the maximum peak height and the maximum pit depth over the
        entire evaluation length.

        Returns
        -------
        Rt : float
        """
        return self.data.max() - self.data.min()

    @cache
    @no_nonmeasured_points
    def Rsk(self):
        """
        Calculates the skewness Rsk over the evaluation length. It is the quotient of the mean cube value of the
        ordinate values and the cube of Rq.

        Returns
        -------
        Rsk : float
        """
        return ((self.data - self.data.mean()) ** 3).sum() / self.data.size / self.Rq() ** 3

    @cache
    @no_nonmeasured_points
    def Rku(self):
        """
        Calculates the kurtosis Rku over the evaluation length. It is the quotient of the mean quartic value of the
        ordinate values and the fourth power of Rq.

        Returns
        -------
        Rku : float
        """
        return ((self.data - self.data.mean()) ** 4).sum() / self.data.size / self.Rq() ** 4

    # Hybrid parameters ################################################################################################

    @cache
    @no_nonmeasured_points
    def Rdq(self):
        """
        Calculates the root mean square slope Rdq over the evaluation length.

        Returns
        -------
        Rdq : float
        """
        return np.sqrt(np.mean((np.diff(self.data) / self.step) ** 2))

    # Functional parameters ############################################################################################

    def Rk(self):
        """
        Calculates the core roughness depth Rk according to ISO 13565-2 in µm.

        Returns
        -------
        Rk : float
        """
        return self.get_abbott_firestone_curve().k()

    def Rpk(self):
        """
        Calculates the reduced peak height Rpk according to ISO 13565-2 in µm.

        Returns
        -------
        Rpk : float
        """
        return self.get_abbott_firestone_curve().pk()

    def Rvk(self):
        """
        Calculates the reduced valley depth Rvk according to ISO 13565-2 in µm.

        Returns
        -------
        Rvk : float
        """
        return self.get_abbott_firestone_curve().vk()

    def Rmr1(self):
        """
        Calculates the material ratio Rmr1 according to ISO 13565-2 in %.

        Returns
        -------
        Rmr1 : float
        """
        return self.get_abbott_firestone_curve().mr1()

    def Rmr2(self):
        """
        Calculates the material ratio Rmr2 according to ISO 13565-2 in %.

        Returns
        -------
        Rmr2 : float
        """
        return self.get_abbott_firestone_curve().mr2()

    def Rmr(self, c):
        """
        Calculates the ratio of the material length at a specified height c (in µm) to the evaluation length.

        Parameters
        ----------
        c : float
            height in µm.

        Returns
        -------
        material ratio : float
        """
        return self.get_abbott_firestone_curve().mr(c)

    def Rmc(self, mr):
        """
        Calculates the height (c) in µm for a given material ratio (mr).

        Parameters
        ----------
        mr : float
            material ratio in %.

        Returns
        -------
        height : float
        """
        return self.get_abbott_firestone_curve().mc(mr)

    def Rxp(self, p=2.5, q=50):
        """
        Calculates the difference in height between the p and q material ratio.

        Parameters
        ----------
        p : float
            material ratio p in %.
        q : float
            material ratio q in %.

        Returns
        -------
        Height difference : float
        """
        return self.Rmc(p) - self.Rmc(q)

    # Plotting #########################################################################################################

    def plot_2d(self, ax=None):
        """
        Plots the profile.

        Parameters
        ----------
        ax : matplotlib axis, default None
            If specified, the plot will be drawn on the specified axis.

        Returns
        -------
        plt.Figure, plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=150, figsize=(10, 3))
        else:
            fig = ax.figure
        ax.set_xlim(0, self.length_um)
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('z [µm]')
        ax.plot(np.linspace(0, self.length_um, self.data.size), self.data, c='k', lw=1)
        return fig, ax

    def show(self):
        """
        Shows the plot of the profile.

        Returns
        -------
        None
        """
        self.plot_2d()
        plt.show()
