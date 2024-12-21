import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.ndimage as ndimage
from .exceptions import FittingError

# Ensure compatibility with differnt numpy versions
if int(np.__version__.split('.')[0]) < 2:
    trapezoid = np.trapz
else:
    trapezoid = np.trapezoid

def interpolate_line_on_2d_array(array, start, end, order=3, num_points=100):
    """
    Interpolates a line between two points on a 2d array using spline interpolation.

    Parameters
    ----------
    array : 2d array-like
    start : tuple[int, int]
        Index of the start point.
    end : tuple[int, int]
        Index of the end point.
    order : int
        Order of spline interpolation. Defaults to 3.
    num_points : int
        Number of points of the interpolated line. Defaults to 100.

    Returns
    -------
    np.ndarray
    """
    coords = np.array([
        np.linspace(start[0], end[0], num_points),
        np.linspace(start[1], end[1], num_points)
    ])
    return ndimage.map_coordinates(array, coords, order=order)


def interp1d(xdata, ydata, assume_sorted=False):
    """
    Creates a function that linearly interpolates the given x- and y-data at any value of x.

    Mimics scipy.interpolate.interp1d, since the scipy version is no longer supported and might
    be removed in future versions. Contrary to the scipy implementation, assume_sorted is False by
    default, since a default True value would lead to more errors. If the xdata is sorted in
    ascending order, then assume_sorted can be set to True to gain a performance increase.

    Parameters
    ----------
    xdata : array_like
        array of x-values
    ydata : array_like
        array of y-values
    assume_sorted : bool, default False
        If True, they xdata array must be supplied with ascendingly ordered values and sorting is skipped.
        If False, the array xdata will be sorted in ascending order and the array ydata will be sorted accordingly.

    Returns
    -------
    function
        Linear interpolation function y(x).
    """
    if not assume_sorted:
        idx_sorted = np.argsort(xdata)
        xdata = xdata[idx_sorted]
        ydata = ydata[idx_sorted]

    @np.vectorize
    def y(x):
        return np.interp(x, xdata, ydata)

    return y


def argclosest(x, xdata):
    """
    Returns the index of the value in an array that is closest to the value x.

    Parameters
    ----------
    x : float
        value to which closest array value index should be computed
    xdata : array_like
        array of x-values

    Returns
    -------
    index : int
        Index of the value in xdata that is closest to x.
    """
    return np.argmin(np.abs(xdata - x))

def closest(x, data):
    """
    Returns the value in an array that is closest to the value x.

    Parameters
    ----------
    x : float
        value to which closest array value index should be computed
    xdata : array_like
        array of x-values

    Returns
    -------
    value
        Value in xdata that is closest to x.
    """
    return data.ravel()[argclosest(x, data.ravel())]


def argmax_all(arr):
    """
    Returns all indices where the array reaches its maximum value.

    Parameters
    ----------
    arr : array-like
        Input array

    Returns
    -------
    numpy.ndarray
        Array of indices where the maximum value occurs
    """
    max_val = np.max(arr)
    return np.where(arr == max_val)[0]


def argmin_all(arr):
    """
    Returns all indices where the array reaches its minimum value.

    Parameters
    ----------
    arr : array-like
        Input array

    Returns
    -------
    numpy.ndarray
        Array of indices where the minimum value occurs
    """
    min_val = np.min(arr)
    return np.where(arr == min_val)[0]

def get_period_fft_1d(xdata, ydata):
    """
    Estimates the dominant period from a 1d periodic profile with uniformly spaced points.

    Parameters
    ----------
    xdata : list-like
        array of uniformly spaced xdata.
    ydata : list-like
        array of ydata with the same size as xdata.

    Returns
    -------
    period : float
    """
    fft = np.abs(np.fft.fft(ydata))
    freq = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    peaks, properties = find_peaks(fft.flatten(), distance=10, prominence=10)
    # Find the prominence of the peaks
    prominences = properties['prominences']
    # Sort in descending order by computing sorting indices
    sorted_indices = np.argsort(prominences)[::-1]
    # Sort peaks in descending order
    peaks_sorted = peaks[sorted_indices]
    period = 1 / np.abs(freq[peaks_sorted[0]])
    return period

def _sinusoid(x, a, p, x0, y0):
    """
    Evaluates a sinusoid with the parameters a, p, x0, y0 at the position x.

    Parameters
    ----------
    x : float | array-like
    amplitude : float
        amplitude.
    period : float
        period.
    x0 : float
        offset in x.
    y0 : float
        offset in y.

    Returns
    -------
    y : float | array-like
    """
    return a * np.sin((x - x0) / p * 2 * np.pi) + y0


class Sinusoid:
    """
    Generic sinusoid representation.

    Parameters
    ----------
    amplitude : float
        amplitude.
    period : float
        period.
    x0 : float
        offset in x.
    y0 : float
        offset in y.
    """
    def __init__(self, amplitude, period, x0, y0):
        self.amplitude = amplitude
        self.period = period
        self.x0 = x0
        self.y0 = y0

    @classmethod
    def from_fit(cls, xdata, ydata, p0=None, infer_p0=False):
        """
        Fit a general sinusoid to x and y data using scipy.optimize.curve_fit. Optionally, an initial guess for the
        sinusoid parameters a, p, x0, y0 can be specified using the p0 keyword argument, where a is the amplitude, p is
        the period, x0 is the lateral and y0 the vertical offset.

        This function uses unbounded fitting, which can result in negative amplitude, because bounded fitting using
        scipy.optimize.curve_fit invokes a different algorithm which seems to perform worse on this specific fitting
        problem. Instead, a result with negative amplitude is converted into the positive amplitude equivalent by phase
        shifting.

        Parameters
        ----------
        xdata : list-like
            array of x-data
        ydata : list-like
            array of y-data
        p0 : list-like[float, float, float, float] | None, defaults to None
            Optional initial guess for the parameters a, p, x0, y0.
        infer_p0 : bool, defaults to False
            If True, automatically infers starting guesses of the parameters. Any values provided to the p0 keyword
            argument will be overwritten with the automatically inferred values.

        Returns
        -------
        Sinusoid
        """
        if infer_p0:
            # Initial guess for the parameters
            a = (np.max(ydata) - np.min(ydata)) / 2
            p = get_period_fft_1d(xdata, ydata)
            x0 = 0
            y0 = np.mean(ydata)
            p0 = (a, p, x0, y0)
        try:
            popt, _ = curve_fit(_sinusoid, xdata, ydata, p0=p0)
        except RuntimeError:
            raise FittingError('Sinusoid fitting was unsuccessful.') from None
        a, p, x0, y0 = popt
        # We don't specifiy positive bounds for a, since the fit quality is worse for some reason if we do
        # Probably because scipy switches to a different algorithm internally when bounds are specified
        # Instead we just construct the equivalent sinusoid with positive a if a < 0
        if a < 0:
            a = np.abs(a)
            x0 = x0 - p / 2
        return cls(a, p, x0, y0)

    def __call__(self, x):
        """
        Computes the value of a generic sinusoid at position x.

        Parameters
        ----------
        x : float | ndarray
            x-value at which to evaluate the sinusoid.

        Returns
        -------
        y : float
        """
        return _sinusoid(x, self.amplitude, self.period, self.x0, self.y0)

    def first_extremum(self):
        """
        Compute the position of the first extremum (peak or valley) for x >= 0.

        Returns
        -------
        xfe : float
        """
        # position of first extremum for x > x0
        x0e = self.x0 + self.period / 4
        # position of first extremum for x >= 0
        xfe = x0e - (x0e // (self.period / 2)) * self.period / 2
        return xfe

    def first_peak(self):
        """
        Compute the position of the first peak for x >= 0.

        Returns
        -------
        xfp : float
        """
        x0e = self.x0 + self.period / 4
        # position of first peak for x >= 0
        xfp = x0e - (x0e // (self.period)) * self.period
        return xfp