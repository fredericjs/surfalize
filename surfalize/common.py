import functools
import logging
logger = logging.getLogger(__name__)
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def get_period_fft_1d(xdata, ydata):
    """
    Estimates the dominant period from a 1d periodic profile with uniformly spaced points.

    Parameters
    ----------
    xdata: list-like
        array of uniformly spaced xdata.
    ydata: list-like
        array of ydata with the same size as xdata.

    Returns
    -------
    period: float
    """
    fft = np.abs(np.fft.fft(ydata))
    freq = np.fft.fftfreq(len(data), d=xdata[1] - xdata[0])
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
    x: float | array-like
    amplitude: float
        amplitude.
    period: float
        period.
    x0: float
        offset in x.
    y0: float
        offset in y.

    Returns
    -------
    y: float | array-lile
    """
    return a * np.sin((x - x0) / p * 2 * np.pi) + y0

class Sinusoid:
    """
    Generic sinusoid representation.

    Parameters
    ----------
    amplitude: float
        amplitude.
    period: float
        period.
    x0: float
        offset in x.
    y0: float
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
        xdata: list-like
            array of x-data
        ydata: list-like
            array of y-data
        p0: list-like[float, float, float, float] | None, defaults to None
            Optional initial guess for the parameters a, p, x0, y0.
        infer_p0: bool, defaults to False
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
        popt, _ = curve_fit(_sinusoid, xdata, ydata, p0=p0)
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
        x: float | ndarray
            x-value at which to evaluate the sinusoid.
        Returns
        -------
        y: float
        """
        return _sinusoid(x, self.amplitude, self.period, self.x0, self.y0)

    def first_extremum(self):
        """
        Compute the position of the first extremum (peak or valley) for x >= 0.

        Returns
        -------
        xfe: float
        """
        return self.x0 % (self.period / 4) + self.period / 4

def register_returnlabels(labels):
    """
    Decorator that registers return labels to be used by surfalize.Batch when evaluating methods with multiple return
    values.

    Parameters
    ----------
    labels: list[str]
        List of labels with the same length as the number of return values of the method to be decorated.

    Returns
    -------
    wrapped_method
    """
    def wrapper(function):
        function.return_labels = labels
        return function
    return wrapper

def cache(method):
    """
    Decorator that enables caching on class instance without creating memory leaks. This is accomplished by relying
    on a cache inside the instance. Classes that want to use this decorator must implement a dictionary called
    '_method_cache' or inherit from CachedInstance.
    This approach is necessary because functools.lru_cache keeps references to the class instance that prevent it from
    being garbage collected indefinitely, thus leaking a substantial amount of memory.
    See https://bugs.python.org/issue19859 for more details.

    This implementation will not work on methods with unhashable arguments.

    Parameters
    ----------
    method: method
        method to be decorated.

    Returns
    -------
    wrapped_method
    """
    @functools.wraps(method)
    def wrapped_method(self, *args, **kwargs):
        # This only work for hashable arguments
        key = (method.__name__, *args, *tuple(kwargs.items()))
        try:
            # Cache hit
            value = self._method_cache[key]
            #print(f'Cache hit on {method.__name__}')
            return value
        except KeyError:
            # Cache miss
            #print(f'Cache miss on {method.__name__}')
            value = method(self, *args, **kwargs)
            self._method_cache[key] = value
            return value

    return wrapped_method


class CachedInstance:
    """
    Mixin class that provides the basic facilities necessary for the cache decorator as well as a method to clear the
    cache.
    """
    def __init__(self):
        self._method_cache = dict()

    def clear_cache(self):
        """
        Clears the cache for the entire instance.

        Returns
        -------
        None
        """
        self._method_cache = dict()