import functools
import logging
logger = logging.getLogger(__name__)
import numpy as np
from scipy.optimize import curve_fit

# Define general sinusoid function for fitting
sinusoid = lambda x, a, p, xo, yo: a*np.sin((x-xo)/p*2*np.pi) + yo

def fit_sinusoid(x, y, p0=None):
    """
    Fit a general sinusoid to x and y data using scipy.optimize.curve_fit. Optionally, an initial guess for the
    sinusoid parameters a, p, x0, y0 can be specified using the p0 keyword argument, where a is the amplitude, p is the
    period, x0 is the lateral and y0 the vertical offset.

    This function uses unbounded fitting, which can result in negative amplitude, because bounded fitting using
    scipy.optimize.curve_fit invokes a different algorithm which seems to perform worse on this specific fitting
    problem. Instead, a result with negative amplitude is converted into the positive amplitude equivalent by phase
    shifting.

    Parameters
    ----------
    x: list-like
        array of x-data
    y: list-like
        array of y-data
    p0: list-like[float, float, float, float] | None, defaults to None
        Optional initial guess for the parameters a, p, x0, y0.

    Returns
    -------
    parameters: tuple[float, float, float, float]
        Fitting parameters a, p, x0, y0
    """
    popt, pcov = curve_fit(sinusoid, x, y, p0=p0)
    a, p , x0, y0 = popt
    # We don't specifiy positive bounds for a, since the fit quality is worse for some reason if we do
    # Probably because scipy switches to a different algorithm internally when bounds are specified
    # Instead we just construct the equivalent sinusoid with positive a if a < 0
    if a < 0:
        a = np.abs(a)
        x0 = x0 - p/2
    return a, p, x0, y0


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