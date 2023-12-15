import numpy as np
from scipy.optimize import curve_fit

# Define general sinusoid function for fitting
sinusoid = lambda x, a, p, xo, yo: a*np.sin((x-xo)/p*2*np.pi) + yo

def fit_sinusoid(x, y, p0=None):
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
    def wrapper(function):
        function.return_labels = labels
        return function
    return wrapper

