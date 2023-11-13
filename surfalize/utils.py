import numpy as np

def interp1d(xdata, ydata, assume_sorted=False):
    """
    Creates a function that linearly interpolates the given x- and y-data at any value of x.

    Mimics scipy.interpolate.interp1d, since the scipy version is no longer supported and might
    be removed in future versions. Contrary to the scipy implementation, assume_sorted is False by
    default, since a default True value would lead to more errors. If the xdata is sorted in
    ascending order, then assume_sorted can be set to True to gain a performance increase.

    Parameters
    ----------
    xdata: array_like
        array of x-values
    ydata: array_like
        array of y-values
    assume_sorted: bool, default False
        If True, they xdata array must be supplied with ascendingly ordered values and sorting is skipped.
        If False, the array xdata will be sorted in ascending order and the array ydata will be sorted accordingly.

    Returns
    -------
    function:
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
    x: float
        value to which closest array value index should be computed
    xdata: array_like
        array of x-values

    Returns
    -------
    index:
        Index of the value in xdata that is closest to x.
    """
    return np.argmin(np.abs(xdata - x))