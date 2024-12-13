from collections.abc import Sequence

import numpy as np


def approximately_equal(a, b, epsilon=1e-6):
    """
    Check if two floating point values are approximately equal.

    Parameters
    ----------
    a : float
        First value.
    b : float
        Second value.
    epsilon : float
        Maximum tolerated difference between the floating point values.
        Defaults to 1e-6.

    Returns
    -------
    bool
    """
    if abs(a - b) < epsilon:
        return True
    return False

def is_list_like(obj):
    """
    Determines whether an object is list-like.

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
        True if object is list-like, False if is is not.
    """
    return isinstance(obj, (Sequence, np.ndarray)) and not isinstance(obj, (str, bytes))

def register_returnlabels(labels):
    """
    Decorator that registers return labels to be used by surfalize.Batch when evaluating methods with multiple return
    values.

    Parameters
    ----------
    labels : list[str]
        List of labels with the same length as the number of return values of the method to be decorated.

    Returns
    -------
    wrapped_method
    """
    def wrapper(function):
        function.return_labels = labels
        return function
    return wrapper
