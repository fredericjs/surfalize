from collections.abc import Sequence
import re
import numpy as np

def remove_parameter_from_docstring(parameter, docstring):
    """
    Removes the documentation of a parameter from a function's docstring.
    Returns the processed docstring.

    Parameters
    ----------
    parameter : str
        Name of the parameter to be removed.
    docstring : str
        Docstring from which the parameter should be removed.
    Returns
    -------
    str
    """
    pattern = parameter + r'.+\n((    |\t).+\n)+'
    return re.sub(pattern, '', docstring)

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