import numpy as np

def is_list_like(obj):
    """
    Determines whether an object is list-like. For now, lists, tuples and numpy arrays are considered list-like.

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
        True if object is list-like, False if is is not.
    """
    if isinstance(obj, (list, tuple, np.ndarray)):
        return True
    return False

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
