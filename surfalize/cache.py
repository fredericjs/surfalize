import functools

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
    method : method
        method to be decorated.

    Returns
    -------
    wrapped_method
    """
    @functools.wraps(method)
    def wrapped_method(self, *args, **kwargs):
        # This only work for hashable arguments
        key = (method.__name__, str(args), str(kwargs.items()))
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

    def create_cache_entry(self, method, entry, args, kwargs):
        """
        Manually creates a cache entry for the specified method.

        Parameters
        ----------
        method : function pointer
            method to cache.
        entry : any
            return value that will be cached
        args
            arguments to the method call for which to create the cache entry
        kwargs
            keyword arguments to the method call for which to create the cache entry

        Returns
        -------
        None
        """
        key = (method.__name__, str(args), str(kwargs.items()))
        self._method_cache[key] = entry