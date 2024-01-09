import cython

import numpy as np
cimport numpy as np

cpdef int argclosest(double[:] arr, double value):
    cdef int low, high, mid
    high = arr.size - 1
    low = 0
    
    while low < high:
        mid = (low + high) // 2
        if arr[mid] < value:
            low = mid + 1
        else:
            high = mid

    return low

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double trapz1d(np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] x):
    cdef double result = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t size = x.size
    
    for i in range(1, size):
        result += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Py_ssize_t binary_search_ascending(double[:] arr, double target):
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = arr.size - 1
    cdef Py_ssize_t result_index = -1
    cdef int mid

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            result_index = mid
            left = mid + 1
        else:
            right = mid - 1
    return result_index

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Py_ssize_t binary_search_descending(double[:] arr, double target):
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = arr.size - 1
    cdef Py_ssize_t result_index = -1
    cdef int mid

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] >= target:
            result_index = mid
            left = mid + 1
        else:
            right = mid - 1
    return result_index

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class LinearInterpolator:
    cdef double[:] xdata
    cdef double[:] ydata
    cdef double size

    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata
        self.size = xdata.size

    cpdef double interpolate(self, double x):
        cdef Py_ssize_t idx
        idx = self.get_index(x)
        cdef double x0 = self.xdata[idx]
        cdef double x1 = self.xdata[idx + 1]
        cdef double y0 = self.ydata[idx]
        cdef double y1 = self.ydata[idx + 1]

        return y0 + (y1 - y0) / (x1 - x0) * (x - x0)

    cpdef int get_index(self, double x):
        raise NotImplementedError

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DescendingLinearInterpolator(LinearInterpolator):

    cpdef int get_index(self, double x):
        return binary_search_descending(self.xdata, x)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class AscendingLinearInterpolator(LinearInterpolator):

    cpdef int get_index(self, double x):
        return binary_search_ascending(self.xdata, x)