import cython
cimport numpy as np
from libc.math cimport fabs, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def height_parameters(np.ndarray[np.float64_t, ndim=2] data, double mean):
    cdef Py_ssize_t i, j, rows, cols, size
    rows = data.shape[0]
    cols = data.shape[1]
    size = rows * cols

    cdef double val = 0.0
    cdef double val2 = 0.0
    cdef double sum1 = 0.0
    cdef double sum2 = 0.0
    cdef double sum3 = 0.0
    cdef double sum4 = 0.0
    cdef double sv = data[0, 0] - mean
    cdef double sp = data[0, 0] - mean

    cdef double sa, sq, sz, ssk, sku

    with nogil:
        for i in range(rows):
            for j in range(cols):
                val = data[i, j] - mean
                val2 = val * val
                sum1 += fabs(val)
                sum2 += val2
                sum3 += val2 * val
                sum4 += val2 * val2
                sp = max(sp, val)
                sv = min(sv, val)

        sa = sum1 / size
        sq = sqrt(sum2 / size)
        sv = fabs(sv)
        sz = sp + sv
        sq2 = sq * sq
        ssk = sum3 / size / (sq2 * sq)
        sku = sum4 / size / (sq2 * sq2)
    return {'Sa': sa, 'Sq': sq, 'Sv': sv, 'Sp': sp, 'Sz': sz, 'Ssk': ssk, 'Sku': sku}