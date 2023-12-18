import cython
import numpy as np
from libc.math cimport sqrt

cdef float triangle_area_f(float x0, float y0, float z0, float x1, float y1, float z1):
    """
    Calculates the area of a 3d triangle spanned by the points A, B, C based on the two spanning vectors
    AB and AC, where AB = (x0, y0, z0) and AC = (x1, y1, z1). The area is calculated by computing the
    magnitude of the normal vector obtained from the cross product of AB and AC.
    """
    return 0.5 * sqrt((y0*z1 - z0*y1)**2 + (z0*x1 - x0*z1)**2 + (x0*y1 - y0*x1)**2)

cdef double triangle_area_d(double x0, double y0, double z0, double x1, double y1, double z1):
    """
    Calculates the area of a 3d triangle spanned by the points A, B, C based on the two spanning vectors
    AB and AC, where AB = (x0, y0, z0) and AC = (x1, y1, z1). The area is calculated by computing the
    magnitude of the normal vector obtained from the cross product of AB and AC.
    """
    return 0.5 * sqrt((y0*z1 - z0*y1)**2 + (z0*x1 - x0*z1)**2 + (x0*y1 - y0*x1)**2)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _surface_area_gwyddion_cy_d(double[:,:] p, double dx, double dy):
    """
    Calculates the surface area of a 3d pointcloud with regular grid spacing in x and y. In each set of four
    neighboring points, a center point is computed with the average height zm of the four corners. Then, four
    triangles are spanned between the four corner points and the center point. This function assumes that the
    height data is extended on all four sides by pre- and appending both a copy of the first and last column
    as well as row. This is done to deal with the border vertices according to the strategy proposed by
    Gwyddion http://gwyddion.net/documentation/user-guide-en/statistical-analysis.html.
    """
    cdef double total_area = 0
    cdef int i, j
    cdef Py_ssize_t imax = p.shape[0]-1
    cdef Py_ssize_t jmax = p.shape[1]-1
    cdef double a1, a2, a3, a4
    cdef double zm
    for i in range(imax):
        for j in range(jmax):
            zm = (p[i,j] + p[i+1,j] + p[i,j+1] + p[i+1,j+1]) / 4
            a1 = triangle_area_d(0, dy, p[i+1,j] - p[i,j], dx/2, dy/2, zm - p[i,j])
            a2 = triangle_area_d(0, dx, p[i,j+1] - p[i,j], dx/2, dy/2, zm - p[i,j])
            a3 = triangle_area_d(0, dy, p[i+1,j] - p[i+1,j+1], dx/2, dy/2, zm - p[i+1,j+1])
            a4 = triangle_area_d(0, dx, p[i,j+1] - p[i+1,j+1], dx/2, dy/2, zm - p[i+1,j+1])

            if i == 0:
                a2 = 0
                a1 /= 2
                a4 /= 2
            elif i == imax:
                a3 = 0
                a1 /= 2
                a4 /= 2
            if j == 0:
                a1 = 0
                a2 /= 2
                a3 /= 2
            elif j == jmax:
                a4 = 0
                a2 /= 2
                a3 /= 2

            total_area += a1 + a2 + a3 + a4

    return total_area

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _surface_area_iso_cy_d(double[:,:] p, double dx, double dy):
    """
    Calculates the surface area of a 3d pointcloud with regular grid spacing in x and y. In each set of four
    neighboring points ABCD, two triangles ABC and ADC are formed. The total surface area is computed as
    the sum off all triangles on the surface. This method is detailed in "Stout, K.J. et al. The development
    of methods for the characterisation of roughness in three dimensions. European Report EUR 15178 EN,
    ISBN 0704413132" and referenced in ISO 25178-2. It is the method used by MountainsMap and its derivatives.
    """
    cdef double total_area = 0
    cdef int i, j
    cdef Py_ssize_t imax = p.shape[0] - 1
    cdef Py_ssize_t jmax = p.shape[1] - 1
    cdef double a1, a2
    for i in range(imax):
        for j in range(jmax):
            a1 = triangle_area_d(0, dy, p[i+1,j] - p[i,j], dx, 0, p[i,j+1] - p[i,j])
            a2 = triangle_area_d(dx, 0, p[i+1,j] - p[i+1,j+1], 0, dy, p[i,j+1] - p[i+1,j+1])
            total_area += a1 + a2
    return total_area

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _surface_area_iso_cy_f(float[:,:] p, double dx, double dy):
    """
    Calculates the surface area of a 3d pointcloud with regular grid spacing in x and y. In each set of four
    neighboring points ABCD, two triangles ABC and ADC are formed. The total surface area is computed as
    the sum off all triangles on the surface. This method is detailed in "Stout, K.J. et al. The development
    of methods for the characterisation of roughness in three dimensions. European Report EUR 15178 EN,
    ISBN 0704413132" and referenced in ISO 25178-2. It is the method used by MountainsMap and its derivatives.
    """
    cdef double total_area = 0
    cdef int i, j
    cdef Py_ssize_t imax = p.shape[0] - 1
    cdef Py_ssize_t jmax = p.shape[1] - 1
    cdef double a1, a2
    for i in range(imax):
        for j in range(jmax):
            a1 = triangle_area_f(0, dy, p[i+1,j] - p[i,j], dx, 0, p[i,j+1] - p[i,j])
            a2 = triangle_area_f(dx, 0, p[i+1,j] - p[i+1,j+1], 0, dy, p[i,j+1] - p[i+1,j+1])
            total_area += a1 + a2
    return total_area


def _extend_array(array):
    """
    Extends the array of shape (N, M) in all directions by a copy of the boundary rows and columns, returning an
    array of shape (N+2, M+2).
    """
    array_ex = np.zeros((array.shape[0]+2, array.shape[1]+2))
    array_ex[1:-1, 1:-1] = array
    array_ex[0] = array_ex[1]
    array_ex[-1] = array_ex[-2]
    array_ex[:,0] = array_ex[:,1]
    array_ex[:,-1] = array_ex[:,-2]
    return array_ex

def surface_area(array, dx, dy, method='iso'):
    """
    Calculates the surface area of an array. The method parameter can be either 'iso' or 'gwyddion'. The default method
    is the 'iso' method proposed by ISO 25178 and used by MountainsMap, whereby two triangles are spanned between four
    corner points. The 'gwyddion' method implements the approach used by the open-source software Gwyddion, whereby
    four triangles are spanned between four corner points and their calculated center point. The method is detailed here
    http://gwyddion.net/documentation/user-guide-en/statistical-analysis.html.

    Parameters
    ----------
    array: array-like
        2d-array of height data
    dx: float
        width per pixel in lateral direction
    dy: float
        height per pixel in vertical direction
    method: str, Default 'iso'
        The method by which to calculate the surface area.
    Returns
    -------
    area: float
    """
    if method == 'iso':
        if array.dtype == np.float32:
            return _surface_area_iso_cy_f(array, dx, dy)
        elif array.dtype == np.float64:
            return _surface_area_iso_cy_d(array, dx, dy)
    if method == 'gwyddion':
        if array.dtype == np.float64:
            extended_array = _extend_array(array)
            return _surface_area_gwyddion_cy_d(extended_array, dx, dy)
        else:
            raise NotImplementedError('This dtype is not currently implemented')