import cython
import numpy as np
from libc.math cimport sqrt

ctypedef fused sfloat:
    float
    double

cdef sfloat triangle_area(sfloat x0, sfloat y0, sfloat z0, sfloat x1, sfloat y1, sfloat z1):
    """
    Calculates the area of a 3d triangle spanned by the points A, B, C based on the two spanning vectors
    AB and AC, where AB = (x0, y0, z0) and AC = (x1, y1, z1). The area is calculated by computing the
    magnitude of the normal vector obtained from the cross product of AB and AC.
    """
    return 0.5 * sqrt((y0*z1 - z0*y1)**2 + (z0*x1 - x0*z1)**2 + (x0*y1 - y0*x1)**2)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def surface_area(sfloat[:,:] p, sfloat dx, sfloat dy):
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
            a1 = triangle_area(0, dy, p[i+1,j] - p[i,j], dx, 0, p[i,j+1] - p[i,j])
            a2 = triangle_area(dx, 0, p[i+1,j] - p[i+1,j+1], 0, dy, p[i,j+1] - p[i+1,j+1])
            total_area += a1 + a2
    return total_area