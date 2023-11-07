import cython
import numpy as np
from libc.math cimport sqrt

cpdef double triangle_area(double x0, double y0, double z0, double x1, double y1, double z1):
    """
    Calculates the area of a 3d triangle spanned by the points A, B, C based on the two spanning vectors
    AB and AC, where AB = (x0, y0, z0) and AC = (x1, y1, z1). The area is calculated by computing the 
    magnitude of the normal vector obtained from the cross product of AB and AC.
    """
    return 0.5 * sqrt((y0*z1 - z0*y1)**2 + (z0*x1 - x0*z1)**2 + (x0*y1 - y0*x1)**2)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _surface_area_cy(double[:,:] p, double dx, double dy):
    """
    Calculates the surface area of a 3d pointcloud with regular grid spacing in x and y. In each set of four
    neighboring points, a center point is computed with the average height zm of the four corners. Then, four
    triangles are spanned between the four corner points and the center point. This function assumes that the
    height data is extended on all four sides by pre- and appending both a copy of the first and last column
    as well as row. This is done to deal with the border vertices according to the strategy proposed by
    Gwyddion http://gwyddion.net/documentation/user-guide-en/statistical-analysis.html.
    """
    cdef float total_area = 0
    cdef int i, j
    cdef int imax = p.shape[0]-1
    cdef int jmax = p.shape[1]-1
    cdef double a1, a2, a3, a4
    cdef double zm
    for i in range(imax):
        for j in range(jmax):
            zm = (p[i,j] + p[i+1,j] + p[i,j+1] + p[i+1,j+1]) / 4 
            a1 = triangle_area(0, dy, p[i+1,j] - p[i,j], dx/2, dy/2, zm - p[i,j])         
            a2 = triangle_area(0, dx, p[i,j+1] - p[i,j], dx/2, dy/2, zm - p[i,j])         
            a3 = triangle_area(0, dy, p[i+1,j] - p[i+1,j+1], dx/2, dy/2, zm - p[i+1,j+1]) 
            a4 = triangle_area(0, dx, p[i,j+1] - p[i+1,j+1], dx/2, dy/2, zm - p[i+1,j+1]) 
                
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

def _extend_array(array):
    array_ex = np.zeros((array.shape[0]+2, array.shape[1]+2))
    array_ex[1:-1, 1:-1] = array
    array_ex[0] = array_ex[1]
    array_ex[-1] = array_ex[-2]
    array_ex[:,0] = array_ex[:,1]
    array_ex[:,-1] = array_ex[:,-2]
    return array_ex

def surface_area(array, dx, dy):
    extended_array = _extend_array(array)
    return _surface_area_cy(extended_array, dx, dy)