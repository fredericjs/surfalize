import numpy as np


class Mask:
    """
    Represents a boolean mask associated with a `Surface` by composition. A masked point is excluded from analysis
    (leveling fits, height statistics, the Abbott-Firestone curve, ...) while its height value is preserved in the
    underlying data. This makes masking a reversible operation, in contrast to non-measured points which are stored as
    NaN and irreversibly discard the height value.

    The mask is lazily allocated: as long as no region is masked, no array is stored and the mask reports as empty. The
    convention follows :class:`numpy.ma.MaskedArray`, i.e. ``True`` marks a point that is masked (ignored).

    Two coordinate systems are supported:

    - The region methods (`add_rectangle`, `add_circle`, ...) operate in physical units (µm) by default, following the
      same convention as `Surface.crop`, where the y-axis is measured from the bottom of the surface.
    - Numpy-style indexing (``surface.mask[10:20, 5:15] = True``) operates in raw array indices (row, column) measured
      from the top-left, identical to indexing the underlying height array.

    Consistent with the rest of the surfalize API, the region methods return a `Surface` and do not modify the original
    by default (``inplace=False``), returning a copy with the updated mask instead. With ``inplace=True`` the surface is
    modified in place and returned, which allows chaining, e.g.
    ``surface.mask.add_rectangle((0, 10, 0, 10), inplace=True).mask.add_circle((50, 50), 5, inplace=True)``.

    The only exception is numpy-style item assignment (``surface.mask[2:5, 1:3] = True``), which is always in-place
    since Python item assignment cannot return a value.

    Parameters
    ----------
    surface : surfalize.Surface
        The surface this mask is associated with.
    array : ndarray[bool] | None, default None
        Optional initial boolean mask. Must match the shape of the surface height data.
    """

    def __init__(self, surface, array=None):
        self._surface = surface
        self._array = None
        if array is not None:
            self.set(array, inplace=True)

    def __repr__(self):
        if self.is_empty:
            return 'Mask(empty)'
        return f'Mask({int(self._array.sum())} of {self._array.size} px masked)'

    # Internal helpers #################################################################################################

    def _changed(self):
        """Invalidates the cache of the associated surface after the mask has been mutated."""
        self._surface.clear_cache()

    def _ensure(self):
        """Allocates the underlying boolean array on first write and returns it."""
        if self._array is None:
            self._array = np.zeros(self._surface.size, dtype=bool)
        return self._array

    def _target(self, inplace):
        """
        Returns the mask to operate on: this mask if inplace, otherwise the mask of a fresh copy of the surface. The
        associated surface is reachable through the returned mask's ``_surface`` attribute.
        """
        return self if inplace else self._surface.copy().mask

    # Query ############################################################################################################

    @property
    def is_empty(self):
        """Returns True if no point is masked."""
        return self._array is None or not self._array.any()

    def any(self):
        """Returns True if at least one point is masked."""
        return not self.is_empty

    def __bool__(self):
        return self.any()

    def to_array(self):
        """
        Returns a concrete boolean mask array, even if the mask is currently empty (in which case an all-False array
        of the surface shape is returned).

        Returns
        -------
        ndarray[bool]
        """
        if self._array is None:
            return np.zeros(self._surface.size, dtype=bool)
        return self._array

    # Direct assignment ################################################################################################

    def set(self, array, inplace=False):
        """
        Replaces the mask with the supplied boolean array.

        Parameters
        ----------
        array : ndarray[bool] | None
            Boolean array matching the surface shape, or None to clear the mask.
        inplace : bool, default False
            If False, return a copy of the surface with the updated mask. If True, modify the surface in place and
            return it.

        Returns
        -------
        surface : surfalize.Surface
        """
        m = self._target(inplace)
        if array is None:
            m._array = None
        else:
            array = np.asarray(array, dtype=bool)
            if array.shape != tuple(m._surface.size):
                raise ValueError(
                    f'Mask shape {array.shape} does not match surface shape {tuple(m._surface.size)}.'
                )
            m._array = array
        m._changed()
        return m._surface

    def __setitem__(self, key, value):
        self._ensure()[key] = value
        self._changed()

    def __getitem__(self, key):
        return self.to_array()[key]

    # Region construction ##############################################################################################

    def _rectangle_region(self, box, in_units):
        """Builds a boolean array that is True inside the specified rectangle."""
        ny, nx = self._surface.size
        if in_units:
            x0 = round(box[0] / self._surface.step_x)
            x1 = round(box[1] / self._surface.step_x)
            y1 = ny - round(box[2] / self._surface.step_y) - 1
            y0 = ny - round(box[3] / self._surface.step_y) - 1
        else:
            x0, x1, y0, y1 = box
        if x0 < 0 or y0 < 0 or x1 > nx - 1 or y1 > ny - 1:
            raise ValueError('Rectangle is out of bounds!')
        region = np.zeros((ny, nx), dtype=bool)
        region[y0:y1 + 1, x0:x1 + 1] = True
        return region

    def _circle_region(self, center, radius, in_units):
        """Builds a boolean array that is True inside the specified circle."""
        ny, nx = self._surface.size
        yy, xx = np.mgrid[0:ny, 0:nx]
        if in_units:
            x = xx * self._surface.step_x
            y = (ny - 1 - yy) * self._surface.step_y
        else:
            x, y = xx, yy
        cx, cy = center
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

    def add_rectangle(self, box, in_units=True, inplace=False):
        """
        Masks all points inside a rectangle.

        Parameters
        ----------
        box : tuple[float, float, float, float]
            Rectangle as a (x0, x1, y0, y1) tuple. If in_units is True, the values are in µm and the y-axis is measured
            from the bottom, matching the convention of `Surface.crop`. Otherwise the values are pixel indices.
        in_units : bool, default True
            If True, interpret box in physical units (µm). If False, interpret box in pixel indices.
        inplace : bool, default False
            If False, return a copy of the surface with the updated mask. If True, modify the surface in place and
            return it.

        Returns
        -------
        surface : surfalize.Surface
        """
        m = self._target(inplace)
        m._ensure()[m._rectangle_region(box, in_units)] = True
        m._changed()
        return m._surface

    def subtract_rectangle(self, box, in_units=True, inplace=False):
        """
        Unmasks all points inside a rectangle. See `add_rectangle` for the parameter description.

        Returns
        -------
        surface : surfalize.Surface
        """
        m = self._target(inplace)
        if m._array is not None:
            m._array[m._rectangle_region(box, in_units)] = False
            m._changed()
        return m._surface

    def add_circle(self, center, radius, in_units=True, inplace=False):
        """
        Masks all points inside a circle.

        Parameters
        ----------
        center : tuple[float, float]
            Center of the circle as a (x, y) tuple. If in_units is True, the values are in µm and y is measured from
            the bottom of the surface. Otherwise they are pixel indices (column, row).
        radius : float
            Radius of the circle in µm (in_units=True) or pixels (in_units=False).
        in_units : bool, default True
            If True, interpret center and radius in physical units (µm). If False, in pixels.
        inplace : bool, default False
            If False, return a copy of the surface with the updated mask. If True, modify the surface in place and
            return it.

        Returns
        -------
        surface : surfalize.Surface
        """
        m = self._target(inplace)
        m._ensure()[m._circle_region(center, radius, in_units)] = True
        m._changed()
        return m._surface

    def subtract_circle(self, center, radius, in_units=True, inplace=False):
        """
        Unmasks all points inside a circle. See `add_circle` for the parameter description.

        Returns
        -------
        surface : surfalize.Surface
        """
        m = self._target(inplace)
        if m._array is not None:
            m._array[m._circle_region(center, radius, in_units)] = False
            m._changed()
        return m._surface

    def threshold(self, below=None, above=None, inplace=False):
        """
        Masks all points whose height value falls below and/or above the specified thresholds.

        Parameters
        ----------
        below : float | None, default None
            If given, all points with a height value smaller than this value are masked.
        above : float | None, default None
            If given, all points with a height value larger than this value are masked.
        inplace : bool, default False
            If False, return a copy of the surface with the updated mask. If True, modify the surface in place and
            return it.

        Returns
        -------
        surface : surfalize.Surface
        """
        if below is None and above is None:
            raise ValueError('At least one of below or above must be specified.')
        m = self._target(inplace)
        data = m._surface.data
        region = np.zeros(m._surface.size, dtype=bool)
        if below is not None:
            region |= data < below
        if above is not None:
            region |= data > above
        m._ensure()[region] = True
        m._changed()
        return m._surface

    def invert(self, inplace=False):
        """
        Inverts the mask: masked points become unmasked and vice versa.

        Parameters
        ----------
        inplace : bool, default False
            If False, return a copy of the surface with the updated mask. If True, modify the surface in place and
            return it.

        Returns
        -------
        surface : surfalize.Surface
        """
        m = self._target(inplace)
        m._array = ~m.to_array()
        m._changed()
        return m._surface

    def clear(self, inplace=False):
        """
        Clears the mask so that no point is masked.

        Parameters
        ----------
        inplace : bool, default False
            If False, return a copy of the surface with the cleared mask. If True, modify the surface in place and
            return it.

        Returns
        -------
        surface : surfalize.Surface
        """
        m = self._target(inplace)
        m._array = None
        m._changed()
        return m._surface
