import numpy as np
import matplotlib.pyplot as plt

from .mathutils import argclosest, interp1d, trapezoid
from .cache import CachedInstance, cache

class AbbottFirestoneCurve(CachedInstance):
    """
    Represents the Abbott-Firestone curve of a Surface or Profile object and provides methods to calculate the
    functional roughness parameters derived from it. The parameter methods are named agnostically with respect to
    the dimensionality of the underlying data, since the calculation is identical for profiles and surfaces. For
    instance, the method 'k' computes the core height, which corresponds to Sk for surfaces and Rk for profiles.

    Parameters
    ----------
    obj : Surface | Profile
        Surface or Profile object from which to calculate the Abbott-Firestone curve.
    nbins : int, default 10000
        Number of bins for the material density histogram. Large numbers result in longer computation time but increased
        accuracy of results. The default value of 10000 represents a reasonable compromise.
    """
    # Width of the equivalence line in % as defined by ISO 25178-2 and ISO 13565-2
    EQUIVALENCE_LINE_WIDTH = 40

    def __init__(self, obj, nbins=10000):
        if obj.has_missing_points:
            raise ValueError("Missing points must be filled before the "
                             "Abbott-Firestone curve can be instantiated.") from None
        super().__init__()
        self._obj = obj
        self._nbins = nbins
        self._calculate_curve()

    @cache
    def _get_material_ratio_curve(self):
        """
        Computes the height bins and cumulated material ratio.

        Returns
        -------
        height, material_ratio : tuple[ndarray[float], ndarray[float]]
        """
        hist, height = np.histogram(self._obj.data, bins=self._nbins)
        hist = hist[::-1]  # sort descending
        height = height[::-1]  # sort descending
        material_ratio = np.append(1, np.cumsum(hist))  # prepend 1 for first bin edge after cumsum
        material_ratio = material_ratio / material_ratio.max() * 100
        return height, material_ratio

    # This is a bit hacky right now with the modified state. Maybe clean that up in the future
    def _calculate_curve(self):
        """
        Performs the calculations necessary for evaluation of the functional parameters. First, the function finds
        the 40% equivalence line and computes its slope as well as intercept with 0% and 100% material ratio.
        The resulting values and a linear interpolator for mc and mr are saved in instance attributes.

        Returns
        -------
        None
        """
        parameters = dict()
        # Using the potentially cached values here
        height, material_ratio = self._get_material_ratio_curve()
        slope_min = None
        istart = 0
        istart_final = 0

        # Interpolation function for bin_centers(cumsum)
        self._smc_fit = interp1d(material_ratio, height)

        while True:
            # The width in material distribution % is 40, so we have to interpolate to find the index
            # where the distance to the starting value is 40
            if material_ratio[istart] > 100 - self.EQUIVALENCE_LINE_WIDTH:
                break
            # Here we interpolate to get exactly 40% width. The remaining inaccuracy comes from the
            # start index resoltion.
            slope = (self._smc_fit(material_ratio[istart] + self.EQUIVALENCE_LINE_WIDTH) - height[
                istart]) / self.EQUIVALENCE_LINE_WIDTH

            # Since slope is always negative, we check that the value is greater if we want
            # minimal gradient. If we find other instances with same slope, we take the first
            # occurence according to ISO 13565-2
            if slope_min is None:
                slope_min = slope
            elif slope > slope_min:
                slope_min = slope
                # Start index of the 40% width equivalence line
                istart_final = istart
            istart += 1

        self._slope = slope_min

        # Intercept of the equivalence line
        self._intercept = height[istart_final] - slope_min * material_ratio[istart_final]

        # Intercept of the equivalence line at 0% ratio
        self._yupper = self._intercept
        # Intercept of the equivalence line at 100% ratio
        self._ylower = slope_min * 100 + self._intercept

        self._smr_fit = interp1d(height, material_ratio)
        self._height = height
        self._material_ratio = material_ratio

    @cache
    def k(self):
        """
        Calculates the core height (Sk for surfaces, Rk for profiles).

        Returns
        -------
        float
        """
        return self._yupper - self._ylower

    def mr(self, c):
        """
        Calculates the material ratio at the height c (Smr(c) for surfaces, Rmr(c) for profiles).

        Parameters
        ----------
        c : float
            Material height.

        Returns
        -------
        float
        """
        return float(self._smr_fit(c))

    def mc(self, mr):
        """
        Calculates the height at the material ratio mr (Smc(mr) for surfaces, Rmc(mr) for profiles).

        Parameters
        ----------
        mr : float
            Material ratio.

        Returns
        -------
        float
        """
        return float(self._smc_fit(mr))

    @cache
    def mr1(self):
        """
        Calculates the material ratio that separates the peaks from the core (Smr1 for surfaces, Rmr1 for profiles).

        Returns
        -------
        float
        """
        return self.mr(self._yupper)

    @cache
    def mr2(self):
        """
        Calculates the material ratio that separates the dales from the core (Smr2 for surfaces, Rmr2 for profiles).

        Returns
        -------
        float
        """
        return self.mr(self._ylower)

    @cache
    def pk(self):
        """
        Calculates the reduced peak height (Spk for surfaces, Rpk for profiles).

        Returns
        -------
        float
        """
        # For now we are using the closest value in the array to ylower
        # This way, we are losing or gaining a bit of area. In the future we might use some
        # additional interpolation. For now this is sufficient.

        # Area enclosed above yupper between y-axis (at x=0) and abbott-firestone curve
        idx = argclosest(self._yupper, self._height)
        A1 = np.abs(trapezoid(self._material_ratio[:idx], x=self._height[:idx]))
        pk = 2 * A1 / self.mr1()
        return pk

    @cache
    def vk(self):
        """
        Calculates the reduced dale height (Svk for surfaces, Rvk for profiles).

        Returns
        -------
        float
        """
        # Area enclosed below ylower between y-axis (at x=100) and abbott-firestone curve
        idx = argclosest(self._ylower, self._height)
        A2 = np.abs(trapezoid(100 - self._material_ratio[idx:], x=self._height[idx:]))
        vk = 2 * A2 / (100 - self.mr2())
        return vk

    @cache
    def vmp(self, p=10):
        """
        Calculates the peak material volume at material ratio p (Vmp).

        Returns
        -------
        float
        """
        idx = argclosest(self.mc(p), self._height)
        return np.abs(trapezoid(self._material_ratio[:idx], x=self._height[:idx]) / 100)

    @cache
    def vmc(self, p=10, q=80):
        """
        Calculates the difference in material volume between material ratios p and q (Vmc).

        Returns
        -------
        float
        """
        idx = argclosest(self.mc(q), self._height)
        return np.abs(trapezoid(self._material_ratio[:idx], x=self._height[:idx])) / 100 - self.vmp(p)

    @cache
    def vvv(self, q=80):
        """
        Calculates the dale void volume at material ratio q (Vvv).

        Returns
        -------
        float
        """
        idx = argclosest(self.mc(q), self._height)
        return np.abs(trapezoid(100 - self._material_ratio[idx:], x=self._height[idx:])) / 100

    @cache
    def vvc(self, p=10, q=80):
        """
        Calculates the difference in void volume between material ratios p and q (Vvc).

        Returns
        -------
        float
        """
        idx = argclosest(self.mc(p), self._height)
        return np.abs(trapezoid(100 - self._material_ratio[idx:], x=self._height[idx:])) / 100 - self.vvv(q)

    def plot(self, nbars=20, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        dist_bars, bins_bars = np.histogram(self._obj.data, bins=nbars)
        dist_bars = np.flip(dist_bars)
        bins_bars = np.flip(bins_bars)

        height, material_ratio = self._get_material_ratio_curve()

        ax.set_xlabel('Material distribution (%)')
        ax.set_ylabel('z (µm)')
        ax2 = ax.twiny()
        ax2.set_xlabel('Material ratio (%)')
        ax.set_box_aspect(1)
        ax2.set_xlim(0, 100)
        ax.set_ylim(self._obj.data.min(), self._obj.data.max())

        ax.barh(bins_bars[:-1] + np.diff(bins_bars) / 2, dist_bars / dist_bars.cumsum().max() * 100,
                height=(self._obj.data.max() - self._obj.data.min()) / nbars, edgecolor='k', color='lightblue')
        ax2.plot(material_ratio, height, c='r', clip_on=True)

        return fig, (ax, ax2)

    def visual_parameter_study(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.set_box_aspect(1)
        ax.set_xlim(0, 100)
        ax.set_ylim(self._height.min(), self._height.max())
        x = np.linspace(0, 100, 10)
        ax.plot(x, self._slope * x + self._intercept, c='k')
        ax.add_patch(plt.Polygon([[0, self._yupper], [0, self._yupper + self.pk()], [self.mr1(), self._yupper]],
                                 fc='orange', ec='k'))
        ax.add_patch(plt.Polygon([[100, self._ylower], [100, self._ylower - self.vk()], [self.mr2(), self._ylower]],
                                 fc='orange', ec='k'))
        ax.plot(self._material_ratio, self._height, c='r')
        ax.axhline(self._ylower, c='k', lw=1)
        ax.axhline(self._yupper, c='k', lw=1)
        ax.axhline(self._ylower - self.vk(), c='k', lw=1)
        ax.axhline(self._yupper + self.pk(), c='k', lw=1)
        ax.plot([self.mr1(), self.mr1()], [0, self._yupper], c='k', lw=1)
        ax.plot([self.mr2(), self.mr2()], [0, self._ylower], c='k', lw=1)

        ax.set_xlabel('Material ratio (%)')
        ax.set_ylabel('Height (µm)')

        return fig, ax
