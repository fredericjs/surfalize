import numpy as np
import matplotlib.pyplot as plt

from .mathutils import argclosest, interp1d
from .cache import CachedInstance, cache

class AbbottFirestoneCurve(CachedInstance):
    """
    Represents the Abbott-Firestone curve of a Surface object and provides methods to calculate the functional
    roughness parameters derived from it.

    Parameters
    ----------
    surface : Surface
        Surface object from which to calcualte the Abbott-Firestone curve
    nbins : int, default 10000
        Number of bins for the material density histogram. Large numbers result in longer computation time but increased
        accuracy of results. The default value of 10000 represents a reasonable compromise.
    """
    # Width of the equivalence line in % as defined by ISO 25178-2
    EQUIVALENCE_LINE_WIDTH = 40

    def __init__(self, surface, nbins=10000):
        super().__init__()
        self._surface = surface
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
        hist, height = np.histogram(self._surface.data, bins=self._nbins)
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
        The resulting values and a linear interpolator for Smc and Smr are saved in instance attributes.

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
    def Sk(self):
        """
        Calculates Sk.

        Returns
        -------
        float
        """
        return self._yupper - self._ylower

    def Smr(self, c):
        """
        Calculates Smr(c).

        Parameters
        ----------
        c : float
            Material height.

        Returns
        -------
        float
        """
        return float(self._smr_fit(c))

    def Smc(self, mr):
        """
        Calculates Smc(mr).

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
    def Smr1(self):
        """
        Calculates Smr1.

        Returns
        -------
        float
        """
        return self.Smr(self._yupper)

    @cache
    def Smr2(self):
        """
        Calculates Smr2.

        Returns
        -------
        float
        """
        return self.Smr(self._ylower)

    @cache
    def Spk(self):
        """
        Calculates Spk.

        Returns
        -------
        float
        """
        # For now we are using the closest value in the array to ylower
        # This way, we are losing or gaining a bit of area. In the future we might use some
        # additional interpolation. For now this is sufficient.

        # Area enclosed above yupper between y-axis (at x=0) and abbott-firestone curve
        idx = argclosest(self._yupper, self._height)
        A1 = np.abs(np.trapz(self._material_ratio[:idx], x=self._height[:idx]))
        Spk = 2 * A1 / self.Smr1()
        return Spk

    @cache
    def Svk(self):
        """
        Calculates Svk.

        Returns
        -------
        float
        """
        # Area enclosed below ylower between y-axis (at x=100) and abbott-firestone curve
        idx = argclosest(self._ylower, self._height)
        A2 = np.abs(np.trapz(100 - self._material_ratio[idx:], x=self._height[idx:]))
        Svk = 2 * A2 / (100 - self.Smr2())
        return Svk

    @cache
    def Vmp(self, p=10):
        idx = argclosest(self.Smc(p), self._height)
        return np.abs(np.trapz(self._material_ratio[:idx], x=self._height[:idx]) / 100)

    @cache
    def Vmc(self, p=10, q=80):
        idx = argclosest(self.Smc(q), self._height)
        return np.abs(np.trapz(self._material_ratio[:idx], x=self._height[:idx])) / 100 - self.Vmp(p)

    @cache
    def Vvv(self, q=80):
        idx = argclosest(self.Smc(80), self._height)
        return np.abs(np.trapz(100 - self._material_ratio[idx:], x=self._height[idx:])) / 100

    @cache
    def Vvc(self, p=10, q=80):
        idx = argclosest(self.Smc(10), self._height)
        return np.abs(np.trapz(100 - self._material_ratio[idx:], x=self._height[idx:])) / 100 - self.Vvv(q)

    def plot(self, nbars=20, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        dist_bars, bins_bars = np.histogram(self._surface.data, bins=nbars)
        dist_bars = np.flip(dist_bars)
        bins_bars = np.flip(bins_bars)

        height, material_ratio = self._get_material_ratio_curve()

        ax.set_xlabel('Material distribution (%)')
        ax.set_ylabel('z (µm)')
        ax2 = ax.twiny()
        ax2.set_xlabel('Material ratio (%)')
        ax.set_box_aspect(1)
        ax2.set_xlim(0, 100)
        ax.set_ylim(self._surface.data.min(), self._surface.data.max())

        ax.barh(bins_bars[:-1] + np.diff(bins_bars) / 2, dist_bars / dist_bars.cumsum().max() * 100,
                height=(self._surface.data.max() - self._surface.data.min()) / nbars, edgecolor='k', color='lightblue')
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
        ax.add_patch(plt.Polygon([[0, self._yupper], [0, self._yupper + self.Spk()], [self.Smr1(), self._yupper]],
                                 fc='orange', ec='k'))
        ax.add_patch(plt.Polygon([[100, self._ylower], [100, self._ylower - self.Svk()], [self.Smr2(), self._ylower]],
                                 fc='orange', ec='k'))
        ax.plot(self._material_ratio, self._height, c='r')
        ax.axhline(self._ylower, c='k', lw=1)
        ax.axhline(self._yupper, c='k', lw=1)
        ax.axhline(self._ylower - self.Svk(), c='k', lw=1)
        ax.axhline(self._yupper + self.Spk(), c='k', lw=1)
        ax.plot([self.Smr1(), self.Smr1()], [0, self._yupper], c='k', lw=1)
        ax.plot([self.Smr2(), self.Smr2()], [0, self._ylower], c='k', lw=1)

        ax.set_xlabel('Material ratio (%)')
        ax.set_ylabel('Height (µm)')

        return fig, ax