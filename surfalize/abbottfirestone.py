from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt

from .utils import argclosest, interp1d

class AbbottFirestoneCurve:
    # Width of the equivalence line in % as defined by ISO 25178-2
    EQUIVALENCE_LINE_WIDTH = 40

    def __init__(self, surface):
        self._surface = surface
        self._calculate_curve()

    @lru_cache
    def _get_material_ratio_curve(self, nbins=1000):
        dist, bins = np.histogram(self._surface.data, bins=nbins)
        bins = np.flip(bins)
        bin_centers = bins[:-1] + np.diff(bins) / 2
        cumsum = np.flip(np.cumsum(dist))
        cumsum = (1 - cumsum / cumsum.max()) * 100
        return nbins, bin_centers, cumsum

    # This is a bit hacky right now with the modified state. Maybe clean that up in the future
    def _calculate_curve(self):
        parameters = dict()
        # Using the potentially cached values here
        nbins, height, material_ratio = self._get_material_ratio_curve()
        # Step in the height array
        dc = np.abs(height[0] - height[1])
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
        self._dc = dc

    @lru_cache
    def Sk(self):
        return self._yupper - self._ylower

    def Smr(self, c):
        return float(self._smr_fit(c))

    def Smc(self, mr):
        return float(self._smc_fit(mr))

    @lru_cache
    def Smr1(self):
        return self.Smr(self._yupper)

    @lru_cache
    def Smr2(self):
        return self.Smr(self._ylower)

    @lru_cache
    def Spk(self):
        # For now we are using the closest value in the array to ylower
        # This way, we are losing or gaining a bit of area. In the future we might use some
        # additional interpolation. For now this is sufficient.

        # Area enclosed above yupper between y-axis (at x=0) and abbott-firestone curve
        idx = argclosest(self._yupper, self._height)
        A1 = np.abs(np.trapz(self._material_ratio[:idx], dx=self._dc))
        Spk = 2 * A1 / self.Smr1()
        return Spk

    @lru_cache
    def Svk(self):
        # Area enclosed below ylower between y-axis (at x=100) and abbott-firestone curve
        idx = argclosest(self._ylower, self._height)
        A2 = np.abs(np.trapz(100 - self._material_ratio[idx:], dx=self._dc))
        Svk = 2 * A2 / (100 - self.Smr2())
        return Svk

    @lru_cache
    def Vmp(self, p=10):
        idx = argclosest(self.Smc(p), self._height)
        return np.trapz(self._material_ratio[:idx], dx=self._dc) / 100

    @lru_cache
    def Vmc(self, p=10, q=80):
        idx = argclosest(self.Smc(q), self._height)
        return np.trapz(self._material_ratio[:idx], dx=self._dc) / 100 - self.Vmp(p)

    @lru_cache
    def Vvv(self, q=80):
        idx = argclosest(self.Smc(80), self._height)
        return np.abs(np.trapz(100 - self._material_ratio[idx:], dx=self._dc)) / 100

    @lru_cache
    def Vvc(self, p=10, q=80):
        idx = argclosest(self.Smc(10), self._height)
        return np.abs(np.trapz(100 - self._material_ratio[idx:], dx=self._dc)) / 100 - self.Vvv(q)

    def visual_parameter_study(self):
        fig, ax = plt.subplots()
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