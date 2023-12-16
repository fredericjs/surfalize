import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from .utils import argclosest
from .common import sinusoid

class Profile:

    def __init__(self, height_data, step, length_um):
        self._data = height_data
        self._step = step
        self._length_um = length_um

    def __repr__(self):
        return f'{self.__class__.__name__}({self._length_um:.2f} µm)'

    def _repr_png_(self):
        """
        Repr method for Jupyter notebooks. When Jupyter makes a call to repr, it checks first if a _repr_png_ is
        defined. If not, it falls back on __repr__.
        """
        self.show()

    def period(self):
        fft = np.abs(np.fft.fft(self._data))
        freq = np.fft.fftfreq(self._data.shape[0], d=self._step)
        peaks, properties = find_peaks(fft.flatten(), distance=10, prominence=10)
        # Find the prominence of the peaks
        prominences = properties['prominences']
        # Sort in descendin'g order by computing sorting indices
        sorted_indices = np.argsort(prominences)[::-1]
        # Sort peaks in descending order
        peaks_sorted = peaks[sorted_indices]
        # Rearrange prominences based on the sorting of peaks
        prominences_sorted = prominences[sorted_indices]
        period = 1 / np.abs(freq[peaks_sorted[0]])
        return period

    def Ra(self):
        return np.abs(self._data - self._data.mean()).sum() / self._data.size

    def Rq(self):
        return np.sqrt(((self._data - self._data.mean()) ** 2).sum() / self._data.size)

    def Rp(self):
        return (self._data - self._data.mean()).max()

    def Rv(self):
        return np.abs((self._data - self._data.mean()).min())

    def Rz(self):
        return self.Rp() + self.Rv()

    def Rsk(self):
        return ((self._data - self._data.mean()) ** 3).sum() / self._data.size / self.Rq() ** 3

    def Rku(self):
        return ((self._data - self._data.mean()) ** 4).sum() / self._data.size / self.Rq() ** 4

    def depth(self, sampling_width=0.2, plot=False, retstd=False):
        period_px = int(self.period() / self._step)
        nintervals = int(self._data.shape[0] / period_px)
        xp = np.arange(self._data.shape[0])
        # Define initial guess for fit parameters
        p0 = ((self._data.max() - self._data.min()) / 2, period_px, 0, self._data.mean())
        # Fit the data to the general sine function
        popt, pcov = curve_fit(sinusoid, xp, self._data, p0=p0)
        # Extract the refined period estimate from the sine function period
        period_sin = popt[1]
        # Extract the lateral shift of the sine fit
        x0 = popt[2]

        depths_line = np.zeros(nintervals * 2)

        if plot:
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.plot(xp, self._data, lw=1.5, c='k', alpha=0.7)
            ax.plot(xp, sinusoid(xp, *popt), c='orange', ls='--')
            ax.set_xlim(xp.min(), xp.max())

        # Loop over each interval
        for i in range(nintervals * 2):
            idx = (0.25 + 0.5 * i) * period_sin + x0

            idx_min = int(idx) - int(period_sin * sampling_width / 2)
            idx_max = int(idx) + int(period_sin * sampling_width / 2)
            if idx_min < 0 or idx_max > self._data.shape[0] - 1:
                depths_line[i] = np.nan
                continue
            depth_mean = self._data[idx_min:idx_max + 1].mean()
            depth_median = np.median(self._data[idx_min:idx_max + 1])
            depths_line[i] = depth_median
            # For plotting
            if plot:
                rx = xp[idx_min:idx_max + 1].min()
                ry = self._data[idx_min:idx_max + 1].min()
                rw = xp[idx_max] - xp[idx_min + 1]
                rh = np.abs(self._data[idx_min:idx_max + 1].min() - self._data[idx_min:idx_max + 1].max())
                rect = plt.Rectangle((rx, ry), rw, rh, facecolor='tab:orange')
                ax.plot([rx, rx + rw], [depth_mean, depth_mean], c='r')
                ax.plot([rx, rx + rw], [depth_median, depth_median], c='g')
                ax.add_patch(rect)

                # Subtract peaks and valleys from eachother by slicing with 2 step
        depths = np.abs(depths_line[0::2] - depths_line[1::2])

        if retstd:
            return np.nanmean(depths), np.nanstd(depths)
        return np.nanmean(depths)

    def show(self):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_xlim(0, self._length_um)
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('z [µm]')
        ax.plot(np.linspace(0, self._length_um, self._data.size), self._data, c='k', lw=1)
        plt.show()