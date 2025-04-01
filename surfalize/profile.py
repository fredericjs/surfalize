import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class Profile:

    def __init__(self, height_data, step, length_um):
        self.data = height_data
        self.step = step
        self.length_um = length_um

    def __repr__(self):
        return f'{self.__class__.__name__}({self.length_um:.2f} µm)'

    def _repr_png_(self):
        """
        Repr method for Jupyter notebooks. When Jupyter makes a call to repr, it checks first if a _repr_png_ is
        defined. If not, it falls back on __repr__.
        """
        self.show()

    def period(self):
        fft = np.abs(np.fft.fft(self.data))
        freq = np.fft.fftfreq(self.data.shape[0], d=self.step)
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
        return np.abs(self.data - self.data.mean()).sum() / self.data.size

    def Rq(self):
        return np.sqrt(((self.data - self.data.mean()) ** 2).sum() / self.data.size)

    def Rp(self):
        return (self.data - self.data.mean()).max()

    def Rv(self):
        return np.abs((self.data - self.data.mean()).min())

    def Rz(self):
        return self.Rp() + self.Rv()

    def Rsk(self):
        return ((self.data - self.data.mean()) ** 3).sum() / self.data.size / self.Rq() ** 3

    def Rku(self):
        return ((self.data - self.data.mean()) ** 4).sum() / self.data.size / self.Rq() ** 4

    def plot_2d(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(dpi=150, figsize=(10, 3))
        else:
            fig = ax.figure
        ax.set_xlim(0, self.length_um)
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('z [µm]')
        ax.plot(np.linspace(0, self.length_um, self.data.size), self.data, c='k', lw=1)
        return fig, ax

    def show(self):
        self.plot_2d()
        plt.show()