import numpy as np
from matplotlib import pyplot as plt

from surfalize.basestudiable import BaseStudiable
from surfalize.cache import cache


class FourierTransform(BaseStudiable):

    @classmethod
    def from_surface(cls, surface, hanning=False, subtract_mean=False):
        data = surface.data
        if subtract_mean:
            data = data - data.mean()
        if hanning:
            hann_window_y = np.hanning(surface.size.y)
            hann_window_x = np.hanning(surface.size.x)
            hann_window_2d = np.outer(hann_window_y, hann_window_x)
            data = data * hann_window_2d
        data = np.fft.fftshift(np.fft.fft2(data))
        step_x = 1 / surface.width
        step_y = 1 / surface.height
        offset_x = - 1 / (2 * surface.step_x)
        offset_y = - 1 / (2 * surface.step_y)
        return cls(data, step_x, step_y, offset_x, offset_y)

    def inverse_transform(self):
        data = np.fft.ifft2(np.fft.ifftshift(self.data))
        data = np.real(data)
        step_x = 1 / self.step_x / self.size.x
        step_y = 1 / self.step_y / self.size.y
        return BaseStudiable(data, step_x, step_y)

    @cache
    def get_peaks_dx_dy(self):
        """
        Calculates the distance in x and y in spatial frequency length units. The zero peak is avoided by
        centering the data around the mean. This method is used by the period and orientation calculation.

        Returns
        -------
        (dx, dy) : tuple[float,float]
            Distance between largest Fourier peaks in x (dx) and in y (dy)
        """
        freq_x = np.fft.fftshift(np.fft.fftfreq(self.size.x, d=self.width / M))  # Frequency values for the x-axis
        freq_y = np.fft.fftshift(np.fft.fftfreq(N, d=self.height / N))  # Frequency values for the y-axis
        # Sort in descending order by computing sorting indices
        idx_y, idx_x = np.unravel_index(np.argsort(fft.flatten())[::-1], fft.shape)
        # Transform into spatial frequencies in length units
        # If this is not done, the computed angle will be wrong since the frequency per pixel
        # resolution is different in x and y due to the different sampling length!
        peaks_x = freq_x[idx_x]
        peaks_y = freq_y[idx_y]
        # Create peak tuples for ease of use
        peak0 = (peaks_x[0], peaks_y[0])
        peak1 = (peaks_x[1], peaks_y[1])
        # Peak1 should always be to the right of peak0
        if peak0[0] > peak1[0]:
            peak0, peak1 = peak1, peak0

        dx = peak1[0] - peak0[0]
        dy = peak0[1] - peak1[1]

        return dx, dy

    def plot(self, log=True, cmap='inferno'):

        # We add a small offset to avoid ln(0)
        data = np.abs(self.data)
        data = np.log10(data+ 1e-10) if log else data

        fig, ax = plt.subplots()
        ax.set_xlabel('Frequency [µm$^{-1}$]')
        ax.set_ylabel('Frequency [µm$^{-1}$]')
        extent = (self.offset_x, self.offset_x + self.width, self.offset_y, self.offset_y + self.height)
        ax.imshow(data, cmap=cmap, extent=extent, origin='lower')
        return ax

    def show(self):
        self.plot()
        plt.show()