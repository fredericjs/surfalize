import numpy as np
import matplotlib.pyplot as plt

from .cache import CachedInstance, cache
from .mathutils import argclosest


class FourierTransform(CachedInstance):
    """
    Represents the 2d Fourier transform of a Surface object and provides methods to calculate the spatial period,
    texture orientation and angular power spectrum derived from it. The transform and its derived quantities are
    cached on the instance, so the FFT is computed only once and shared between the period, orientation and angular
    power spectrum calculations.

    Parameters
    ----------
    surface : Surface
        Surface object on which to calculate the Fourier transform.
    """
    def __init__(self, surface):
        if surface.has_missing_points:
            raise ValueError("Missing points must be filled before "
                             "the Fourier transform can be instantiated.") from None
        super().__init__()
        self._surface = surface

    @cache
    def _get_peak_dx_dy(self):
        """
        Calculates the distance in x and y in spatial frequency length units between the two largest Fourier peaks.
        The zero peak is avoided by centering the data around the mean. This is used by the period and orientation
        calculation.

        Returns
        -------
        (dx, dy) : tuple[float, float]
            Distance between largest Fourier peaks in x (dx) and in y (dy)
        """
        surface = self._surface
        # Get rid of the zero peak in the DFT for data that features a substantial offset in the z-direction
        # by centering the values around the mean
        data = surface.data - surface.data.mean()
        fft = np.abs(np.fft.fftshift(np.fft.fft2(data)))
        N, M = surface.size
        # Calculate the frequency values for the x and y axes
        freq_x = np.fft.fftshift(np.fft.fftfreq(M, d=surface.width_um / M))  # Frequency values for the x-axis
        freq_y = np.fft.fftshift(np.fft.fftfreq(N, d=surface.height_um / N))  # Frequency values for the y-axis
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

    @cache
    def period(self):
        """
        Calculates the 1d spatial period based on the Fourier transform.

        Returns
        -------
        period : float
        """
        dx, dy = self._get_peak_dx_dy()
        return 2 / np.hypot(dx, dy)

    @cache
    def period_x_y(self):
        """
        Calculates the spatial period along the x and y axes based on the Fourier transform.

        Returns
        -------
        (periodx, periody) : tuple[float, float]
        """
        dx, dy = self._get_peak_dx_dy()
        periodx = np.inf if dx == 0 else np.abs(2 / dx)
        periody = np.inf if dy == 0 else np.abs(2 / dy)
        return periodx, periody

    @cache
    def orientation(self):
        """
        Computes the orientation angle of the dominant texture towards the vertical axis from the peaks of the Fourier
        transform. Note that the refined orientation estimate (which additionally samples spatial profiles) lives on
        the Surface class, since it is not a purely Fourier-based quantity.

        Returns
        -------
        angle : float
            Angle of the dominant texture to the vertical axis
        """
        dx, dy = self._get_peak_dx_dy()
        # Account for special cases
        if dx == np.inf or dx == 0:
            return 90
        if dy == np.inf or dy == 0:
            return 0
        return np.rad2deg(np.arctan(dy / dx))

    @cache
    def dominant_wavelength(self):
        """
        Calculates the dominant spatial wavelength (Ssw), the wavelength corresponding to the largest absolute value
        of the Fourier transform of the ordinate values. The mean is subtracted beforehand to avoid the zero peak.

        Returns
        -------
        wavelength : float
        """
        surface = self._surface
        data = surface.data - surface.data.mean()
        fft = np.abs(np.fft.fftshift(np.fft.fft2(data)))
        N, M = surface.size
        freq_x = np.fft.fftshift(np.fft.fftfreq(M, d=surface.width_um / M))
        freq_y = np.fft.fftshift(np.fft.fftfreq(N, d=surface.height_um / N))
        iy, ix = np.unravel_index(np.argmax(fft), fft.shape)
        radial_frequency = np.hypot(freq_x[ix], freq_y[iy])
        return np.inf if radial_frequency == 0 else 1 / radial_frequency

    @cache
    def _angular_spectrum(self, angle_step, squared):
        """
        Computes the angular spectrum by accumulating the (squared) amplitude of the 2d Fourier transform into angular
        bins. With squared=False this yields the angular amplitude density fAAD, with squared=True the angular power
        density fAPD as defined in ISO 25178-2 (3.2.15.2 and 3.2.15.3).

        Parameters
        ----------
        angle_step : float
            Angular resolution of the spectrum in degree.
        squared : bool
            If True, the squared amplitude (power) is accumulated, otherwise the amplitude.

        Returns
        -------
        (angles, spectrum) : tuple[ndarray, ndarray]
        """
        surface = self._surface
        N, M = surface.size.y, surface.size.x
        # The mean is subtracted to remove the zero-frequency (DC) component, which would otherwise dominate the
        # spectrum and be accumulated into the 0 degree bin for surfaces with a height offset.
        amplitude = np.abs(np.fft.fft2(surface.data - surface.data.mean()))
        weights = amplitude ** 2 if squared else amplitude
        # Physical spatial frequencies (1/length) so that the angle is correct even when the sampling spacing differs
        # between the x and y direction.
        freq_y = np.fft.fftfreq(N, d=surface.height_um / N)
        freq_x = np.fft.fftfreq(M, d=surface.width_um / M)
        freq_y, freq_x = np.meshgrid(freq_y, freq_x, indexing='ij')
        # The texture direction has no sign, so opposing frequencies are folded into the [0, 180) degree range.
        theta = np.mod(np.rad2deg(np.arctan2(freq_y, freq_x)), 180.0)
        angles = np.arange(0, 180, angle_step)
        bin_idx = np.minimum((theta / angle_step).astype(int), angles.size - 1)
        spectrum = np.bincount(bin_idx.ravel(), weights=weights.ravel(), minlength=angles.size)
        return angles, spectrum

    @cache
    def angular_power_spectrum(self, angle_step=1):
        """
        Computes the angular power spectrum (angular power density fAPD) by integrating the squared amplitude of the
        Fourier transform over angular bins.

        Parameters
        ----------
        angle_step : float
            Angular resolution of the power spectrum in degree.

        Returns
        -------
        (angles, spectrum) : tuple[ndarray, ndarray]
        """
        return self._angular_spectrum(angle_step, squared=True)

    def Std(self, angle_step=0.5):
        """
        Calculates the texture direction parameter Std, the angle at which the angular spectrum is the largest. It
        represents the lay of the surface texture.

        ISO 25178-2 (4.3.4) defines Std via the angular amplitude density fAAD. In practice the angular power density
        fAPD (squared amplitude) is used here, which matches the behaviour of commercial software (e.g. MountainsMap).
        The two only differ for surfaces with several competing lays of similar strength, where the texture direction
        is intrinsically ambiguous; for a single dominant lay the result is identical.

        Parameters
        ----------
        angle_step : float
            Angular resolution of the spectrum in degree. Defaults to 0.5

        Returns
        -------
        float
        """
        angles, spectrum = self._angular_spectrum(angle_step, squared=True)
        return angles[np.argmax(spectrum)]

    def plot(self, ax=None, log=True, hanning=False, subtract_mean=True, fxmax=None, fymax=None,
             cmap='inferno', adjust_colormap=True):
        """
        Plots the 2d Fourier transform of the surface. Optionally, a Hanning window can be applied to reduce spectral
        leakage effects that occur when analyzing a signal of finite sample length.

        Parameters
        ----------
        ax : matplotlib axis, default None
            If specified, the plot will be drawn on the specified axis.
        log : bool, Default True
            Shows the logarithm of the Fourier Transform to increase peak visibility.
        hanning : bool, Default False
            Applys a Hanning window to the data before the transform.
        subtract_mean : bool, Default False
            Subtracts the mean of the data before the transform to avoid the zero peak.
        fxmax : float, Default None
            Maximum frequency displayed in x. The plot will be cropped to -fxmax : fxmax.
        fymax : float, Default None
            Maximum frequency displayed in y. The plot will be cropped to -fymax : fymax.
        cmap : str, Default 'inferno'
            Matplotlib colormap with which to map the data.
        adjust_colormap : bool, Default True
            If True, the colormap starts at the mean and ends at 0.7 times the maximum of the data
            to increase peak visibility.

        Returns
        -------
        plt.Figure, plt.Axes
        """
        surface = self._surface
        if ax is None:
            fig, ax = plt.subplots(dpi=150)
        else:
            fig = ax.figure

        N, M = surface.size
        data = surface.data
        if subtract_mean:
            data = data - surface.data.mean()

        if hanning:
            hann_window_y = np.hanning(N)
            hann_window_x = np.hanning(M)
            hann_window_2d = np.outer(hann_window_y, hann_window_x)
            data = data * hann_window_2d

        fft = np.abs(np.fft.fftshift(np.fft.fft2(data)))

        # Calculate the frequency values for the x and y axes
        freq_x = np.fft.fftshift(np.fft.fftfreq(M, d=surface.width_um / M))  # Frequency values for the x-axis
        freq_y = np.fft.fftshift(np.fft.fftfreq(N, d=surface.height_um / N))  # Frequency values for the y-axis

        if log:
            # We add a small offset to avoid ln(0)
            fft = np.log10(fft + 1e-10)
        ixmin = 0
        ixmax = M - 1
        iymin = 0
        iymax = N - 1

        if fxmax is not None:
            ixmax = argclosest(fxmax, freq_x)
            ixmin = M - ixmax
            fft = fft[:, ixmin:ixmax + 1]

        if fymax is not None:
            iymax = argclosest(fymax, freq_y)
            iymin = N - iymax
            fft = fft[iymin:iymax + 1]

        vmin = None
        vmax = None
        if adjust_colormap:
            vmin = fft.mean()
            vmax = 0.7 * fft.max()

        ax.set_xlabel('Frequency [µm$^{-1}$]')
        ax.set_ylabel('Frequency [µm$^{-1}$]')
        extent = (freq_x[ixmin], freq_x[ixmax], freq_y[iymax], freq_y[iymin])

        ax.imshow(fft, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        return fig, ax

    def plot_angular_power_spectrum(self, ax=None, angle_step=1):
        """
        Plots the angular power spectrum on a polar axis.

        Returns
        -------
        plt.Figure, plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=150, subplot_kw={'projection': 'polar'})
        else:
            fig = ax.figure
            rows, cols, start, stop = ax.get_subplotspec().get_geometry()
            ax.remove()
            ax = fig.add_subplot(rows, cols, start + 1, projection='polar')

        angles, spectrum = self.angular_power_spectrum(angle_step=angle_step)

        ax.plot(np.deg2rad(angles), spectrum, clip_on=False)
        ax.set_theta_direction(1)
        ax.set_theta_zero_location('E')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_yticks([])

        return fig, ax
