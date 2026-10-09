"""
Profile feature parameters according to ISO 21920-2:2021.

The profile feature parameters ``Rpd``, ``Rvd``, ``Rmpc``, ``Rmvc``, ``R5p``, ``R5v`` and ``R10z`` are computed by
`ProfileFeatureParameters`, which uses the ``featurecharacterization2d`` package.

https://github.com/mts-public/feature-characterization-for-profile-surface-texture
"""
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import featurecharacterization2d as fc2d

from .cache import CachedInstance, cache
from .feature import DEFAULT_PRUNING, _N_FIVE_POINT


class ProfileFeatureParameters(CachedInstance):
    """
    Computes the ISO 21920-2:2021 (default settings of ISO 21920-3:2021)
    profile feature parameters of a `Profile` by watershed segmentation and Wolf pruning.

    - ``Rpd`` / ``Rvd``: FC; P (V); Wolfprune 5 %; All; Count; Density, in 1/cm
    - ``Rmpc`` / ``Rmvc``: FC; P (V); Wolfprune 5 %; All; Curvature; Mean, in 1/µm
    - ``R5p`` / ``R5v``: FC; P (V); Wolfprune 5 %; Top (Bot) 5; PVh; Mean, in µm
    - ``R10z`` = ``R5p`` + ``R5v``, in µm

    Parameters
    ----------
    profile : Profile
        Profile object on which to compute the feature parameters. Must not contain non-measured.
    """

    def __init__(self, profile):
        if profile.has_missing_points:
            raise ValueError("Non-measured points must be filled before feature parameters can be computed.")

        super().__init__()
        self._profile = profile
        # Reference all heights to the mean line, so that R5p/R5v are measured from the mean line.
        self._data = np.ascontiguousarray(profile.data - profile.data.mean(), dtype=np.float64)
        # fc2d expects the step in mm, whereas Profile.step expects µm
        self._step_mm = profile.step / 1000
        # Rz as computed by fc2d, used as the
        # reference for the Wolf pruning threshold.
        # TODO: Replace by Profile.Rz once it follows ISO 21920-2.
        self._rz = float(fc2d.maximum_height(self._data, self._step_mm))

    @cache
    def _segment(self, invert, pruning):
        """
        Performs the watershed segmentation and Wolf pruning for either dales or hills.
        By default fc2d's segmentation excludes incomplete motifs at the profile border.

        Parameters
        ----------
        invert : bool
            If False, the dales (pits) are segmented. If True, the profile is inverted by fc2d
            first, so that the hills (peaks) are segmented.
        pruning : float
            Wolf pruning threshold as a percentage of Rz. Motifs whose local depth/height is smaller than this
            threshold are merged into a neighbouring motif across their lower peak (higher pit).

        Returns
        -------
        featurecharacterization2d.motif.Motif
            Container of the significant motifs, which may be empty.
        """
        threshold = pruning / 100 * self._rz
        # Feature type codes of fc2d: 'P' segments the peaks (hills), 'V' the pits (dales).
        feature_type = 'P' if invert else 'V'
        return fc2d.Watershed(self._data, self._step_mm, feature_type, 'Wolfprune', threshold).motifs()

    def _parameter(self, invert, pruning, significant, n_significant, attribute, statistic):
        """
        Evaluates a feature parameter in the feature characterization convention of ISO 21920-2 (feature type;
        pruning; significant features; attribute; statistic) on the significant motifs. Returns NaN if the profile has
        no significant motifs.
        """
        motifs = self._segment(invert=invert, pruning=pruning)
        # feature_parameter marks the non-significant motifs in place
        value = fc2d.feature_parameter(self._data, self._step_mm, deepcopy(motifs), significant, n_significant,
                                       attribute, statistic, np.nan)[0]
        return float(value)

    @cache
    def Rpd(self, pruning=DEFAULT_PRUNING):
        """
        Calculates Rpd in 1/cm.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Rz.

        Returns
        -------
        Rpd : float
        """
        return self._parameter(invert=True, pruning=pruning, significant='All', n_significant=1,
                               attribute='Count', statistic='Density')

    @cache
    def Rvd(self, pruning=DEFAULT_PRUNING):
        """
        Calculates Rvd in 1/cm.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Rz.

        Returns
        -------
        Rvd : float
        """
        return self._parameter(invert=False, pruning=pruning, significant='All', n_significant=1,
                               attribute='Count', statistic='Density')

    @cache
    def Rmpc(self, pruning=DEFAULT_PRUNING):
        """
        Calculates Rmpc in 1/µm.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Rz.

        Returns
        -------
        Rmpc : float
        """
        return self._parameter(invert=True, pruning=pruning, significant='All', n_significant=1,
                               attribute='Curvature', statistic='Mean')

    @cache
    def Rmvc(self, pruning=DEFAULT_PRUNING):
        """
        Calculates Rmvc in 1/µm.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Rz.

        Returns
        -------
        Rmvc : float
        """
        return self._parameter(invert=False, pruning=pruning, significant='All', n_significant=1,
                               attribute='Curvature', statistic='Mean')

    @cache
    def R5p(self, pruning=DEFAULT_PRUNING):
        """
        Calculates R5p in µm.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Rz.

        Returns
        -------
        R5p : float
        """
        return self._parameter(invert=True, pruning=pruning, significant='Top', n_significant=_N_FIVE_POINT,
                               attribute='PVh', statistic='Mean')

    @cache
    def R5v(self, pruning=DEFAULT_PRUNING):
        """
        Calculates R5v in µm.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Rz.

        Returns
        -------
        R5v : float
        """
        return self._parameter(invert=False, pruning=pruning, significant='Bot', n_significant=_N_FIVE_POINT,
                               attribute='PVh', statistic='Mean')

    @cache
    def R10z(self, pruning=DEFAULT_PRUNING):
        """
        Calculates R10z in µm.

        Parameters
        ----------
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Rz.

        Returns
        -------
        R10z : float
        """
        return self.R5p(pruning=pruning) + self.R5v(pruning=pruning)

    def plot_segmentation(self, kind='dale', pruning=DEFAULT_PRUNING, ax=None):
        """
        Plots the profile together with the boundaries of the significant motifs and their critical points.

        Parameters
        ----------
        kind : {'dale', 'hill'}, default 'dale'
            Whether to plot the dale (pit) or hill (peak) segmentation.
        pruning : float, default 5
            Wolf pruning threshold as a percentage of Rz.
        ax : matplotlib axis, default None
            If specified, the plot is drawn on the given axis.

        Returns
        -------
        plt.Figure, plt.Axes
        """
        if kind not in ('dale', 'hill'):
            raise ValueError("kind must be either 'dale' or 'hill'.")
        motifs = self._segment(invert=(kind == 'hill'), pruning=pruning)
        step = self._profile.step
        x = np.arange(self._data.size) * step

        if ax is None:
            fig, ax = plt.subplots(dpi=150, figsize=(10, 3))
        else:
            fig = ax.figure
        ax.plot(x, self._data, c='k', lw=1)
        if len(motifs):
            # The motifs are bounded by their enclosing peaks (dales) or pits (hills). A critical point on a plateau
            # lies at its middle, which is a fractional sample index for an even number of samples; np.interp then
            # returns the plateau value.
            for boundary in np.unique(np.concatenate([motifs.ilp, motifs.ihp])) * step:
                ax.axvline(boundary, c='k', lw=0.5)
            positions = motifs.iv * step
            heights = np.interp(positions, x, self._data)
            ax.scatter(positions, heights, c='white', edgecolors='k', marker='s', s=20, zorder=3,
                       label='pits' if kind == 'dale' else 'peaks')
            ax.legend(loc='upper right', fontsize=6)
        ax.set_xlim(0, self._profile.length_um)
        ax.set_xlabel('x / µm')
        ax.set_ylabel('z / µm')
        return fig, ax
