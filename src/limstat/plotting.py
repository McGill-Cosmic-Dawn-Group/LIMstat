import numpy as np
import warnings
from astropy import units
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_ps2d(
        pspec_2d, kperp_bins, kpara_bins, dimless=False,
        label=r'P($k_\parallel$,$k_\perp$) [(Jy/beam)$^2$ Mpc$^3$]',
        vmin=None, vmax=None, title=None, cmap='viridis', ax=None,
    ):
    """
    Method to plot cylindrical power spectrum with logarithmic colorbar.

    Parameters
    ----------
        pspec_2d: 2D array of floats
            Array containing the cylindrical power spectrum.
            Shape: (nkperp, nkpara).
        kperp_bins: array of floats
            Array containing the k_perpendicular bins used
            to compute the cylindrical power spectrum.
            Size: nkperp.
        kpara_bins: array of floats
            Array containing the k_parallel bins used
            to compute the cylindrical power spectrum.
            Size: nkpara.  
        dimless: boolean
            Whether the power spectrum is dimensionless or not.
            Default: False.
        label: str
            Label for the colorbar.
            Default is P(k) in (Jy/beam)2 Mpc3.
        vmin: float
            Minimum value used for the colorbar.
            Default is None.
        vmax: float
            Maximum value used for the colorbar.
            Default is None.
        title: str
            Title for the axis.
            Default is None.
        cmap: str
            Matplotlib colormap to use.
            Default is viridis.
        ax: matplotlib.axes object
            Axis to plot the figure on.
            Default is None (new figure and axis are generated).


    """

    kperp_bins = np.atleast_1d(kperp_bins)
    kpara_bins = np.atleast_1d(kpara_bins)
    assert np.shape(pspec_2d) == (kperp_bins.size, kpara_bins.size), \
        "Input pspec must have shape (kperp_bins.size, kpara_bins.size)."
    if np.any(pspec_2d < 0):
        warnings.warn(
            'There are negative values in your pspec. '
            'Absolute value will be used for the figure.'
        )
        pspec_2d = np.abs(pspec_2d)

    existing_axis = True
    if ax is None:
        fig, ax = plt.subplots(1, 1,)
        existing_axis = False
    if dimless:
        pspec2d *= kperp_bins**2 * kpara_bins *1./2./np.pi**2
        label = r'$\Delta^2(k)$ [K$^2$]'

    im = ax.pcolor(
        kperp_bins,
        kpara_bins,
        pspec_2d,
        shading='auto',
        cmap=cmap,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    )
    if not existing_axis:
        plt.colorbar(im, label=label, ax=ax)
        ax.set_ylabel(r'k$_\parallel$ [Mpc$^{-1}]$')
        ax.set_xlabel(r'k$_\perp$ [Mpc$^{-1}]$')
    if title is not None:
        ax.set_title(title)


def plot_ps1d(pspec_1d, kbins, yerr=None, title=None, dimless=False, ax=None, plot_kwargs={}):
    """
    Method to plot spherical power spectrum.

    Parameters
    ----------
        pspec_1d: 1D array of floats
            Array containing the spherical power spectrum.
            Must have same shape as kbins.
        kbins: array of floats
            Array containing the spherical k-bins used
            Must have same shape as pspec_1d.
        yerr: array of floats (optional)
            Array containing the errors on pspec_1d.
            Must have same shape as pspec_1d.
            Default is None.
        title: str
            Title for the axis.
            Default is None.
        dimless: boolean
            Whether the power spectrum is dimensionless or not.
            Default: False.
        ax: matplotlib.axes object
            Axis to plot the figure on.
            Default is None (new figure and axis are generated).


    """

    assert kbins.size == pspec_1d.size, \
        "pspec_1d and kbins must have identical size."
    if yerr is not None:
        assert kbins.size == yerr.size, \
            "yerr and kbins must have identical size."

    m = pspec_1d > 0.
    if ax is None:
        fig, ax = plt.subplots()
    ls = plot_kwargs.get("ls", '-')
    color = plot_kwargs.get("color", 'C0')
    lw = plot_kwargs.get("lw", 1.5)
    label = plot_kwargs.get("label", None)

    if dimless:
        if yerr is not None:
            yerr = kbins[m]**3 * yerr[m] / 2./np.pi**2
        ax.errorbar(kbins[m], kbins[m]**3 * pspec_1d[m]/2./np.pi**2, yerr=yerr, color=color, marker='.', capsize=2, ls=ls, lw=lw, label=label)
        ylabel = r'$\Delta^2(k)$ [K$^2$]'
    else:
        if yerr is not None:
            yerr = yerr[m]
        ax.errorbar(kbins[m], pspec_1d[m], yerr=yerr, color=color, marker='.', capsize=2, label=label, ls=ls, lw=lw)
        ylabel = r'$P(k)$ [K$^2$ Mpc$^3$]'
    ax.set_yscale('log')
    ax.set_xscale('log')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(r'$k$ [Mpc$^{-1}]$')
    ax.set_ylabel(ylabel)

def plot_map(box, fov, ifreq=None, label=r'$T$ [K]', cmap='RdBu_r', title=None, norm=None, ax=None):
    """
    Method to plot 2D sky map from lightcone.

    Parameters
    ----------
        box: 3D array of floats
            Array containing the lightcone.
            Dimensions (npix, npix, nfreqs).
        fov: float
            Field of view corresponding to the image.
            Must have units.
        ifreq: int
            Which frequency channel to plot.
            Default is None: nfreqs//2.
        label: str
            Label for the colorbar.
            Default is T [K].
        title: str
            Title for the axis.
            Default is None.
        cmap: str
            Matplotlib colormap to use.
            Default is RdBu_r.
        norm: matplotlib.colors.Normalize object.
        ax: matplotlib.axes object
            Axis to plot the figure on.
            Default is None (new figure and axis are generated).


    """
    
    if ifreq is None:
        ifreq = box.shape[-1]//2
    else:
        assert ifreq < box.shape[-1], \
            "ifreq must be smaller than box.shape[-1]."

    xlin = np.linspace(-fov.to(units.deg).value/2, fov.to(units.deg).value/2, box.shape[0])
    existing_axis = True
    if ax is None:
        fig, ax = plt.subplots(1, 1,)
        existing_axis = False

    im = ax.pcolor(
        xlin, xlin,
        box[:, :, ifreq],
        shading='auto',
        norm=norm,
        cmap=cmap
    )
    plt.colorbar(im, label=label, ax=ax)
    ax.set_ylabel(rf'$\theta$ [{fov.unit}]')
    ax.set_xlabel(rf'$\theta$ [{fov.unit}]')
    if title is not None:
        ax.set_title(title)
    
