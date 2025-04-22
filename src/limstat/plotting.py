import numpy as np
import warnings
from astropy import units
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_ps2d(
        pspec_2d, kperp_bins, kpara_bins, dimless=False,
        label=r'P($k_\parallel$,$k_\perp$) [(Jy/beam)$^2$ Mpc$^3$]',
        little_h=False,
        norm=None, title=None, cmap='viridis', ax=None,
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
        little_h: bool
            Whether units are Mpc/h or not.
            Default is False.
        norm: matplotlib.colors.Normalize object.
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
    assert np.shape(pspec_2d) == (kpara_bins.size, kperp_bins.size), \
        "Input pspec must have shape (kperp_bins.size, kpara_bins.size)."
    if np.any(pspec_2d < 0):
        warnings.warn(
            'There are negative values in your pspec. '
            'Absolute value will be used for the figure.'
        )
        pspec_2d = np.abs(pspec_2d)
    if little_h:
        h = 'h'
    else:
        h = ''

    existing_axis = True
    if ax is None:
        fig, ax = plt.subplots(1, 1,)
        existing_axis = False
    if dimless:
        k = np.sqrt(kperp_bins[None, :]**2 + kpara_bins[:, None]**2)
        pspec_2d *= k**3 /2./np.pi**2
        label = r'$\Delta^2(k)$ [K$^2$]'

    im = ax.pcolor(
        kperp_bins,
        kpara_bins,
        pspec_2d,
        shading='nearest',
        cmap=cmap,
        norm=norm,
        edgecolors='w', linewidth=0.5,
    )
    if not existing_axis:
        plt.colorbar(im, label=label, ax=ax)
        ax.set_ylabel(rf'k$_\parallel$ [{h}Mpc$^{{-1}}]$')
        ax.set_xlabel(rf'k$_\perp$ [{h}Mpc$^{{-1}}]$')
    if title is not None:
        ax.set_title(title)


def plot_ps1d(
        pspec_1d, kbins, yerr=None, 
        title=None, dimless=False, little_h=False,
        ax=None, log=True, **kargs):
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
        little_h: bool
            Whether units are Mpc/h or not.
            Default is False.
        ax: matplotlib.axes object
            Axis to plot the figure on.
            Default is None (new figure and axis are generated).


    """

    assert kbins.size == pspec_1d.size, \
        "pspec_1d and kbins must have identical size."

    m = pspec_1d > 0.
    existing_axis = True
    if ax is None:
        existing_axis = False
        fig, ax = plt.subplots()
    # ls = plot_kwargs.get("ls", '-')
    # color = plot_kwargs.get("color", 'C0')
    # lw = plot_kwargs.get("lw", 1.5)
    # label = plot_kwargs.get("label", None)
    if little_h:
        h = 'h'
    else:
        h = ''

    if dimless:
        ax.errorbar(kbins[m], kbins[m]**3 * pspec_1d[m]/2./np.pi**2, **kargs)
        ylabel = r'$\Delta^2(k)$ [K$^2$]'
    else:
        ax.errorbar(kbins[m], pspec_1d[m], **kargs)
        if little_h:
            ylabel = r'$P(k)$ [K$^2$ $h^{-3}$Mpc$^3$]'
        else:
            ylabel = r'$P(k)$ [K$^2$ Mpc$^3$]'
    if log:
        ax.set_yscale('log')
    ax.set_xscale('log')
    if (title is not None) and not existing_axis:
        ax.set_title(title)
    ax.set_xlabel(rf'$k$ [{h}Mpc$^{{-1}}]$')
    ax.set_ylabel(ylabel)

def plot_map(
    box, fov, ifreq=None, 
    label=r'$T$ [K]', cmap='RdBu_r', 
    title=None, norm=None, ax=None,
    uv=False, cosmo_space=False):
    """
    Method to plot 2D sky map from lightcone.

    Parameters
    ----------
        box: 2D or 3D array of floats
            Array containing the lightcone.
            Dimensions (npix, npix, nfreqs).
        fov: float
            Field of view corresponding to the image.
            Must have units.
            If cosmo_space is True, then should be a
            comoving size, in units eq. to Mpc.
        ifreq: int
            Which frequency channel to plot if box is 3D.
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
        uv: boolean
            Whether you are plotting a (u, v) map or not.
            Default: False.
        cosmo_space: float
            Whether the map is in cosmological space or not.
            Default is False.

    """
    ang_res = fov / box.shape[0]
    if box.ndim == 3:
        if ifreq is None:
            ifreq = box.shape[-1]//2
        else:
            assert ifreq < box.shape[-1], \
                "ifreq must be smaller than box.shape[-1]."
        image = box[:, :, ifreq]
    elif box.ndim == 2:
        image = np.copy(box)
    else:
        raise ValueError('box must be of dimension 2 or 3.')

    if uv:
        xlin = np.linspace(
            1.22/fov.to(units.rad).value,
            1.22/ang_res.to(units.rad).value,
            box.shape[0]
        )
    elif cosmo_space:
        xlin = np.linspace(
            0,
            fov.value,
            box.shape[0]
        )
    else:
        xlin = np.linspace(
            -fov.value/2,
            fov.value/2,
            box.shape[0]
        )
    existing_axis = True
    if ax is None:
        fig, ax = plt.subplots(1, 1,)
        existing_axis = False

    im = ax.pcolor(
        xlin, xlin,
        image,
        shading='auto',
        norm=norm,
        cmap=cmap
    )
    plt.colorbar(im, label=label, ax=ax)
    if uv:
        ax.set_ylabel(rf'$u$')
        ax.set_xlabel(rf'$v$')
    elif cosmo_space:
        ax.set_xlabel(rf'$L_x$ [{fov.unit}]')
        ax.set_ylabel(rf'$L_y$ [{fov.unit}]')
        dx = np.diff(ax.get_xticks())[-1]
        ax.set_yticks(np.arange(xlin.min(), xlin.max(), dx))
        ax.set_xticks(np.arange(xlin.min(), xlin.max(), dx))
    else:
        ax.set_ylabel(rf'$\theta$ [{fov.unit}]')
        ax.set_xlabel(rf'$\theta$ [{fov.unit}]')
    if title is not None:
        ax.set_title(title)
    
