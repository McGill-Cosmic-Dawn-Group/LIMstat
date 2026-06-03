********************************************************************************
``limstat``: simulation and statistical analysis tools for line intensity mapping
********************************************************************************

``limstat`` is a Python package for building end-to-end line intensity mapping
(LIM) simulations and estimating their statistical observables. It was developed
for the forecasting and error-propagation framework described in
`Fronenberg & Liu (2024), Forecasts and Statistical Insights for Line Intensity
Mapping Cross-Correlations: A Case Study with 21cm x [CII]
<https://arxiv.org/abs/2407.14588>`_.

The package is organized around a modular LIM analysis workflow:

* define survey geometry and cosmological conversion factors;
* generate Gaussian random-field signal cubes from input auto-spectra;
* generate pairs of correlated fields from a scale-dependent correlation
  coefficient;
* add foreground contaminants and thermal noise;
* model simple single-dish and interferometric instrumental responses;
* estimate auto- and cross-power spectra in 3D, cylindrical, and spherical bins;
* compute power-spectrum window functions associated with point-spread functions.

The components can be used together as an end-to-end Monte Carlo pipeline, or
individually when only one part of the simulation or analysis is needed.

.. inclusion-marker-installation-do-not-remove

Installation
============

For users
---------

We recommend installing ``limstat`` in a fresh virtual environment:

::

   $ conda create -n limstat python=3.11
   $ conda activate limstat
   $ git clone https://github.com/McGill-Cosmic-Dawn-Group/LIMstat.git
   $ cd LIMstat
   $ python3 -m pip install .

This installs the package and its required dependencies.

For developers
--------------

For local development, install in editable mode with the development extras:

::

   $ python3 -m pip install -e .[dev]

Dependencies
^^^^^^^^^^^^

Core dependencies include ``numpy``, ``scipy``, ``astropy``,
``cached_property``, ``healpy``, ``uvtools``, ``matplotlib``, and
``deprecated``. If you are using ``conda``, you may wish to install these
manually from conda-forge before installing the package:

::

   $ conda install -c conda-forge numpy astropy cached_property scipy healpy uvtools matplotlib deprecated


Running Tests
^^^^^^^^^^^^^

``limstat`` uses ``pytest``. From the repository root, run:

::

   $ pytest

.. exclusion-marker-installation-do-not-remove

Package Overview
================

``limstat`` currently exposes the following main modules:

``limstat.cosmo_units``
    Converts between observational coordinates and comoving coordinates for a
    coeval LIM cube. The ``cosmo_units`` class stores box lengths, pixel sizes,
    Fourier-space resolution, volume elements, redshift, and frequency metadata.

``limstat.simulations``
    Contains simulation models for cosmological signal cubes, correlated signal
    pairs, 21 cm foregrounds, CO interloper foregrounds, and interferometric
    thermal noise. The cosmological signal model takes either a callable power
    spectrum or a two-row ``[k, P(k)]`` array.

``limstat.instruments`` and ``limstat.fast_interferometer``
    Provide simple instrumental response models. ``single_dish_instrument``
    convolves sky cubes with a diffraction-limited Gaussian beam.
    ``fast_interferometer`` simulates interferometric imaging from antenna
    positions: baseline geometry, half-wave UV gridding, redundant-baseline
    counting, dirty maps, PSFs, thermal noise, optional user-supplied UV count
    maps, and multi-frequency dirty cubes.

``limstat.power_spectrum``
    Estimates auto- and cross-power spectra from 3D cubes. It supports direct
    Fourier-space spectra, cylindrical ``P(k_parallel, k_perp)`` binning,
    spherical ``P(k)`` binning, optional tapering through ``uvtools``, unit
    conversion through ``astropy``, and PSF normalization.

``limstat.window_function``
    Computes approximate cylindrical and spherical power-spectrum window
    functions for a supplied point-spread function.

``limstat.plotting``
    Includes convenience plotting helpers for maps, 1D power spectra, and 2D
    cylindrical power spectra.

Quick Start
===========

The example below creates a coeval survey cube, draws a Gaussian random field
from an input power spectrum, and estimates the resulting 1D and 2D power
spectra.

.. code-block:: python

   import numpy as np
   from astropy import units

   from limstat.cosmo_units import cosmo_units
   from limstat.simulations import cosmological_signal
   from limstat.power_spectrum import power_spectrum


   def gaussian_ps(k, mu=0.8, sigma=0.1, amp=1e-2):
       """Toy input power spectrum in K^2 Mpc^3."""
       return amp * np.exp(-0.5 * ((k - mu) / sigma) ** 2)


   npix = 64
   nfreqs = 64
   ang_res = 15 * units.arcsec
   fov = (npix * ang_res).to(units.rad)

   freqs = np.linspace(142, 158, nfreqs) * units.MHz

   cu = cosmo_units(
       x_npix=npix,
       y_npix=npix,
       theta_x=fov,
       theta_y=fov,
       freqs=freqs,
       rest_freq=1420 * units.MHz,
   )

   signal = cosmological_signal(
       ps=gaussian_ps,
       cosmo_units=cu,
   )
   cube = signal.make_universe()

   pspec = power_spectrum(
       data=cube * units.K,
       cosmo_units=cu,
   )

   k_1d, p_1d = pspec.compute_1D_pspec()
   k_par, k_perp, p_2d = pspec.compute_2D_pspec()

Correlated Fields
=================

For LIM cross-correlations, ``cosmological_signal`` can generate two fields
with a chosen scale-dependent correlation coefficient ``r(k)``. Internally, the
second field is generated by applying the desired auto-spectrum ratio and a
random phase model whose variance is set by ``r(k)``. This follows the
decorrelation formalism used in the accompanying publication.

.. code-block:: python

   def r_of_k(k):
       """Toy model: anti-correlated on large scales, correlated on small scales."""
       return np.where(k < 1.0, -0.8, 0.8)


   cube_a, cube_b = signal.make_correlated_universes(r_of_k)

   cross = power_spectrum(
       data=cube_a * units.K,
       data2=cube_b * units.K,
       cosmo_units=cu,
   )

   cross_cube = cross.FFT_crossxy()
   k_cross, p_cross = cross.compute_1D_pspec(ps_data=cross_cube)

Interferometer imaging
======================

``limstat.fast_interferometer`` maps sky models through a UV-domain pipeline
(instantaneous uv coverage, no rotation synthesis). Sky maps must have shape
``(y_npix, x_npix)`` matching the instrument constructor. Set ``T_sys``,
``t_obs``, and ``bandwidth`` for noise.

Typical workflow:

* ``get_bls(freq)`` — baseline vectors and wavelength-scaled ``u``, ``v``
* ``get_uvmap_halfwave(freq)`` — grid coverage on a half-wave lattice
  (``du = dv = 0.5``); stores ``count_map`` for redundant baselines
* ``get_dirty_map(sky_map, freq)`` — noiseless or noisy dirty map on the sky grid
* ``get_psf(freq)`` — dirty beam (PSF)
* ``get_dirty_cube(sky_cube, freqs)`` — image a cube with frequency on the
  **last** axis ``(y_npix, x_npix, n_freq)``, recomputing UV coverage at each
  channel

**Custom UV.** Supply bin edges and a per-bin count map ``N`` instead of
antenna binning. Coverage is derived automatically as a binary mask
(``uv_map = 1`` where ``N > 0``). Pass either a tuple or a dict:

.. code-block:: python

   from limstat.fast_interferometer import fast_interferometer

   inst = fast_interferometer(
       ant_locs=ant_locs,
       theta_x=fov,
       theta_y=fov,
       x_npix=npix,
       y_npix=npix,
       T_sys=200 * units.K,
       t_obs=1000 * units.hr,
       bandwidth=8 * units.MHz,
   )

   freq = 150 * units.MHz
   dirty = inst.get_dirty_map(true_sky, freq, noise=False)

   # User count map N on a UV grid (u_grid, v_grid are bin edges in wavelengths)
   N_uv = (u_grid, v_grid, count_map)
   dirty = inst.get_dirty_map(true_sky, freq, N_uv=N_uv)

   # Multi-frequency cube: UV coverage updates each channel (u = b nu / c)
   freqs = np.linspace(140, 160, 10) * units.MHz
   sky_cube = np.stack([true_sky] * len(freqs), axis=2)
   dirty_cube = inst.get_dirty_cube(sky_cube, freqs, noise=True, redundancy=True)

1D array layouts (east–west or north–south only) are detected automatically
(``array_layout`` of ``'ew_only'`` or ``'ns_only'``).

Tutorials
=========

Worked examples live in the ``tutorials`` directory. The primary tutorial,
``tutorial_cosmo_signal.ipynb``, demonstrates:

* defining a simulation volume with ``cosmo_units``;
* generating Gaussian random-field cubes;
* comparing recovered and input power spectra;
* generating two correlated LIM fields;
* estimating auto- and cross-power spectra.

``tutorial_interferometer.ipynb`` demonstrates ``fast_interferometer`` end to
end: antenna layouts, half-wave UV maps, dirty maps, PSFs, thermal noise,
custom ``N_uv`` injection, multi-frequency ``get_dirty_cube``, and 1D
east–west / north–south fringe examples.

The ``Testing 3D Pspec + Window.ipynb`` notebook is an exploratory notebook for
3D power-spectrum and window-function calculations.

Citation
========

If you use ``limstat`` in work related to the published cross-correlation
forecasting framework, please cite:

.. code-block:: bibtex

   @article{FronenbergLiu2024,
     title = {Forecasts and Statistical Insights for Line Intensity Mapping Cross-Correlations: A Case Study with 21cm x [CII]},
     author = {Fronenberg, Hannah and Liu, Adrian},
     year = {2024},
     eprint = {2407.14588},
     archivePrefix = {arXiv},
     primaryClass = {astro-ph.CO}
   }

Development Status
==================

``limstat`` is research software under active development. The code is most
mature for Gaussian signal simulations, correlated fields, thermal-noise
realizations, power-spectrum estimation, and ``fast_interferometer`` dirty-map
imaging. Some foreground and window-function tools are still evolving and
should be validated for a given science analysis.

