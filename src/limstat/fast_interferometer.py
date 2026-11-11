import warnings

import numpy as np
from astropy import units
from scipy.signal.windows import get_window


def _fft_centred(array, measure, forward):
    """FFT with the shared shift convention, scaled by ``measure``.

    ``forward=True`` uses ``fft2``. ``forward=False`` uses ``ifft2``.
    """
    centred = np.fft.ifftshift(array, axes=(0, 1))
    spectrum = np.fft.fft2(centred) if forward else np.fft.ifft2(centred)
    return np.fft.fftshift(spectrum, axes=(0, 1)) * measure


_WINDOWS_WITHOUT_SHAPE = frozenset({
    'blackmanharris', 'hann', 'hamming', 'blackman', 'bartlett',
})
_WINDOW_SHAPE_KEY = {
    'kaiser': 'beta',
    'tukey': 'alpha',
    'gaussian': 'std',
}
_TAPER_PARAMETER_DOC = """\
        taper : str or callable, optional
            Window applied to the gridded UV weights after sampling.
            ``None`` leaves the weights unchanged. A string selects a
            scipy window placed on the bounding box of the nonzero
            weights. ``scale`` is accepted for every named window. It
            sets the window length relative to that box: ``1`` puts the
            window end on the coverage edge, larger than ``1`` is
            gentler, and smaller than ``1`` reaches zero inside the
            coverage. A coverage of only a few cells has those cells on
            the box edge, so ``scale=1`` downweights them.
            ``blackmanharris``, ``hann``, ``hamming``, ``blackman``, and
            ``bartlett`` take only ``scale``.
            ``kaiser`` also takes ``beta``, ``tukey`` also takes
            ``alpha``, and ``gaussian`` also takes ``std`` as a fraction
            of the window length. A callable receives the UV weights and
            returns a window of the same shape; ``scale`` is not applied
            again. An unknown name, an unknown key, or a shape key
            passed to a window that does not use it raises
            ``ValueError``. ``count_map`` stays the raw sample counts.
        taper_kwargs : dict, optional
            ``scale`` (default 1, must be positive) and, for the windows
            that use one, ``beta`` (Kaiser), ``alpha`` (Tukey), or
            ``std`` (Gaussian, fraction of the window length). Ignored
            when ``taper`` is ``None``. A callable rejects
            ``taper_kwargs``.
"""


def _append_taper_doc(method):
    method.__doc__ = method.__doc__.replace(
        '__TAPER_PARAMETERS__', _TAPER_PARAMETER_DOC.strip('\n'),
    )
    return method


def _parse_taper(taper, taper_kwargs):
    """Return None, ('callable', fn), or ('window', name, scale, shape)."""
    if taper is None:
        return None
    if taper_kwargs is not None and not isinstance(taper_kwargs, dict):
        raise TypeError('taper_kwargs must be a dict or None.')
    if callable(taper):
        if taper_kwargs:
            raise ValueError(
                'taper_kwargs is not used when taper is a callable. '
                'The callable returns the window.',
            )
        return ('callable', taper)
    if not isinstance(taper, str):
        raise TypeError('taper must be None, a window name, or a callable.')

    name = taper.strip().lower()
    known = _WINDOWS_WITHOUT_SHAPE | frozenset(_WINDOW_SHAPE_KEY)
    if name not in known:
        choices = ', '.join(sorted(known))
        raise ValueError(f'Unknown taper {taper!r}. Choose from: {choices}.')

    kwargs = {} if taper_kwargs is None else dict(taper_kwargs)
    scale = kwargs.pop('scale', 1.0)
    try:
        scale = float(scale)
    except (TypeError, ValueError) as err:
        raise ValueError('taper scale must be a positive number.') from err
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError('taper scale must be positive.')

    shape_key = _WINDOW_SHAPE_KEY.get(name)
    if shape_key is None:
        if kwargs:
            unknown = ', '.join(sorted(str(key) for key in kwargs))
            raise ValueError(
                f'{name} accepts only scale. Unexpected taper_kwargs: {unknown}.',
            )
        return ('window', name, scale, None)

    if shape_key not in kwargs:
        raise ValueError(f'{name} requires taper_kwargs {shape_key!r}.')
    shape_value = kwargs.pop(shape_key)
    if kwargs:
        unknown = ', '.join(sorted(str(key) for key in kwargs))
        raise ValueError(f'Unexpected taper_kwargs: {unknown}.')
    return ('window', name, scale, shape_value)


def _scaled_window(name, n, scale, shape_value):
    """Symmetric 1D window of length ``n``, built at ``round(n * scale)``."""
    n_win = max(1, int(round(n * scale)))
    if name == 'gaussian':
        window = ('gaussian', float(shape_value) * n_win)
    elif name == 'kaiser':
        window = ('kaiser', float(shape_value))
    elif name == 'tukey':
        window = ('tukey', float(shape_value))
    else:
        window = name
    full = np.asarray(get_window(window, n_win, fftbins=False), dtype=float)
    if n_win == n:
        return full
    if n_win > n:
        start = (n_win - n) // 2
        return full[start:start + n]
    placed = np.zeros(n, dtype=float)
    start = (n - n_win) // 2
    placed[start:start + n_win] = full
    return placed


def _aperture_window(uv_map, name, scale, shape_value):
    """Separable window on the bounding box of nonzero UV weights."""
    weights = np.asarray(uv_map)
    window = np.zeros(weights.shape, dtype=float)
    mask = np.abs(weights) > 0
    if not np.any(mask):
        return window
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    wy = _scaled_window(name, y1 - y0, scale, shape_value)
    wx = _scaled_window(name, x1 - x0, scale, shape_value)
    window[y0:y1, x0:x1] = np.outer(wy, wx)
    return window


class fast_interferometer(object):
    """
    Simulate UV coverage, dirty maps, PSF, and thermal noise.

    The UV weight is the sample count in each cell (natural weighting).
    A dirty map is the inverse transform of ``weight * V``, divided by
    ``sum(weight) * du * dv * dl * dm``. A weight of 1 on every sky mode
    returns the sky. A constant weight scales the PSF and this divisor
    together, so a 1 K point source still peaks at 1.

    ``get_psf`` is the inverse transform of the weights and is not divided
    by that factor. ``peak_normalize`` divides the PSF by its maximum.

    Visibility noise is
    ``T_sys * theta_x * theta_y / sqrt(bandwidth * t_obs)``.
    ``get_noise_map`` splits that across the real and imaginary parts,
    multiplies by the count, and divides by the same factor. With
    ``redundancy=True`` the draw is ``sigma / sqrt(count)``, so the image
    rms scales as ``1/sqrt(count)``. Noise added inside ``get_dirty_map``
    is not split by ``sqrt(2)``.

    How the sky is padded onto the UV grid is described in
    ``docs/fast_interferometer_gridding.md``.
    """

    def __init__(
        self,
        ant_locs=None,
        theta_x=None,
        theta_y=None,
        x_npix=None,
        y_npix=None,
        T_sys=None,
        t_obs=None,
        bandwidth=None,
    ):
        """
        Parameters
        ----------
        ant_locs : array_like, optional
            Antenna coordinates in meters, shape (n_ants, 2). Pass with astropy
            units. Required for antenna-based UV gridding; omit when using
            ``N_uv``.
        theta_x, theta_y : Quantity
            Field of view along x and y (converted to radians internally).
        x_npix, y_npix : int
            Number of sky pixels along x and y; ``sky_map.shape`` must be ``(y_npix, x_npix)``.
        T_sys : Quantity, optional
            System temperature in K (required for noise).
        t_obs : Quantity, optional
            Observation time (required for noise).
        bandwidth : Quantity, optional
            Bandwidth of the input map in Hz (required for noise).
        """
        if theta_x is None or theta_y is None or x_npix is None or y_npix is None:
            raise ValueError(
                'theta_x, theta_y, x_npix, and y_npix are required.',
            )

        self.ants = None
        if ant_locs is not None:
            self._set_ants(ant_locs)
        self.theta_x = theta_x.to(units.rad).value
        self.theta_y = theta_y.to(units.rad).value
        self.x_npix = x_npix
        self.y_npix = y_npix

        if T_sys is not None:
            self.T_sys = T_sys.to(units.K).value
        if t_obs is not None:
            self.t_obs = t_obs.to(units.s).value
        if bandwidth is not None:
            self.bandwidth = bandwidth.to(units.Hz).value

        self._sky_coords_cache = None
        self.uv_mode = None

    def _set_ants(self, ant_locs):
        """Store antenna coordinates in meters."""
        self.ants = ant_locs.to(units.m).value
        return self.ants

    def _require_ants(self, ant_locs=None):
        """Return stored antenna coords, updating from ``ant_locs`` if given."""
        if ant_locs is not None:
            self._set_ants(ant_locs)
        if self.ants is None:
            raise ValueError(
                'ant_locs are required for antenna-based UV gridding. '
                'Pass ant_locs to the constructor or this method, '
                'or provide N_uv instead.'
            )
        return self.ants

    def get_bls(self, freq, ant_locs=None):
        """
        Unique baseline vectors in metres, and ``u``, ``v`` in wavelengths.

        Each pair of antennas contributes a baseline and its conjugate.
        Autocorrelations are omitted. ``u = b / lambda``.
        """
        n_ants = self._require_ants(ant_locs).shape[0]
        i, j = np.triu_indices(n_ants, k=1)
        bls = self.ants[i] - self.ants[j]
        total_bls = np.vstack([bls, -bls])

        # Normalize -0 to 0 for stable np.unique
        total_bls = np.where(total_bls == 0, 0, total_bls)

        n_expected = n_ants * (n_ants - 1)
        if len(total_bls) != n_expected:
            raise ValueError('The total number of baselines is not correct.')

        self.unique_bls, self.counts = np.unique(
            total_bls, axis=0, return_counts=True,
        )
        
        print(f'{total_bls.shape[0]} total baseline vectors, {self.unique_bls.shape[0]} unique baseline vectors')

        frequency = freq.to(units.Hz).value
        wavelength = 3e8 / frequency
        self.u_coords = self.unique_bls[:, 0] / wavelength
        self.v_coords = self.unique_bls[:, 1] / wavelength
        
        return self.unique_bls

    def _parse_n_uv(self, N_uv=None):
        """Return (u_grid, v_grid, count_map) or None if using antenna binning."""
        if N_uv is None:
            return None
        if len(N_uv) != 3:
            raise ValueError(
                'N_uv must be a 3-tuple (u_grid, v_grid, count_map).',
            )
        u_grid, v_grid, count_map = N_uv
        return (
            np.asarray(u_grid, dtype=float),
            np.asarray(v_grid, dtype=float),
            np.asarray(count_map, dtype=float),
        )

    def _edges_from_centres(self, centres):
        """Bin edges whose midpoints are ``centres``."""
        step = np.diff(centres)
        edges = np.empty(len(centres) + 1, dtype=float)
        edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
        edges[0] = centres[0] - 0.5 * step[0]
        edges[-1] = centres[-1] + 0.5 * step[-1]
        return edges

    def _as_n_uv_edges(self, u_grid, v_grid, count_map):
        """
        Return bin edges for ``N_uv``.

        ``count_map.shape == (len(v) - 1, len(u) - 1)`` means ``u`` and ``v``
        are already edges. ``count_map.shape == (len(v), len(u))`` means they
        are cell centres; edges are built from those centres and a warning is
        issued.
        """
        if u_grid.ndim != 1 or v_grid.ndim != 1:
            raise ValueError('u_grid and v_grid must be 1D arrays.')
        if len(u_grid) < 2 or len(v_grid) < 2:
            raise ValueError(
                'u_grid and v_grid must each have at least two samples.',
            )
        if np.any(np.diff(u_grid) <= 0) or np.any(np.diff(v_grid) <= 0):
            raise ValueError('u_grid and v_grid must be strictly increasing.')

        edge_shape = (len(v_grid) - 1, len(u_grid) - 1)
        centre_shape = (len(v_grid), len(u_grid))
        if count_map.shape == edge_shape:
            return u_grid, v_grid
        if count_map.shape == centre_shape:
            warnings.warn(
                'N_uv u_grid and v_grid have the same length as '
                f'count_map.shape {count_map.shape}, so they are treated as '
                'bin centres. Pass edges one sample longer than the count '
                'map to skip this.',
                UserWarning,
                stacklevel=4,
            )
            return (
                self._edges_from_centres(u_grid),
                self._edges_from_centres(v_grid),
            )
        raise ValueError(
            f'count_map shape {count_map.shape} must be {edge_shape} for '
            f'bin edges or {centre_shape} for bin centres.',
        )

    def _apply_n_uv(self, u_grid, v_grid, count_map):
        """
        Place ``count_map`` on the sky UV extent at the finer cell size.

        ``ant_locs`` are not used. Counts outside the sky Nyquist limit
        are dropped.
        """
        u_grid, v_grid = self._as_n_uv_edges(u_grid, v_grid, count_map)
        du_in = float(np.median(np.diff(u_grid)))
        dv_in = float(np.median(np.diff(v_grid)))
        u_cent = 0.5 * (u_grid[:-1] + u_grid[1:])
        v_cent = 0.5 * (v_grid[:-1] + v_grid[1:])
        self.uv_mode = 'nuv'
        self._set_matched_grid(du_in, dv_in)
        if count_map.shape[1] == 1:
            self.array_layout = 'ns_only'
        elif count_map.shape[0] == 1:
            self.array_layout = 'ew_only'
        else:
            self.array_layout = '2d'
        same_cell = (
            abs(self.du - du_in) <= 1e-8 * du_in
            and abs(self.dv - dv_in) <= 1e-8 * dv_in
        )
        if same_cell:
            uv_map = self._embed_count_map(u_grid, v_grid, count_map)
        else:
            uv_map = self._beam_pad_counts(u_cent, v_cent, count_map, du_in, dv_in)
        return uv_map

    def get_uvmap_halfwave(self, freq, N_uv=None, ant_locs=None):
        """
        Build the working UV coverage map.

        Antenna path
            Bin baselines onto a half-wave lattice (du=dv=0.5) whose extent is
            the Nyquist limit of the sky pixels. The sky is zero-padded to the
            image that Fourier-transforms onto this lattice; there is no
            interpolation.

        N_uv path
            The working cell is the finer of the supplied spacing and the
            sky's own FFT spacing, and the axis stops at the sky Nyquist
            limit. A finer supplied spacing zero-pads the sky image. A
            coarser count map is inverse-transformed, zero-padded, and
            transformed back onto that grid. Counts outside the sky
            Nyquist limit are dropped. ``ant_locs`` are not used.

        Sets ``self.u_grid``, ``self.v_grid``, ``self.count_map``, and
        ``self.array_layout`` (``'2d'``, ``'ew_only'``, or ``'ns_only'``).
        Those grids are the padded hybrid plane, not the input edges.
        ``docs/fast_interferometer_gridding.md`` shows where each path
        pads and where the transforms are taken.

        Parameters
        ----------
        freq : Quantity
            Observing frequency (used for antenna-based gridding only).
        N_uv : tuple, optional
            ``(u_grid, v_grid, count_map)`` in wavelengths. ``u_grid`` and
            ``v_grid`` are bin edges when ``count_map`` is one cell shorter
            on each axis, or bin centres when the lengths match. Centres
            produce a warning and are converted to edges.
        ant_locs : array_like, optional
            Antenna coordinates for the antenna path. Overrides constructor
            ``ant_locs`` if given.
        """
        parsed = self._parse_n_uv(N_uv)
        if parsed is not None:
            return self._apply_n_uv(*parsed)

        self.get_bls(freq, ant_locs=ant_locs)
        self.uv_mode = 'antenna_halfwave'
        dl, dm = self._set_hybrid_grid(0.5, 0.5)
        self.array_layout = self._array_layout_from_baselines()
        uv_map = self._grid_baselines_on_halfwave(dl, dm)
        return uv_map

    def _apply_taper(self, uv_map, taper, taper_kwargs):
        """Multiply gridded UV weights by a window. ``count_map`` is unchanged."""
        parsed = _parse_taper(taper, taper_kwargs)
        if parsed is None:
            return uv_map
        if parsed[0] == 'callable':
            window = np.asarray(parsed[1](uv_map), dtype=float)
            if window.shape != np.shape(uv_map):
                raise ValueError(
                    f'taper returned shape {window.shape}, '
                    f'expected {np.shape(uv_map)}.',
                )
            return np.asarray(uv_map, dtype=float) * window
        window = _aperture_window(uv_map, parsed[1], parsed[2], parsed[3])
        return np.asarray(uv_map, dtype=float) * window

    @_append_taper_doc
    def get_psf(
        self, freq, N_uv=None, ant_locs=None,
        peak_normalize=False, taper=None, taper_kwargs=None,
        ):
        """Point spread function (dirty beam) on the sky pixel grid.

        This is the inverse transform of the gridded weights. It is not
        divided by ``sum(weight) * du * dv * dl * dm``. The dirty map of
        a 1 K point source equals this PSF divided by its maximum.

        Parameters
        ----------
        freq : Quantity
            Observing frequency.
        N_uv : tuple, optional
            ``(u_grid, v_grid, count_map)`` in wavelengths. Edges are one
            sample longer than the count map; matching lengths are centres.
        ant_locs : array_like, optional
            Antenna coordinates for the antenna path. Overrides constructor
            ``ant_locs`` if given.
        peak_normalize : bool, optional
            Divide the PSF by its maximum when True.
        __TAPER_PARAMETERS__
        """
        uv_map = self.get_uvmap_halfwave(
            freq, N_uv=N_uv, ant_locs=ant_locs,
        )
        uv_map = self._apply_taper(uv_map, taper, taper_kwargs)
        psf = self._ifft_uv_to_sky(uv_map)
        if peak_normalize:
            psf /= np.nanmax(psf)
        return psf.real

    def sky_uv_coords(self):
        """Cached (u, v, l, m, dl, dm) for the natural sky FFT grid."""
        if self._sky_coords_cache is None:
            L = np.sin(self.theta_x)
            M = np.sin(self.theta_y)

            dl = L / self.x_npix
            dm = M / self.y_npix

            l = np.arange(-L / 2, L / 2, dl)
            m = np.arange(-M / 2, M / 2, dm)

            u = np.fft.fftshift(np.fft.fftfreq(len(l), d=dl))
            v = np.fft.fftshift(np.fft.fftfreq(len(m), d=dm))

            self._sky_coords_cache = (u, v, l, m, dl, dm)

        return self._sky_coords_cache

    def _set_hybrid_grid(self, du, dv):
        """
        UV lattice with the given cell size, extending to the sky Nyquist limit.

        The padded image is ``1 / du`` wide in direction cosine, so its FFT
        spacing equals ``du``. ``n_pad = round(1 / (du * dl))``. ``u = 0`` is
        a cell centre. ``du = dv = 0.5`` is the antenna half-wave grid.
        """
        _, _, _, _, dl, dm = self.sky_uv_coords()
        self.du = float(du)
        self.dv = float(dv)
        self.n_pad_x = int(round(1.0 / (self.du * dl)))
        self.n_pad_y = int(round(1.0 / (self.dv * dm)))
        self.u_grid = (
            np.arange(self.n_pad_x + 1) - self.n_pad_x // 2 - 0.5
        ) * self.du
        self.v_grid = (
            np.arange(self.n_pad_y + 1) - self.n_pad_y // 2 - 0.5
        ) * self.dv
        n_bytes = self.n_pad_y * self.n_pad_x * 16
        print(
            f'UV plane {self.n_pad_y} x {self.n_pad_x} '
            f'(du={self.du:g}, dv={self.dv:g}, {n_bytes / 1e6:.1f} MB complex)'
        )
        return dl, dm

    def _sky_sampling(self):
        """Sky pixel size, native FFT spacing, and Nyquist limit."""
        _, _, _, _, dl, dm = self.sky_uv_coords()
        du_sky = 1.0 / (self.x_npix * dl)
        dv_sky = 1.0 / (self.y_npix * dm)
        return dl, dm, du_sky, dv_sky, 0.5 / dl, 0.5 / dm

    def _set_matched_grid(self, du_in, dv_in):
        """
        UV grid at the finer cell size, out to the sky Nyquist limit.

        The cell is the smaller of the supplied spacing and the sky FFT
        spacing. The axis length is ``round(1 / (du dl))``, so the image
        keeps the sky pixel size and the transform stops at ``1 / (2 dl)``.
        """
        _, _, du_sky, dv_sky, _, _ = self._sky_sampling()
        return self._set_hybrid_grid(
            min(float(du_in), du_sky), min(float(dv_in), dv_sky),
        )

    def _paste_centre(self, src, dst):
        """Copy ``src`` into the centre of ``dst``."""
        y0 = dst.shape[0] // 2 - src.shape[0] // 2
        x0 = dst.shape[1] // 2 - src.shape[1] // 2
        self._copy_overlap(src, dst, y0, x0)
        return dst

    def _beam_pad_counts(self, u_cent, v_cent, counts, du_in, dv_in):
        """
        Resample a coarse regular count map onto the working UV grid.

        The count array is transformed as an image. Zero-padding its beam
        refines the cell size to the working spacing while keeping the
        original extent, and the result is pasted onto the sky UV grid so
        unsampled modes stay zero. Cells whose centres lie outside that
        grid are removed first. Nearest-cell binning is not used: samples
        that sit halfway between coarse cells would otherwise merge and
        leave a comb, whose transform is a repeated beam.
        """
        counts = np.asarray(counts, dtype=float).copy()
        iu = np.rint(np.asarray(u_cent) / self.du).astype(np.intp) + self.n_pad_x // 2
        iv = np.rint(np.asarray(v_cent) / self.dv).astype(np.intp) + self.n_pad_y // 2
        inside_u = (iu >= 0) & (iu < self.n_pad_x)
        inside_v = (iv >= 0) & (iv < self.n_pad_y)
        outside = ~inside_v[:, None] | ~inside_u[None, :]
        n_drop = float(np.sum(counts[outside]))
        if n_drop:
            warnings.warn(
                f'{n_drop:g} counts fall outside the sky UV extent '
                'and were dropped.',
                UserWarning,
                stacklevel=4,
            )
            counts[outside] = 0.0

        n_y, n_x = counts.shape
        n_fine_y = max(n_y, int(round(n_y * dv_in / self.dv)))
        n_fine_x = max(n_x, int(round(n_x * du_in / self.du)))
        scale = du_in * dv_in * n_y * n_x
        beam = _fft_centred(counts, scale, forward=False)
        beam_pad = np.zeros((n_fine_y, n_fine_x), dtype=complex)
        self._paste_centre(beam, beam_pad)
        refined = np.real(_fft_centred(beam_pad, 1.0 / scale, forward=True))

        # Index n//2 is the phase centre of the FFT, for even and odd lengths.
        iu_mid = int(np.rint(u_cent[n_x // 2] / self.du)) + self.n_pad_x // 2
        iv_mid = int(np.rint(v_cent[n_y // 2] / self.dv)) + self.n_pad_y // 2
        x0 = iu_mid - n_fine_x // 2
        y0 = iv_mid - n_fine_y // 2
        count_map = np.zeros((self.n_pad_y, self.n_pad_x), dtype=float)
        self._copy_overlap(refined, count_map, y0, x0)
        self.count_map = count_map
        return self.count_map

    def _embed_count_map(self, u_edges, v_edges, counts):
        """Nearest-cell the input counts onto the hybrid plane."""
        u_cent = 0.5 * (u_edges[:-1] + u_edges[1:])
        v_cent = 0.5 * (v_edges[:-1] + v_edges[1:])
        vv, uu = np.meshgrid(v_cent, u_cent, indexing='ij')
        iu, iv = self._halfwave_cell_index(uu, vv)
        flat_counts = np.asarray(counts, dtype=float).ravel()
        iu = iu.ravel()
        iv = iv.ravel()
        inside = self._cells_in_range(iu, iv)
        n_drop = float(np.sum(flat_counts[~inside]))
        if n_drop:
            warnings.warn(
                f'{n_drop:g} counts fall outside the sky UV extent '
                'and were dropped.',
                UserWarning,
                stacklevel=3,
            )
        count_map = np.zeros((self.n_pad_y, self.n_pad_x), dtype=float)
        if np.any(inside):
            np.add.at(count_map, (iv[inside], iu[inside]), flat_counts[inside])
        self.count_map = count_map
        return count_map

    def _array_layout_from_baselines(self):
        if len(self.u_coords) == 0:
            return '2d'
        if np.allclose(self.u_coords, 0.0):
            return 'ns_only'
        if np.allclose(self.v_coords, 0.0):
            return 'ew_only'
        return '2d'

    def _halfwave_cell_index(self, u, v):
        iu = np.rint(np.asarray(u, dtype=float) / self.du).astype(np.intp)
        iv = np.rint(np.asarray(v, dtype=float) / self.dv).astype(np.intp)
        return iu + self.n_pad_x // 2, iv + self.n_pad_y // 2

    def _cells_in_range(self, iu, iv):
        return (
            (iu >= 0) & (iu < self.n_pad_x)
            & (iv >= 0) & (iv < self.n_pad_y)
        )

    def _grid_baselines_on_halfwave(self, dl, dm):
        """Nearest-cell binning. A baseline is kept only if it and its conjugate fit."""
        iu, iv = self._halfwave_cell_index(self.u_coords, self.v_coords)
        iu_conj, iv_conj = self._halfwave_cell_index(-self.u_coords, -self.v_coords)
        inside = (
            self._cells_in_range(iu, iv)
            & self._cells_in_range(iu_conj, iv_conj)
            & (np.abs(self.u_coords) <= 1.0 / (2.0 * dl))
            & (np.abs(self.v_coords) <= 1.0 / (2.0 * dm))
        )
        n_drop = int(np.sum(self.counts[~inside])) if len(self.counts) else 0
        if n_drop:
            warnings.warn(
                f'{n_drop} baseline vectors fall outside the sky Nyquist '
                f'(|u| <= {1.0 / (2.0 * dl):.4g}, |v| <= {1.0 / (2.0 * dm):.4g}) '
                'and were dropped.',
                UserWarning,
                stacklevel=3,
            )
        count_map = np.zeros((self.n_pad_y, self.n_pad_x), dtype=float)
        if np.any(inside):
            np.add.at(
                count_map,
                (iv[inside], iu[inside]),
                self.counts[inside],
            )
        self.count_map = count_map
        return count_map

    def _sky_origin(self):
        """Padded-image index of sky pixel (0, 0). The sky centre stays at n//2."""
        return (
            self.n_pad_y // 2 - self.y_npix // 2,
            self.n_pad_x // 2 - self.x_npix // 2,
        )

    def _copy_overlap(self, src, dst, y0, x0):
        """Copy ``src`` into ``dst`` with ``src[0, 0]`` landing at ``(y0, x0)``."""
        y_src0 = max(0, -y0)
        x_src0 = max(0, -x0)
        y_dst0 = max(0, y0)
        x_dst0 = max(0, x0)
        y_n = min(src.shape[0] - y_src0, dst.shape[0] - y_dst0)
        x_n = min(src.shape[1] - x_src0, dst.shape[1] - x_dst0)
        if y_n > 0 and x_n > 0:
            dst[y_dst0:y_dst0 + y_n, x_dst0:x_dst0 + x_n] = src[
                y_src0:y_src0 + y_n, x_src0:x_src0 + x_n,
            ]
        return dst

    def _pad_sky(self, sky_map):
        """Copy the sky into the padded image by pixel index. No resampling."""
        sky_map = np.asarray(sky_map, dtype=float)
        if sky_map.shape != (self.y_npix, self.x_npix):
            raise ValueError(
                f'sky_map shape {sky_map.shape} must be (y_npix, x_npix) = '
                f'({self.y_npix}, {self.x_npix}).',
            )
        padded = np.zeros((self.n_pad_y, self.n_pad_x), dtype=float)
        y0, x0 = self._sky_origin()
        if (
            y0 < 0 or x0 < 0
            or y0 + self.y_npix > self.n_pad_y
            or x0 + self.x_npix > self.n_pad_x
        ):
            warnings.warn(
                'The sky is wider than the image set by the UV cell size, '
                'so pixels outside that image were not copied into the pad.',
                UserWarning,
                stacklevel=3,
            )
        return self._copy_overlap(sky_map, padded, y0, x0)

    def _crop_to_sky(self, image):
        y0, x0 = self._sky_origin()
        sky = np.zeros((self.y_npix, self.x_npix), dtype=image.dtype)
        return self._copy_overlap(image, sky, -y0, -x0)

    def _ifft_uv_to_sky(self, uv_plane):
        """Inverse FFT a UV plane and crop to the sky pixels."""
        n_y, n_x = uv_plane.shape
        image = _fft_centred(
            uv_plane, self.du * self.dv * n_y * n_x, forward=False,
        )
        return self._crop_to_sky(image)

    def _dirty_map_halfwave(self, sky_map, uv_map, noise, redundancy):
        _, _, _, _, dl, dm = self.sky_uv_coords()
        padded = self._pad_sky(sky_map)
        V = _fft_centred(padded, dl * dm, forward=True)
        if noise:
            noise_draw = self._draw_uv_noise(
                uv_map.shape, uv_map, redundancy, normalize_sqrt2=False,
            )
            dirty_uv = uv_map * (V + noise_draw)
        else:
            dirty_uv = uv_map * V
        dirty_map = self._ifft_uv_to_sky(dirty_uv)
        peak = self._weight_norm(uv_map)
        if peak != 0.0:
            dirty_map = dirty_map / peak
        return np.real(dirty_map)

    def _weight_norm(self, uv_map):
        """Mean weight. A weight of 1 on every sky mode returns the sky.

        The forward transform multiplies by ``dl dm`` and the inverse by
        ``du dv n_y n_x``. That pair already inverts a uniform weight, so
        the divisor is ``sum(W) du dv dl dm``.
        """
        _, _, _, _, dl, dm = self.sky_uv_coords()
        return float(np.sum(uv_map) * self.du * self.dv * dl * dm)

    def _uv_noise_scale(self, redundancy):
        noise_level = self.compute_noise()
        if redundancy:
            scale = np.zeros_like(self.count_map, dtype=float)
            np.divide(
                noise_level,
                np.sqrt(self.count_map),
                where=self.count_map > 0,
                out=scale,
            )
            return scale
        return noise_level

    def _draw_uv_noise(self, shape, uv_map, redundancy, normalize_sqrt2=True):
        """Complex UV noise. ``normalize_sqrt2`` splits ``sigma`` across real and imag."""
        scale = self._uv_noise_scale(redundancy)
        noise_map = (
            np.random.normal(0, scale, shape)
            + 1j * np.random.normal(0, scale, shape)
        )
        if normalize_sqrt2:
            noise_map /= np.sqrt(2)
        return np.where(uv_map != 0, noise_map, 0)

    @_append_taper_doc
    def get_dirty_map(
        self,
        sky_map,
        freq,
        noise=False,
        redundancy=True,
        N_uv=None,
        ant_locs=None,
        taper=None,
        taper_kwargs=None,
        ):
        """
        Dirty map on the sky grid; optional thermal noise in the UV plane.

        The map is divided by ``sum(weight) * du * dv * dl * dm``. A
        weight of 1 on every sky mode returns the sky, and a 1 K point
        source peaks at 1. When ``noise`` is True, each of the real and
        imaginary parts is drawn at the full visibility ``sigma``.
        ``get_noise_map`` divides that draw by ``sqrt(2)``.

        Parameters
        ----------
        sky_map : array_like
            Sky brightness, shape ``(y_npix, x_npix)``.
        freq : Quantity
            Observing frequency.
        noise : bool, optional
            Add thermal noise in the UV plane when True.
        redundancy : bool, optional
            When True, draw the UV noise as ``sigma / sqrt(count)``. The
            image rms then scales as ``1/sqrt(count)``.
        N_uv : tuple, optional
            ``(u_grid, v_grid, count_map)`` in wavelengths. Edges are one
            sample longer than the count map; matching lengths are centres.
            The working grid matches both this map and the sky.
        ant_locs : array_like, optional
            Antenna coordinates for the antenna path. Overrides constructor
            ``ant_locs`` if given.
        __TAPER_PARAMETERS__
        """
        uv_map = self.get_uvmap_halfwave(
            freq, N_uv=N_uv, ant_locs=ant_locs,
        )
        uv_map = self._apply_taper(uv_map, taper, taper_kwargs)
        return self._dirty_map_halfwave(
            sky_map, uv_map, noise, redundancy,
        )

    def _validate_per_channel_uv(self, n_freq, N_uv=None):
        if N_uv is not None and len(N_uv) != n_freq:
            raise ValueError(
                f'N_uv must have length {n_freq} (one entry per frequency channel).',
            )

    @_append_taper_doc
    def get_dirty_cube(
        self,
        sky_cube,
        freqs,
        noise=False,
        redundancy=True,
        N_uv=None,
        ant_locs=None,
        taper=None,
        taper_kwargs=None,
        ):
        """
        Image a multi-frequency sky cube by calling ``get_dirty_map`` per channel.

        Parameters
        ----------
        sky_cube : array_like
            Shape ``(y_npix, x_npix, n_freq)`` in K; frequency on the last axis.
        freqs : array_like
            Observing frequency per channel (astropy Quantity), length ``n_freq``.
        noise : bool
            If True, add independent thermal noise per channel (requires
            ``T_sys``, ``t_obs``, ``bandwidth`` on the instrument).
        redundancy : bool
            Scale noise by ``sqrt(count_map)`` per UV bin when True.
        N_uv : sequence, optional
            Length ``n_freq``; each element is ``(u_grid, v_grid, count_map)``.
        ant_locs : array_like, optional
            Antenna coordinates for the antenna path. Overrides constructor
            ``ant_locs`` if given.
        __TAPER_PARAMETERS__

        Returns
        -------
        dirty_cube : ndarray
            Shape ``(y_npix, x_npix, n_freq)``.

        Notes
        -----
        UV coverage and gridding are recomputed at each frequency (antenna path).
        After the call, ``self`` retains the UV state from the **last** channel.
        Set ``bandwidth`` to the per-channel value if channels are independent.
        """
        sky_cube = np.asarray(sky_cube, dtype=float)
        if sky_cube.ndim != 3:
            raise ValueError(
                'sky_cube must be 3D with shape (y_npix, x_npix, n_freq).',
            )
        if sky_cube.shape[:2] != (self.y_npix, self.x_npix):
            raise ValueError(
                f'sky_cube spatial shape {sky_cube.shape[:2]} must match '
                f'(y_npix, x_npix) = ({self.y_npix}, {self.x_npix}).',
            )

        freqs = np.atleast_1d(freqs)
        n_freq = sky_cube.shape[2]
        if len(freqs) != n_freq:
            raise ValueError(
                f'len(freqs)={len(freqs)} must match n_freq={n_freq}.',
            )

        self._validate_per_channel_uv(n_freq, N_uv=N_uv)

        dirty_cube = np.empty_like(sky_cube)
        for i in range(n_freq):
            n_uv_i = N_uv[i] if N_uv is not None else None
            dirty_cube[:, :, i] = self.get_dirty_map(
                sky_cube[:, :, i],
                freqs[i],
                noise=noise,
                redundancy=redundancy,
                N_uv=n_uv_i,
                ant_locs=ant_locs,
                taper=taper,
                taper_kwargs=taper_kwargs,
            )

        return dirty_cube

    @_append_taper_doc
    def get_noise_map(
        self, freq, redundancy=True, N_uv=None, ant_locs=None,
        taper=None, taper_kwargs=None,
        ):
        """Noise-only image realization on the sky grid.

        ``sigma = T_sys * theta_x * theta_y / sqrt(bandwidth * t_obs)``
        is split across the real and imaginary parts, multiplied by the
        sample count, and divided by
        ``sum(weight) * du * dv * dl * dm``. With ``redundancy=True``
        the draw is ``sigma / sqrt(count)``, so the image rms scales as
        ``1/sqrt(count)``.

        Parameters
        ----------
        freq : Quantity
            Observing frequency.
        redundancy : bool, optional
            Scale the UV noise by ``sqrt(count_map)`` per bin when True.
            ``count_map`` is the raw sample count, before the taper.
        N_uv : tuple, optional
            ``(u_grid, v_grid, count_map)`` in wavelengths.
        ant_locs : array_like, optional
            Antenna coordinates for the antenna path. Overrides constructor
            ``ant_locs`` if given.
        __TAPER_PARAMETERS__
        """
        uv_map = self.get_uvmap_halfwave(
            freq, N_uv=N_uv, ant_locs=ant_locs,
        )
        uv_map = self._apply_taper(uv_map, taper, taper_kwargs)

        noise_map = self._draw_uv_noise(
            uv_map.shape, uv_map, redundancy, normalize_sqrt2=True,
        )

        noise_map *= uv_map # add natural weighting to the noise map

        position_noise_map = self._ifft_uv_to_sky(noise_map)
        peak = self._weight_norm(uv_map)
        if peak != 0.0:
            position_noise_map = position_noise_map / peak
        return np.real(position_noise_map)

    def compute_noise(self):
        """
        Per-visibility noise standard deviation (K).

        ``T_sys * theta_x * theta_y / sqrt(bandwidth * t_obs)``, with
        angles in radians, time in seconds, and bandwidth in hertz.
        """
        if self.T_sys is None or self.t_obs is None or self.bandwidth is None:
            raise ValueError(
                'T_sys, t_obs, and bandwidth must be set to compute noise.',
            )

        try:
            Tsys = self.T_sys.value
        except AttributeError:
            Tsys = self.T_sys

        solid_angle = self.theta_x * self.theta_y
        return (Tsys * solid_angle) / np.sqrt(self.bandwidth * self.t_obs)



