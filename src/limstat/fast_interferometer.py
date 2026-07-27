import numpy as np
from astropy import units
from scipy import stats
import scipy.interpolate


def _nan_to_zero(arr):
    out = np.asarray(arr, dtype=float)
    return np.nan_to_num(out, nan=0.0, copy=True)


class fast_interferometer(object):
    """Simulate UV coverage, dirty maps, PSF, and thermal noise for an interferometer."""

    def __init__(
        self,
        ant_locs,
        theta_x,
        theta_y,
        x_npix,
        y_npix,
        T_sys=None,
        t_obs=None,
        bandwidth=None,
    ):
        """
        Parameters
        ----------
        ant_locs : array_like
            Antenna coordinates in meters, shape (n_ants, 2). Pass with astropy units.
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
        self.ants = ant_locs.to(units.m).value
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

    def get_bls(self, freq):
        """Unique baseline vectors (m) and wavelength-scaled u, v coordinates."""
        n_ants = self.ants.shape[0]
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

    def _bin_uv_map(self, v_bins, u_bins):
        """Bin baseline u,v onto grid edges; binned_statistic_2d x=v, y=u."""

        ######this non-uniform weighting is never used ############
        weights = np.ones(len(self.u_coords))
        binned_uv = stats.binned_statistic_2d(
            self.v_coords,
            self.u_coords,
            weights,
            statistic='mean',
            bins=[v_bins, u_bins],
        )
        ############################################################

        #this is the uniform weighting that is used throughout the code
        binned_counts = stats.binned_statistic_2d(
            self.v_coords,
            self.u_coords,
            self.counts,
            statistic='sum',
            bins=[v_bins, u_bins],
        )

        #check that ubins = vbins = 0 has binned_uv =0
        #find where u_bins = 0 and v_bins = 0
        u_0_idx = np.where(u_bins == 0)[0]
        v_0_idx = np.where(v_bins == 0)[0]
        if len(u_0_idx) > 0 and len(v_0_idx) > 0:
            if binned_uv.statistic[v_0_idx[0],u_0_idx[0]] != 0:
                print(f'(u,v) = (0,0) is not 0 in the uv_map, setting to 0')
                #set binned_uv.statistic[v_0_idx[0],u_0_idx[0]] = 0
                binned_uv.statistic[v_0_idx[0],u_0_idx[0]] = 0
            else: 
                pass

        self.count_map = _nan_to_zero(binned_counts.statistic)
        # old: return _nan_to_zero(binned_uv.statistic)
        return self.count_map # Leon wants natural weighting counts.

    def _parse_custom_uv(self, N_uv=None, custom_uv=None):
        """Return (u_grid, v_grid, count_map) or None if using antenna binning."""
        if N_uv is not None and custom_uv is not None:
            raise ValueError('Pass only one of N_uv or custom_uv, not both.')

        if N_uv is not None:
            if len(N_uv) != 3:
                raise ValueError(
                    'N_uv must be a 3-tuple (u_grid, v_grid, count_map).',
                )
            u_grid, v_grid, count_map = N_uv
        elif custom_uv is not None:
            if 'count_map' in custom_uv:
                count_map = custom_uv['count_map']
            elif 'N' in custom_uv:
                count_map = custom_uv['N']
            else:
                raise ValueError(
                    "custom_uv must include 'count_map' or 'N'.",
                )
            u_grid = custom_uv['u_grid']
            v_grid = custom_uv['v_grid']
        else:
            return None

        return (
            np.asarray(u_grid, dtype=float),
            np.asarray(v_grid, dtype=float),
            np.asarray(count_map, dtype=float),
        )

    def _apply_custom_uv(self, u_grid, v_grid, count_map):
        """
        Use user-supplied UV bin edges and count map.

        ``count_map`` is stored as-is; ``uv_map`` is the binary coverage mask
        (1 where count > 0, else 0).
        """
        if u_grid.ndim != 1 or v_grid.ndim != 1:
            raise ValueError('u_grid and v_grid must be 1D bin-edge arrays.')
        if len(u_grid) < 2 or len(v_grid) < 2:
            raise ValueError(
                'u_grid and v_grid must each have at least two bin edges.',
            )
        if np.any(np.diff(u_grid) <= 0) or np.any(np.diff(v_grid) <= 0):
            raise ValueError('u_grid and v_grid must be strictly increasing.')

        expected_shape = (len(v_grid) - 1, len(u_grid) - 1)
        if count_map.shape != expected_shape:
            raise ValueError(
                f'count_map shape {count_map.shape} must match '
                f'(len(v_grid)-1, len(u_grid)-1) = {expected_shape}.',
            )

        self.u_grid = u_grid
        self.v_grid = v_grid
        self.count_map = count_map.copy()
        self.du = float(np.median(np.diff(self.u_grid)))
        self.dv = float(np.median(np.diff(self.v_grid)))

        if len(self.u_grid) == 2:
            self.array_layout = 'ns_only'
        elif len(self.v_grid) == 2:
            self.array_layout = 'ew_only'
        else:
            self.array_layout = '2d'
        #TODO Leon claims the uv_map here should be the count_map, not the uv_map.
        # Old uv_map = np.where(self.count_map > 0, 1.0, 0.0)
        uv_map = self.count_map

        u_nat, v_nat, _, _, _, _ = self.sky_uv_coords()
        self._set_halfwave_uv_centers(u_nat, v_nat)

        return uv_map

    def get_uvmap_halfwave(self, freq, N_uv=None, custom_uv=None):
        """
        Grid instantaneous UV coverage on a half-wave lattice (du=dv=0.5).

        Sets ``self.u_grid``, ``self.v_grid``, ``self.count_map``, and
        ``self.array_layout`` (``'2d'``, ``'ew_only'``, or ``'ns_only'``).

        Parameters
        ----------
        freq : Quantity
            Observing frequency (used for antenna-based gridding only).
        N_uv : tuple, optional
            ``(u_grid, v_grid, count_map)`` with bin **edges** in wavelengths
            and per-bin sample counts. Coverage ``uv_map`` is derived as a
            binary mask where ``count_map > 0``.
        custom_uv : dict, optional
            Same as ``N_uv`` with keys ``u_grid``, ``v_grid``, and
            ``count_map`` (or ``N``).
        """
        custom = self._parse_custom_uv(N_uv=N_uv, custom_uv=custom_uv)
        if custom is not None:
            return self._apply_custom_uv(*custom)

        self.get_bls(freq)
        self.du = 0.5
        self.dv = 0.5
        u_max = np.abs(self.u_coords).max() if len(self.u_coords) else 0.0
        v_max = np.abs(self.v_coords).max() if len(self.v_coords) else 0.0
        self.u_grid = np.arange(-u_max, u_max + self.du, self.du)
        self.v_grid = np.arange(-v_max, v_max + self.dv, self.dv)

        if len(self.u_grid) <= 1:
            self.array_layout = 'ns_only'
        elif len(self.v_grid) <= 1:
            self.array_layout = 'ew_only'
        else:
            self.array_layout = '2d'

        if self.array_layout == '2d':
            uv_map = self._bin_uv_map(self.v_grid, self.u_grid)
        elif self.array_layout == 'ns_only':
            u_sky, _, _, _, _, _ = self.sky_uv_coords()
            uv_map = self._bin_uv_map(self.v_grid, u_sky)
        elif self.array_layout == 'ew_only':
            _, v_sky, _, _, _, _ = self.sky_uv_coords()
            uv_map = self._bin_uv_map(v_sky, self.u_grid)
        else:
            raise RuntimeError(f'Unknown array_layout: {self.array_layout}')      

        return uv_map

    def get_psf(self, freq, N_uv=None, custom_uv=None, peak_normalize=False):
        """Point spread function (dirty beam) on the sky pixel grid."""
        #TODO Leon claims the uv_map here should be the count_map, not the uv_map.
        # It should be peak noramlized to 1

        uv_map = self.get_uvmap_halfwave(freq, N_uv=N_uv, custom_uv=custom_uv)
        u_nat, v_nat, _, _, _, _ = self.sky_uv_coords()
        self._set_halfwave_uv_centers(u_nat, v_nat)

        npix_0, npix_1 = uv_map.shape
        psf = (
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uv_map, axes=(0, 1))))
            * (self.du * self.dv * npix_0 * npix_1)
        )
        psf = self.interp_dirty_map_to_sky(psf)
        # # peak normalize the psf to 1
        if peak_normalize == True:
            psf /= np.nanmax(psf)
        else:
            pass
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

    def sky_fft_to_uv(self, sky_map):
        """FFT sky_map to UV; same scaling as get_dirty_map."""
        _, _, _, _, dl, dm = self.sky_uv_coords()

        return (
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky_map, axes=(0, 1))))
            * (dl * dm)
        )

    def _set_halfwave_uv_centers(self, u_nat, v_nat):
        if self.array_layout == '2d':
            self.u_hw = 0.5 * (self.u_grid[:-1] + self.u_grid[1:])
            self.v_hw = 0.5 * (self.v_grid[:-1] + self.v_grid[1:])
        elif self.array_layout == 'ns_only':
            self.u_hw = u_nat[:-1]
            self.v_hw = 0.5 * (self.v_grid[:-1] + self.v_grid[1:])
        elif self.array_layout == 'ew_only':
            self.u_hw = 0.5 * (self.u_grid[:-1] + self.u_grid[1:])
            self.v_hw = v_nat[:-1]

    def interp_sky_fft_to_halfwave(
        self,
        sky_map,
        freq,
        uv_map=None,
        N_uv=None,
        custom_uv=None,
        method='linear',
        fill_value=0.0,
        ):
        """
        Interpolate model visibilities onto the half-wave grid from get_uvmap_halfwave.

        If ``uv_map`` is provided, ``get_uvmap_halfwave`` is not called again (grids
        must already be set on ``self``). Pass ``N_uv`` or ``custom_uv`` to use
        a user-supplied count map and bin edges instead of antenna binning.
        """
        if N_uv is not None or custom_uv is not None:
            uv_map = self.get_uvmap_halfwave(
                freq, N_uv=N_uv, custom_uv=custom_uv,
            )
        elif uv_map is None:
            uv_map = self.get_uvmap_halfwave(freq)
        elif not hasattr(self, 'u_grid'):
            self.get_uvmap_halfwave(freq)

        sky_fft = self.sky_fft_to_uv(sky_map)
        u_nat, v_nat, _, _, _, _ = self.sky_uv_coords()
        self._set_halfwave_uv_centers(u_nat, v_nat)

        interp_re = scipy.interpolate.RegularGridInterpolator(
            (v_nat, u_nat),
            sky_fft.real,
            method=method,
            bounds_error=False,
            fill_value=fill_value,
        )
        interp_im = scipy.interpolate.RegularGridInterpolator(
            (v_nat, u_nat),
            sky_fft.imag,
            method=method,
            bounds_error=False,
            fill_value=fill_value,
        )
        VV, UU = np.meshgrid(self.v_hw, self.u_hw, indexing='ij')
        pts = np.column_stack([VV.ravel(), UU.ravel()])
        V_hw = (interp_re(pts) + 1j * interp_im(pts)).reshape(
            len(self.v_hw), len(self.u_hw),
        )

        return V_hw, uv_map

    def interp_dirty_map_to_sky(self, dirty_map):
        """Interpolate a dirty map from the half-wave (l, m) grid to the sky grid."""
        _, _, l_sky, m_sky, _, _ = self.sky_uv_coords()

        l_hw = np.fft.fftshift(np.fft.fftfreq(len(self.u_hw), d=self.du))
        m_hw = np.fft.fftshift(np.fft.fftfreq(len(self.v_hw), d=self.dv))

        interp = scipy.interpolate.RegularGridInterpolator(
            (m_hw, l_hw),
            dirty_map,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        MM, LL = np.meshgrid(m_sky, l_sky, indexing='ij')
        pts = np.column_stack([MM.ravel(), LL.ravel()])

        return interp(pts).reshape(len(m_sky), len(l_sky))

    def _uv_noise_scale(self, redundancy):
        noise_level = self.compute_noise()
        if redundancy:
            scale = np.zeros_like(self.count_map, dtype=float)
            np.divide(
                noise_level,
                np.sqrt(self.count_map),
                where=self.count_map != 0,
                out=scale,
            )
            return scale
        return noise_level

    def _draw_uv_noise(self, shape, uv_map, redundancy, normalize_sqrt2=True):
        scale = self._uv_noise_scale(redundancy)
        if redundancy:
            a = np.random.normal(0, scale, shape)
            b = np.random.normal(0, scale, shape)
        else:
            a = np.random.normal(0, scale, shape)
            b = np.random.normal(0, scale, shape)

        noise_map = a + 1j * b
        # really need this renormalization because of the way the noise is drawn twice for real and imaginary components. 
        if normalize_sqrt2:
            noise_map /= np.sqrt(2)
        return np.where(uv_map != 0, noise_map, 0)

    def get_dirty_map(
        self,
        sky_map,
        freq,
        noise=False,
        redundancy=True,
        N_uv=None,
        custom_uv=None,
        ):
        """
        Dirty map on the sky grid; optional thermal noise in the UV plane.

        Parameters
        ----------
        N_uv : tuple, optional
            ``(u_grid, v_grid, count_map)`` — user count map per UV bin; bin
            edges in wavelengths. Coverage is ``(count_map > 0)``.
        custom_uv : dict, optional
            ``{'u_grid', 'v_grid', 'count_map'}`` (or ``'N'`` for counts).
        """
        uv_map = self.get_uvmap_halfwave(
            freq, N_uv=N_uv, custom_uv=custom_uv,
        )
        V_hw, _ = self.interp_sky_fft_to_halfwave(sky_map, freq, uv_map=uv_map)
        dirty_uv = uv_map * V_hw
        if noise:
            dirty_uv += self._draw_uv_noise(
                dirty_uv.shape, uv_map, redundancy, normalize_sqrt2=False,
            )

        npix_interp = V_hw.shape[0] * V_hw.shape[1]
        dirty_map = (
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(dirty_uv, axes=(0, 1))))
            * self.du * self.dv * npix_interp
        )

        dirty_map = self.interp_dirty_map_to_sky(dirty_map)
        # peak normalize the dirty map to 1
        psf = self.get_psf(freq, peak_normalize=False)
        dirty_map /= np.nanmax(psf)

        return dirty_map.real

    def _validate_per_channel_uv(self, n_freq, N_uv=None, custom_uv=None):
        if N_uv is not None and custom_uv is not None:
            raise ValueError(
                'Pass only one of N_uv or custom_uv for cube imaging, not both.',
            )
        if N_uv is not None and len(N_uv) != n_freq:
            raise ValueError(
                f'N_uv must have length {n_freq} (one entry per frequency channel).',
            )
        if custom_uv is not None and len(custom_uv) != n_freq:
            raise ValueError(
                f'custom_uv must have length {n_freq} (one entry per channel).',
            )

    def get_dirty_cube(
        self,
        sky_cube,
        freqs,
        noise=False,
        redundancy=True,
        N_uv=None,
        custom_uv=None,
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
        custom_uv : sequence, optional
            Length ``n_freq``; each element is a dict for ``get_dirty_map``.

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

        self._validate_per_channel_uv(n_freq, N_uv=N_uv, custom_uv=custom_uv)

        dirty_cube = np.empty_like(sky_cube)
        for i in range(n_freq):
            n_uv_i = N_uv[i] if N_uv is not None else None
            custom_i = custom_uv[i] if custom_uv is not None else None
            dirty_cube[:, :, i] = self.get_dirty_map(
                sky_cube[:, :, i],
                freqs[i],
                noise=noise,
                redundancy=redundancy,
                N_uv=n_uv_i,
                custom_uv=custom_i,
            )

        return dirty_cube

    def get_noise_map(
        self, freq, redundancy=True, N_uv=None, custom_uv=None,
        ):
        """Noise-only image realization on the sky grid."""
        uv_map = self.get_uvmap_halfwave(
            freq, N_uv=N_uv, custom_uv=custom_uv,
        )
        u_nat, v_nat, _, _, _, _ = self.sky_uv_coords()
        self._set_halfwave_uv_centers(u_nat, v_nat)

        noise_map = self._draw_uv_noise(
            uv_map.shape, uv_map, redundancy, normalize_sqrt2=True,
        )

        npix_0, npix_1 = uv_map.shape
        position_noise_map = (
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(noise_map, axes=(0, 1))))
            * self.du * self.dv * npix_0 * npix_1
        )
        position_noise_map = self.interp_dirty_map_to_sky(position_noise_map)
        return position_noise_map.real

    def compute_noise(self):
        """Per-visibility noise standard deviation (K)."""
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
