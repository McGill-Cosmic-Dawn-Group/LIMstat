import numpy as np
import pytest
from astropy import units

from limstat.fast_interferometer import fast_interferometer


def _make_instrument(ant_locs=None, npix=32, fov_deg=10.0):
    theta = fov_deg * units.deg
    return fast_interferometer(
        ant_locs=ant_locs,
        theta_x=theta,
        theta_y=theta,
        x_npix=npix,
        y_npix=npix,
        T_sys=200 * units.K,
        t_obs=100 * units.hr,
        bandwidth=8 * units.MHz,
    )


def _grid_ants(n_ants=8, aperture_m=30.0, seed=0):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_ants)))
    xs = np.linspace(-aperture_m / 2, aperture_m / 2, side)
    ys = np.linspace(-aperture_m / 2, aperture_m / 2, side)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])[:n_ants]
    coords += rng.normal(0, 0.2, coords.shape)
    return coords * units.m


def _line_ants_along_x(n=16, aperture_m=100.0):
    x = np.linspace(-aperture_m / 2, aperture_m / 2, n)
    y = np.zeros(n)
    return np.column_stack([x, y]) * units.m


def _line_ants_along_y(n=16, aperture_m=100.0):
    x = np.zeros(n)
    y = np.linspace(-aperture_m / 2, aperture_m / 2, n)
    return np.column_stack([x, y]) * units.m


def _delta_sky(npix, cy=None, cx=None, amp=1.0):
    sky = np.zeros((npix, npix))
    cy = npix // 2 if cy is None else cy
    cx = npix // 2 if cx is None else cx
    sky[cy, cx] = amp
    return sky


@pytest.fixture
def freq():
    return 150 * units.MHz


def test_2d_layout_shapes(freq):
    npix = 32
    inst = _make_instrument(_grid_ants(), npix=npix)
    uv_map = inst.get_uvmap_halfwave(freq)

    assert inst.array_layout == '2d'
    assert uv_map.ndim == 2
    assert inst.count_map.shape == uv_map.shape

    sky = _delta_sky(npix)
    dirty = inst.get_dirty_map(sky, freq, noise=False)
    assert dirty.shape == (npix, npix)
    assert np.isfinite(dirty).all()


def test_ew_only_layout_and_fringes(freq):
    npix = 32
    inst = _make_instrument(_line_ants_along_x(), npix=npix)
    inst.get_uvmap_halfwave(freq)

    assert inst.array_layout == 'ew_only'

    dirty = inst.get_dirty_map(_delta_sky(npix), freq, noise=False)
    assert dirty.shape == (npix, npix)
    assert dirty.max() > 0

    # EW baselines: fringe pattern varies primarily along x (l)
    grad_x = np.abs(np.diff(dirty, axis=1)).sum()
    grad_y = np.abs(np.diff(dirty, axis=0)).sum()
    assert grad_x > grad_y


def test_ns_only_layout_and_fringes(freq):
    npix = 32
    inst = _make_instrument(_line_ants_along_y(), npix=npix)
    inst.get_uvmap_halfwave(freq)

    assert inst.array_layout == 'ns_only'

    dirty = inst.get_dirty_map(_delta_sky(npix), freq, noise=False)
    assert dirty.shape == (npix, npix)
    assert dirty.max() > 0

    grad_x = np.abs(np.diff(dirty, axis=1)).sum()
    grad_y = np.abs(np.diff(dirty, axis=0)).sum()
    assert grad_y > grad_x


def test_noise_map_runs(freq):
    npix = 32
    inst = _make_instrument(_grid_ants(), npix=npix)
    noise_img = inst.get_noise_map(freq, redundancy=True)
    assert noise_img.shape == (npix, npix)
    assert np.isfinite(noise_img).all()


def _synthetic_n_uv(nu=9, nv=9, du=0.5, dv=0.5):
    u_grid = np.arange(nu) * du - (nu - 1) * du / 2
    v_grid = np.arange(nv) * dv - (nv - 1) * dv / 2
    count_map = np.zeros((nv - 1, nu - 1))
    mid_u, mid_v = (nu - 1) // 2, (nv - 1) // 2
    count_map[mid_v, :] = 3.0
    count_map[:, mid_u] = 2.0
    return u_grid, v_grid, count_map


def test_n_uv_padded_grid_matches_input_spacing(freq):
    npix = 32
    inst = _make_instrument(npix=npix)
    u_grid, v_grid, count_map = _synthetic_n_uv()
    N_uv = (u_grid, v_grid, count_map)

    dirty = inst.get_dirty_map(_delta_sky(npix), freq, N_uv=N_uv)

    assert dirty.shape == (npix, npix)
    assert np.isfinite(dirty).all()
    assert inst.uv_mode == 'nuv'
    assert inst.du == pytest.approx(0.5)
    assert inst.dv == pytest.approx(0.5)
    assert inst.count_map.shape == (inst.n_pad_y, inst.n_pad_x)
    assert inst.count_map.sum() == pytest.approx(count_map.sum())

    uv_map = inst.get_uvmap_halfwave(freq, N_uv=N_uv)
    assert uv_map.shape == inst.count_map.shape
    assert uv_map.sum() == pytest.approx(count_map.sum())


def test_n_uv_does_not_use_antenna_grid(freq):
    npix = 32
    inst = _make_instrument(_grid_ants(n_ants=6), npix=npix)
    u_grid, v_grid, count_map = _synthetic_n_uv(du=1.0, dv=1.25)
    N_uv = (u_grid, v_grid, count_map)

    inst.get_dirty_map(_delta_sky(npix), freq, N_uv=N_uv)

    assert inst.uv_mode == 'nuv'
    assert not hasattr(inst, 'u_coords')
    assert inst.du == pytest.approx(1.0)
    assert inst.dv == pytest.approx(1.25)
    assert inst.count_map.sum() == pytest.approx(count_map.sum())
    dl = np.sin((10.0 * units.deg).to(units.rad).value) / npix
    assert inst.n_pad_x == int(round(1.0 / (1.0 * dl)))
    assert inst.n_pad_y == int(round(1.0 / (1.25 * dl)))


def test_n_uv_non_halfwave_spacing(freq):
    npix = 32
    inst = _make_instrument(npix=npix)
    u_grid, v_grid, count_map = _synthetic_n_uv(nu=11, nv=11, du=1.0, dv=1.25)
    N_uv = (u_grid, v_grid, count_map)

    dirty = inst.get_dirty_map(_delta_sky(npix), freq, N_uv=N_uv)
    psf = inst.get_psf(freq, N_uv=N_uv)

    assert dirty.shape == (npix, npix)
    assert psf.shape == (npix, npix)
    assert np.isfinite(dirty).all()
    assert np.isfinite(psf).all()
    assert inst.du == pytest.approx(1.0)
    assert inst.dv == pytest.approx(1.25)
    assert inst.count_map.shape == (inst.n_pad_y, inst.n_pad_x)


def test_n_uv_centres_warn_and_match_edge_spacing(freq):
    npix = 32
    inst = _make_instrument(npix=npix)
    u_edges, v_edges, count_map = _synthetic_n_uv(du=1.0, dv=1.25)
    u_cent = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_cent = 0.5 * (v_edges[:-1] + v_edges[1:])

    with pytest.warns(UserWarning, match='bin centres'):
        dirty = inst.get_dirty_map(
            _delta_sky(npix), freq, N_uv=(u_cent, v_cent, count_map),
        )

    assert dirty.shape == (npix, npix)
    assert np.isfinite(dirty).all()
    assert inst.du == pytest.approx(1.0)
    assert inst.dv == pytest.approx(1.25)
    assert inst.count_map.sum() == pytest.approx(count_map.sum())


def test_coarse_n_uv_is_padded_to_the_sky(freq):
    npix = 64
    inst = _make_instrument(npix=npix)
    du = 8.0
    edges = (np.arange(9) - 4) * du
    counts = np.ones((8, 8))
    dirty = inst.get_dirty_map(np.ones((npix, npix)), freq, N_uv=(edges, edges, counts))
    dl = np.sin((10.0 * units.deg).to(units.rad).value) / npix
    assert inst.n_pad_x == npix
    assert inst.n_pad_y == npix
    assert inst.du == pytest.approx(1.0 / (npix * dl))
    u_cent = 0.5 * (inst.u_grid[:-1] + inst.u_grid[1:])
    assert u_cent.max() <= 0.5 / dl
    assert dirty.shape == (npix, npix)
    assert np.isfinite(dirty).all()
    # A hard zero frame means the sky was cropped into a smaller image.
    assert np.max(np.abs(dirty[:8, npix // 2])) > 0
    psf = inst.get_psf(freq, N_uv=(edges, edges, counts))
    assert psf.shape == (npix, npix)
    assert np.isfinite(psf).all()


def test_uv_complete_returns_the_sky(freq):
    npix = 32
    inst = _make_instrument(npix=npix)
    yy, xx = np.mgrid[0:npix, 0:npix]
    cy = cx = npix // 2
    sky = np.exp(-((xx - cx + 2) ** 2 + (yy - cy - 1) ** 2) / (2 * 2.0 ** 2))
    u, v, _, _, _, _ = inst.sky_uv_coords()
    with pytest.warns(UserWarning, match='bin centres'):
        dirty = inst.get_dirty_map(
            sky, freq, N_uv=(u, v, np.ones_like(sky)), noise=False,
        )
    assert inst.count_map.shape == sky.shape
    assert np.allclose(inst.count_map, 1.0)
    assert np.max(np.abs(dirty - sky)) < 1e-8


def test_n_uv_shape_mismatch_raises(freq):
    inst = _make_instrument(npix=32)
    u_grid = np.linspace(-2, 2, 6)
    v_grid = np.linspace(-2, 2, 5)
    count_map = np.ones((3, 3))
    with pytest.raises(ValueError, match='bin centres'):
        inst.get_uvmap_halfwave(freq, N_uv=(u_grid, v_grid, count_map))


def test_n_uv_beyond_nyquist_is_clipped_to_the_sky(freq):
    npix = 32
    inst = _make_instrument(npix=npix)
    du = dv = 1.0
    u_grid = np.arange(-100, 101, du)
    v_grid = np.arange(-4, 5, dv)
    count_map = np.ones((len(v_grid) - 1, len(u_grid) - 1))
    with pytest.warns(UserWarning, match='sky UV extent'):
        dirty = inst.get_dirty_map(
            _delta_sky(npix), freq, N_uv=(u_grid, v_grid, count_map),
        )
    dl = np.sin((10.0 * units.deg).to(units.rad).value) / npix
    assert dirty.shape == (npix, npix)
    assert np.isfinite(dirty).all()
    assert inst.du == pytest.approx(du)
    assert inst.n_pad_x == int(round(1.0 / (du * dl)))
    assert inst.count_map.sum() < count_map.sum()
    u_cent = 0.5 * (inst.u_grid[:-1] + inst.u_grid[1:])
    assert u_cent.max() <= 0.5 / dl
    assert inst.u_grid[-1] == pytest.approx(0.5 / dl, abs=du)


def test_ant_locs_on_method_without_init_ants(freq):
    npix = 32
    inst = _make_instrument(npix=npix)
    sky = _delta_sky(npix)
    dirty = inst.get_dirty_map(sky, freq, ant_locs=_grid_ants(), noise=False)
    assert dirty.shape == (npix, npix)
    assert np.isfinite(dirty).all()
    assert inst.uv_mode == 'antenna_halfwave'


def test_antenna_path_requires_ant_locs(freq):
    inst = _make_instrument(npix=32)
    with pytest.raises(ValueError, match='ant_locs are required'):
        inst.get_dirty_map(_delta_sky(32), freq)


def test_get_dirty_cube():
    npix = 32
    n_freq = 4
    inst = _make_instrument(_grid_ants(), npix=npix)
    freqs = np.linspace(140, 160, n_freq) * units.MHz

    sky_cube = np.zeros((npix, npix, n_freq))
    for i in range(n_freq):
        sky_cube[:, :, i] = _delta_sky(npix, amp=1.0 + 0.1 * i)

    dirty_cube = inst.get_dirty_cube(sky_cube, freqs, noise=False)

    assert dirty_cube.shape == (npix, npix, n_freq)
    assert np.isfinite(dirty_cube).all()
    assert np.abs(dirty_cube[:, :, 0] - dirty_cube[:, :, 1]).max() > 0


def test_get_dirty_cube_n_uv_list():
    npix = 32
    inst = _make_instrument(npix=npix)
    freqs = np.array([140.0, 160.0]) * units.MHz

    u0, v0, n0 = _synthetic_n_uv()
    u1, v1, n1 = _synthetic_n_uv(nu=11, nv=11)
    N_uv_list = [(u0, v0, n0), (u1, v1, n1)]

    sky_cube = np.stack([_delta_sky(npix), _delta_sky(npix)], axis=2)
    dirty_cube = inst.get_dirty_cube(sky_cube, freqs, N_uv=N_uv_list)

    assert dirty_cube.shape == (npix, npix, 2)
    assert np.isfinite(dirty_cube).all()
    assert inst.count_map.sum() == pytest.approx(n1.sum())


def test_psf_matches_direct_sum(freq):
    npix = 32
    rng = np.random.default_rng(1)
    ants = rng.uniform(-10.0, 10.0, size=(40, 2))
    inst = _make_instrument(ants * units.m, npix=npix)
    psf = inst.get_psf(freq)
    psf = psf / psf.max()

    lam = 3e8 / freq.to(units.Hz).value
    dl = np.sin((10.0 * units.deg).to(units.rad).value) / npix
    coord = (np.arange(npix) - npix // 2) * dl
    ex = np.exp(2j * np.pi * np.outer(ants[:, 0], coord) / lam)
    ey = np.exp(2j * np.pi * np.outer(ants[:, 1], coord) / lam)
    beam = ey.T @ ex
    n_ant = len(ants)
    exact = (np.abs(beam) ** 2 - n_ant) / (n_ant * (n_ant - 1))
    exact = exact / exact.max()
    assert np.max(np.abs(psf - exact)) < 0.01


def test_point_source_amplitude_is_shift_invariant(freq):
    npix = 32
    inst = _make_instrument(_grid_ants(), npix=npix)
    centre = inst.get_dirty_map(_delta_sky(npix), freq, noise=False)
    cy = cx = npix // 2
    assert np.unravel_index(centre.argmax(), centre.shape) == (cy, cx)
    for off in (4, 8, 14):
        dirty = inst.get_dirty_map(
            _delta_sky(npix, cy=cy, cx=cx + off), freq, noise=False,
        )
        assert np.unravel_index(dirty.argmax(), dirty.shape) == (cy, cx + off)
        assert dirty.max() == pytest.approx(centre.max(), rel=1e-5, abs=1e-8)


def test_halfwave_grid_is_centred_and_symmetric(freq):
    inst = _make_instrument(_grid_ants(), npix=32)
    uv_map = inst.get_uvmap_halfwave(freq)
    u_cent = 0.5 * (inst.u_grid[:-1] + inst.u_grid[1:])
    v_cent = 0.5 * (inst.v_grid[:-1] + inst.v_grid[1:])
    assert inst.du == 0.5
    assert inst.dv == 0.5
    assert u_cent[inst.n_pad_x // 2] == pytest.approx(0.0)
    assert v_cent[inst.n_pad_y // 2] == pytest.approx(0.0)
    assert np.allclose(uv_map, uv_map[::-1, ::-1])


def test_ew_psf_constant_along_y(freq):
    npix = 32
    inst = _make_instrument(_line_ants_along_x(), npix=npix)
    psf = inst.get_psf(freq)
    assert inst.array_layout == 'ew_only'
    assert np.allclose(psf, np.broadcast_to(psf[:1, :], psf.shape))


def test_ns_psf_constant_along_x(freq):
    npix = 32
    inst = _make_instrument(_line_ants_along_y(), npix=npix)
    psf = inst.get_psf(freq)
    assert inst.array_layout == 'ns_only'
    assert np.allclose(psf, np.broadcast_to(psf[:, :1], psf.shape))


def test_odd_and_nonsquare_sky(freq):
    y_npix, x_npix = 17, 31
    inst = fast_interferometer(
        ant_locs=_grid_ants(),
        theta_x=8 * units.deg,
        theta_y=5 * units.deg,
        x_npix=x_npix,
        y_npix=y_npix,
        T_sys=200 * units.K,
        t_obs=10 * units.hr,
        bandwidth=1 * units.MHz,
    )
    sky = np.zeros((y_npix, x_npix))
    sky[y_npix // 2, x_npix // 2] = 1.0
    dirty = inst.get_dirty_map(sky, freq, noise=False)
    psf = inst.get_psf(freq)
    assert dirty.shape == (y_npix, x_npix)
    assert psf.shape == (y_npix, x_npix)
    assert np.isfinite(dirty).all()
    assert np.isfinite(psf).all()


def _coarse_n_uv(du=8.0):
    edges = (np.arange(9) - 4) * du
    return edges, edges, np.ones((8, 8))


def _coverage_edge_ratio(raw, tapered):
    mask = np.abs(raw) > 0
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    y0 = int(rows[0])
    span = raw[y0, cols[0]:cols[-1] + 1]
    x = int(cols[0]) + int(np.argmax(np.abs(span)))
    return tapered[y0, x] / raw[y0, x]


def test_taper_none_matches_untapered_psf(freq):
    npix = 64
    inst = _make_instrument(npix=npix)
    n_uv = _coarse_n_uv()
    plain = inst.get_psf(freq, N_uv=n_uv)
    explicit = inst.get_psf(
        freq, N_uv=n_uv, taper=None, taper_kwargs={'scale': 2},
    )
    assert np.allclose(plain, explicit)


def test_blackmanharris_lowers_sidelobes_and_keeps_the_peak(freq):
    npix = 64
    inst = _make_instrument(npix=npix)
    n_uv = _coarse_n_uv()
    psf = inst.get_psf(freq, N_uv=n_uv)
    psf_t = inst.get_psf(freq, N_uv=n_uv, taper='blackmanharris')
    cy = cx = npix // 2
    # Blackman-Harris puts its first null about four times farther out
    # than the untapered sinc, so compare beyond that null.
    untapered = psf[cy, cx:] / np.nanmax(psf)
    first_null = int(np.flatnonzero(untapered < 0)[0])
    radius = min(4 * first_null, npix // 2 - 1)
    yy, xx = np.ogrid[:npix, :npix]
    outside = (yy - cy) ** 2 + (xx - cx) ** 2 > radius ** 2
    side = np.max(np.abs(psf[outside] / np.nanmax(psf)))
    side_t = np.max(np.abs(psf_t[outside] / np.nanmax(psf_t)))
    assert side_t < side

    dirty = inst.get_dirty_map(
        _delta_sky(npix), freq, N_uv=n_uv, taper='blackmanharris',
    )
    assert np.unravel_index(dirty.argmax(), dirty.shape) == (cy, cx)
    assert dirty[cy, cx] == pytest.approx(1.0, abs=1e-6)
    tapered_counts = inst.count_map.copy()
    raw = inst.get_uvmap_halfwave(freq, N_uv=n_uv)
    assert np.allclose(tapered_counts, raw)


def test_taper_scale_keeps_more_weight_on_the_edge(freq):
    npix = 64
    inst = _make_instrument(npix=npix)
    raw = inst.get_uvmap_halfwave(freq, N_uv=_coarse_n_uv())
    narrow = inst._apply_taper(raw, 'blackmanharris', {'scale': 1.0})
    broad = inst._apply_taper(raw, 'blackmanharris', {'scale': 1.5})
    assert _coverage_edge_ratio(raw, broad) > _coverage_edge_ratio(raw, narrow)

    steep = inst._apply_taper(raw, 'kaiser', {'beta': 14})
    flat = inst._apply_taper(raw, 'kaiser', {'beta': 0})
    assert _coverage_edge_ratio(raw, steep) < _coverage_edge_ratio(raw, flat)


def test_callable_taper_scales_the_psf(freq):
    npix = 64
    inst = _make_instrument(npix=npix)
    n_uv = _coarse_n_uv()
    psf = inst.get_psf(freq, N_uv=n_uv)

    def half(uv):
        return np.full(np.shape(uv), 0.5)

    tapered = inst.get_psf(freq, N_uv=n_uv, taper=half)
    assert np.allclose(tapered, 0.5 * psf)


def test_taper_arguments_raise(freq):
    npix = 64
    inst = _make_instrument(npix=npix)
    n_uv = _coarse_n_uv()
    with pytest.raises(ValueError, match='Unknown taper'):
        inst.get_psf(freq, N_uv=n_uv, taper='no-such-window')
    with pytest.raises(ValueError, match='positive'):
        inst.get_psf(freq, N_uv=n_uv, taper='hann', taper_kwargs={'scale': 0})
    with pytest.raises(ValueError, match='positive'):
        inst.get_psf(freq, N_uv=n_uv, taper='hann', taper_kwargs={'scale': -1})
    with pytest.raises(ValueError, match='only scale'):
        inst.get_psf(
            freq, N_uv=n_uv,
            taper='blackmanharris', taper_kwargs={'beta': 4},
        )
    with pytest.raises(ValueError, match='beta'):
        inst.get_psf(freq, N_uv=n_uv, taper='kaiser')
    with pytest.raises(ValueError, match='not used'):
        inst.get_psf(
            freq, N_uv=n_uv,
            taper=lambda uv: np.ones(np.shape(uv)),
            taper_kwargs={'scale': 1},
        )
    with pytest.raises(ValueError, match='shape'):
        inst.get_psf(freq, N_uv=n_uv, taper=lambda uv: np.ones((2, 2)))


def test_taper_threads_through_cube_and_noise(freq):
    npix = 64
    inst = _make_instrument(npix=npix)
    n_uv = _coarse_n_uv()
    sky = _delta_sky(npix)
    dirty = inst.get_dirty_map(
        sky, freq, N_uv=n_uv, taper='blackmanharris',
    )
    cube = np.stack([sky, sky], axis=2)
    dirty_cube = inst.get_dirty_cube(
        cube,
        np.array([150.0, 150.0]) * units.MHz,
        N_uv=[n_uv, n_uv],
        taper='blackmanharris',
    )
    assert np.allclose(dirty_cube[:, :, 0], dirty)

    np.random.seed(0)
    plain = inst.get_noise_map(freq, N_uv=n_uv, redundancy=False)
    np.random.seed(0)
    tapered = inst.get_noise_map(
        freq, N_uv=n_uv, redundancy=False, taper='blackmanharris',
    )
    assert tapered.shape == (npix, npix)
    assert np.isfinite(tapered).all()
    assert not np.allclose(plain, tapered)


def test_taper_parameters_are_documented():
    for name in ('get_psf', 'get_dirty_map', 'get_noise_map', 'get_dirty_cube'):
        doc = getattr(fast_interferometer, name).__doc__
        assert '__TAPER_PARAMETERS__' not in doc
        for phrase in (
            'scale', 'blackmanharris', 'beta', 'alpha', 'std', 'few cells',
        ):
            assert phrase in doc


def test_baselines_beyond_nyquist_are_dropped(freq):
    ants = np.array([[0.0, 0.0], [10.0, 0.0], [400.0, 0.0]]) * units.m
    inst = _make_instrument(ants, npix=32)
    with pytest.warns(UserWarning, match='sky Nyquist'):
        uv_map = inst.get_uvmap_halfwave(freq)
    # Only the +/- 10 m baseline lands inside the sky Nyquist limit.
    assert uv_map.sum() == 2


