import numpy as np
import pytest
from astropy import units

from limstat.fast_interferometer import fast_interferometer


def _make_instrument(ant_locs, npix=32, fov_deg=10.0):
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


def _synthetic_custom_uv(nu=9, nv=9, du=0.5, dv=0.5):
    u_grid = np.arange(nu) * du - (nu - 1) * du / 2
    v_grid = np.arange(nv) * dv - (nv - 1) * dv / 2
    count_map = np.zeros((nv - 1, nu - 1))
    mid_u, mid_v = (nu - 1) // 2, (nv - 1) // 2
    count_map[mid_v, :] = 3.0
    count_map[:, mid_u] = 2.0
    return u_grid, v_grid, count_map


def test_custom_uv_dict_dirty_map(freq):
    npix = 32
    inst = _make_instrument(_grid_ants(n_ants=6), npix=npix)
    u_grid, v_grid, count_map = _synthetic_custom_uv()

    custom_uv = {
        'u_grid': u_grid,
        'v_grid': v_grid,
        'count_map': count_map,
    }
    sky = _delta_sky(npix)
    dirty = inst.get_dirty_map(sky, freq, custom_uv=custom_uv)

    assert dirty.shape == (npix, npix)
    assert np.isfinite(dirty).all()
    assert np.allclose(inst.count_map, count_map)

    uv_map = inst.get_uvmap_halfwave(freq, custom_uv=custom_uv)
    expected_uv = np.where(count_map > 0, 1.0, 0.0)
    assert np.allclose(uv_map, expected_uv)
    assert uv_map.sum() == (count_map > 0).sum()


def test_custom_uv_n_uv_tuple_dirty_map(freq):
    npix = 32
    inst = _make_instrument(_grid_ants(n_ants=6), npix=npix)
    u_grid, v_grid, count_map = _synthetic_custom_uv()
    N_uv = (u_grid, v_grid, count_map)

    sky = _delta_sky(npix)
    dirty = inst.get_dirty_map(sky, freq, N_uv=N_uv)

    assert dirty.shape == (npix, npix)
    assert np.allclose(inst.count_map, count_map)

    uv_map = inst.get_uvmap_halfwave(freq, N_uv=N_uv)
    assert np.allclose(uv_map, np.where(count_map > 0, 1.0, 0.0))


def test_custom_uv_both_args_raises(freq):
    npix = 32
    inst = _make_instrument(_grid_ants(n_ants=6), npix=npix)
    u_grid, v_grid, count_map = _synthetic_custom_uv()
    with pytest.raises(ValueError, match='only one'):
        inst.get_uvmap_halfwave(
            freq,
            N_uv=(u_grid, v_grid, count_map),
            custom_uv={'u_grid': u_grid, 'v_grid': v_grid, 'N': count_map},
        )


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


def test_get_dirty_cube_custom_uv_list():
    npix = 32
    n_freq = 2
    inst = _make_instrument(_grid_ants(n_ants=6), npix=npix)
    freqs = np.array([140.0, 160.0]) * units.MHz

    u0, v0, n0 = _synthetic_custom_uv()
    u1, v1, n1 = _synthetic_custom_uv(nu=11, nv=11)
    N_uv_list = [(u0, v0, n0), (u1, v1, n1)]

    sky_cube = np.stack([_delta_sky(npix), _delta_sky(npix)], axis=2)
    inst.get_dirty_cube(sky_cube, freqs, N_uv=N_uv_list)

    inst.get_uvmap_halfwave(freqs[1], N_uv=N_uv_list[1])
    assert np.allclose(inst.count_map, n1)


def test_get_dirty_cube_N_uv_and_custom_uv_raises():
    npix = 32
    inst = _make_instrument(_grid_ants(n_ants=6), npix=npix)
    freqs = np.array([150.0, 155.0]) * units.MHz
    u, v, n = _synthetic_custom_uv()
    sky_cube = np.stack([_delta_sky(npix), _delta_sky(npix)], axis=2)
    with pytest.raises(ValueError, match='only one'):
        inst.get_dirty_cube(
            sky_cube,
            freqs,
            N_uv=[(u, v, n), (u, v, n)],
            custom_uv=[{'u_grid': u, 'v_grid': v, 'N': n}] * 2,
        )
