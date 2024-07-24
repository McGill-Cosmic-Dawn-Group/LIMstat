import sys
import unittest
import pytest

import numpy as np
from astropy import units

from ska_simulator.simulations import foregrounds


class test_foregrounds(unittest.TestCase):
    def setUp(self):

        nfreqs = 15
        df = 100e3 * units.Hz
        avg_nu = 150.e6 * units.Hz

        # observations
        self.spw_window = np.arange(avg_nu.value - df.value*nfreqs//2,
                                    avg_nu.value + df.value*(nfreqs//2),
                                    step=df.value) * units.Hz
        self.npix = 50
        self.ang_res = 50. * units.arcsec
        self.RA = 0.*units.deg
        self.Dec = -30. * units.deg
        self.beam_area = 1.65e-03 * units.deg**2

        # foregrounds prameters
        self.point_sources = True
        self.diffuse = True
        self.use_pygsm = True

        # point sources parameters
        self.n_src = 110
        self.min_flux = 0.1 * units.Jy
        self.max_flux = 100. * units.Jy

        # diffuse foregrounds parameters
        self.sync_params = [2.8, 0.1, 335.4]
        self.ff_params = [2.15, 0.01, 33.5]
        self.n_upt = 25

        self.fg = foregrounds(
            freqs=self.spw_window,
            npix=self.npix,
            ang_res=self.ang_res,
            beam_area=self.beam_area,
            RA=self.RA,
            Dec=self.Dec,
            point_sources=self.point_sources,
            diffuse=self.diffuse,
            use_pygsm=self.use_pygsm,
            )

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):

        # check freqs units
        # issue warning if freqs fed without unit
        pytest.warns(UserWarning,
                     foregrounds,
                     freqs=self.spw_window.value,
                     npix=self.npix,
                     ang_res=self.ang_res,
                     beam_area=self.beam_area,
                     RA=self.RA,
                     Dec=self.Dec,
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError, 
                      foregrounds,
                      freqs=self.spw_window.value * units.K,
                      npix=self.npix,
                      ang_res=self.ang_res,
                      beam_area=self.beam_area,
                      RA=self.RA,
                      Dec=self.Dec,
                      )
        # convert inplace if compatible unit
        _ = foregrounds(
            freqs=self.spw_window.to(units.MHz),
            npix=self.npix,
            ang_res=self.ang_res,
            beam_area=self.beam_area,
            RA=self.RA,
            Dec=self.Dec,
            point_sources=self.point_sources,
            diffuse=self.diffuse,
            use_pygsm=self.use_pygsm,
            )

        # check ang_res units
        # issue warning if freqs fed without unit
        pytest.warns(UserWarning,
                     foregrounds,
                     freqs=self.spw_window,
                     npix=self.npix,
                     ang_res=self.ang_res.value,
                     beam_area=self.beam_area,
                     RA=self.RA,
                     Dec=self.Dec,
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError, 
                      foregrounds,
                      freqs=self.spw_window,
                      npix=self.npix,
                      ang_res=self.ang_res.value * units.Jy,
                      beam_area=self.beam_area,
                      RA=self.RA,
                      Dec=self.Dec,
                      )
        # convert inplace if compatible unit
        _ = foregrounds(
            freqs=self.spw_window,
            npix=self.npix,
            ang_res=self.ang_res.to(units.deg),
            beam_area=self.beam_area,
            RA=self.RA,
            Dec=self.Dec,
            point_sources=self.point_sources,
            diffuse=self.diffuse,
            use_pygsm=self.use_pygsm,
            )

        # check beam_area units
        # issue warning if fed without unit
        pytest.warns(UserWarning,
                     foregrounds,
                     freqs=self.spw_window,
                     npix=self.npix,
                     ang_res=self.ang_res,
                     beam_area=self.beam_area.value,
                     RA=self.RA,
                     Dec=self.Dec,
                     )
        # raise error if fed in wrong unit
        pytest.raises(ValueError, 
                      foregrounds,
                      freqs=self.spw_window,
                      npix=self.npix,
                      ang_res=self.ang_res,
                      beam_area=self.beam_area.value * units.Jy,
                      RA=self.RA,
                      Dec=self.Dec,
                      )
        # convert inplace if compatible unit
        _ = foregrounds(
            freqs=self.spw_window,
            npix=self.npix,
            ang_res=self.ang_res,
            beam_area=self.beam_area.to(units.deg**2),
            RA=self.RA,
            Dec=self.Dec,
            point_sources=self.point_sources,
            diffuse=self.diffuse,
            use_pygsm=self.use_pygsm,
            )

        # check RA units
        # issue warning if freqs fed without unit
        pytest.warns(UserWarning,
                     foregrounds,
                     freqs=self.spw_window,
                     npix=self.npix,
                     ang_res=self.ang_res,
                     beam_area=self.beam_area,
                     RA=self.RA.value,
                     Dec=self.Dec,
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError, 
                      foregrounds,
                      freqs=self.spw_window,
                      npix=self.npix,
                      ang_res=self.ang_res,
                      beam_area=self.beam_area,
                      RA=self.RA.value * units.K,
                      Dec=self.Dec,
                      )
        # convert inplace if compatible unit
        _ = foregrounds(
            freqs=self.spw_window,
            npix=self.npix,
            ang_res=self.ang_res,
            beam_area=self.beam_area,
            RA=self.RA.to(units.arcsec),
            Dec=self.Dec,
            point_sources=self.point_sources,
            diffuse=self.diffuse,
            use_pygsm=self.use_pygsm,
            )

        # check Dec units
        # issue warning if freqs fed without unit
        pytest.warns(UserWarning,
                     foregrounds,
                     freqs=self.spw_window,
                     npix=self.npix,
                     ang_res=self.ang_res,
                     beam_area=self.beam_area,
                     RA=self.RA,
                     Dec=self.Dec.value,
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError, 
                      foregrounds,
                      freqs=self.spw_window,
                      npix=self.npix,
                      ang_res=self.ang_res,
                      beam_area=self.beam_area,
                      RA=self.RA,
                      Dec=self.Dec.value * units.K,
                      )
        # convert inplace if compatible unit
        _ = foregrounds(
            freqs=self.spw_window,
            npix=self.npix,
            ang_res=self.ang_res,
            beam_area=self.beam_area,
            RA=self.RA,
            Dec=self.Dec.to(units.arcsec),
            point_sources=self.point_sources,
            diffuse=self.diffuse,
            use_pygsm=self.use_pygsm,
            )
        
    def test_generate_point_sources(self):

        # correct usage
        pt_map = self.fg.generate_point_sources(
             min_flux=self.min_flux,
             max_flux=self.max_flux,
        )
        assert pt_map.shape == (self.fg.npix, self.fg.npix, self.fg.nfreqs)
        assert pt_map.unit == units.K

        # check flux units
        # issue warning if fed without unit
        pytest.warns(UserWarning,
                     self.fg.generate_point_sources,
                     min_flux=self.min_flux.value,
                     max_flux=self.max_flux,
                     )
        pytest.warns(UserWarning,
                     self.fg.generate_point_sources,
                     min_flux=self.min_flux,
                     max_flux=self.max_flux.value,
                     )
        # raise error if fed in wrong unit
        pytest.raises(ValueError, 
                      self.fg.generate_point_sources,
                      min_flux=self.min_flux,
                      max_flux=self.max_flux.value * units.Hz,
                      )
        pytest.raises(ValueError, 
                      self.fg.generate_point_sources,
                      min_flux=self.min_flux.value * units.Hz,
                      max_flux=self.max_flux,
                      )
        # convert inplace if compatible unit
        _ = self.fg.generate_point_sources(
             min_flux=self.min_flux.si,
             max_flux=self.max_flux,
        )
        _ = self.fg.generate_point_sources(
             min_flux=self.min_flux,
             max_flux=self.max_flux.si,
        )
        # flux must be positive and max_flux larger than min_flux
        pytest.raises(
            ValueError,
            self.fg.generate_point_sources,
            min_flux=0.*units.Jy,
            max_flux=self.max_flux,
        )
        pytest.raises(
            ValueError,
            self.fg.generate_point_sources,
            min_flux=self.max_flux,
            max_flux=self.min_flux,
        )

    def test_generate_diffuse(self):
        
        # normal usage
        diff_map = self.fg.generate_diffuse()
        assert diff_map.shape == (self.fg.npix, self.fg.npix, self.fg.nfreqs)
        assert diff_map.value.any()
        assert diff_map.unit == units.K
        zero_map = self.fg.generate_diffuse(
            synchrotron=False,
            free_free=False,
            unresolved_pt=False
            )
        assert np.allclose(zero_map, 0.)

    def test_generate_synchrotron(self):

        # correct usage with pygsm
        sync_map = self.fg.generate_synchrotron() 
        assert sync_map.shape == (self.fg.npix, self.fg.npix, self.fg.nfreqs)
        assert sync_map.unit == units.K

        # without pygsm
        fg_nogsm = foregrounds(
            freqs=self.spw_window,
            npix=self.npix,
            ang_res=self.ang_res,
            beam_area=self.beam_area,
            RA=self.RA,
            Dec=self.Dec,
            point_sources=self.point_sources,
            diffuse=self.diffuse,
            use_pygsm=False,
            )
        _ = fg_nogsm.generate_synchrotron(sync_params=self.sync_params)
        # wrong sync_params inputs
        pytest.raises(
            ValueError,
            fg_nogsm.generate_synchrotron,
            sync_params=[2.8, 0.1],
        )
        pytest.warns(
            UserWarning,
            fg_nogsm.generate_synchrotron,
            sync_params=[2.8, 0.1, 10000.],
        )

    def test_generate_free_free(self):

        # correct usage
        free_map = self.fg.generate_free_free()
        assert free_map.shape == (self.fg.npix, self.fg.npix, self.fg.nfreqs)
        assert free_map.unit == units.K

        # wrong input for ff_params
        pytest.raises(
            ValueError,
            self.fg.generate_free_free,
            ff_params=[2.8, 0.1],
        )
        pytest.warns(
            UserWarning,
            self.fg.generate_free_free,
            ff_params=[2.8, 0.1, 1000.],
        )

    def test_generate_unresolved(self):

        # correct usage
        upt_map = self.fg.generate_unresolved_point_sources()
        assert upt_map.shape == (self.fg.npix, self.fg.npix, self.fg.nfreqs)
        assert upt_map.unit == units.K

        upt_map = self.fg.generate_unresolved_point_sources(avg_n_src=12)

        # check min_flux units
        # issue warning if fed without unit
        pytest.warns(UserWarning,
                     self.fg.generate_unresolved_point_sources,
                     min_flux=1.
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError, 
                      self.fg.generate_unresolved_point_sources,
                      min_flux=1.*units.K
                      )
        # convert inplace if compatible unit
        _ = self.fg.generate_unresolved_point_sources(
            min_flux=1.*units.Jy
            )

        # check max_flux units
        # issue warning if fed without unit
        pytest.warns(UserWarning,
                     self.fg.generate_unresolved_point_sources,
                     max_flux=1.
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError, 
                      self.fg.generate_unresolved_point_sources,
                      max_flux=1.*units.K
                      )
        # convert inplace if compatible unit
        _ = self.fg.generate_unresolved_point_sources(
            max_flux=1.*units.Jy
            )

        # check ref_freq units
        # issue warning if fed without unit
        pytest.warns(UserWarning,
                     self.fg.generate_unresolved_point_sources,
                     ref_freq=150.
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError, 
                      self.fg.generate_unresolved_point_sources,
                      ref_freq=150. * units.K
                      )
        # convert inplace if compatible unit
        _ = self.fg.generate_unresolved_point_sources(
            ref_freq=(150.*units.Hz).si
            )

        # TODO: add tests relative to numba
        
    def test_generate_model(self):

        # correct usage
        model = self.fg.generate_model()
        assert model.shape == (self.fg.npix, self.fg.npix, self.fg.nfreqs)
        assert model.unit == units.K

        # no diffuse
        fg1 = foregrounds(
            freqs=self.spw_window,
            npix=self.npix,
            ang_res=self.ang_res,
            beam_area=self.beam_area,
            RA=self.RA,
            Dec=self.Dec,
            point_sources=self.point_sources,
            diffuse=False,
            use_pygsm=self.use_pygsm,
            )
        model1 = fg1.generate_model()
        assert model1.shape == (fg1.npix, fg1.npix, fg1.nfreqs)
        assert model1.unit == units.K

        # no point sources
        fg2 = foregrounds(
            freqs=self.spw_window,
            npix=self.npix,
            ang_res=self.ang_res,
            beam_area=self.beam_area,
            RA=self.RA,
            Dec=self.Dec,
            point_sources=False,
            diffuse=self.diffuse,
            use_pygsm=self.use_pygsm,
            )
        model2 = fg2.generate_model()
        assert model2.shape == (fg2.npix, fg2.npix, fg2.nfreqs)
        assert model2.unit == units.K
