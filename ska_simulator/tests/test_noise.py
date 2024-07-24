import sys
import unittest
import pytest

import numpy as np
from astropy import units

from ska_simulator.simulations import thermal_noise

class test_noise(unittest.TestCase):
    def setUp(self):

        self.npix = 50

        nfreqs = 15
        df = 100e3 * units.Hz
        avg_nu = 150.e6 * units.Hz

        self.freqs = np.arange(avg_nu.value - df.value*nfreqs//2,
                                    avg_nu.value + df.value*(nfreqs//2),
                                    step=df.value) * units.Hz
        
        self.nfreqs = self.freqs.size
        self.nb_antennas = 100
        self.total_obs_time = 1000. * units.hour
        self.integration_time = 10. * units.s
        self.daily_obs_time = 4. * units.hour
        self.collecting_area = 100. * units.m**2
        self.Tsys = 15 * units.K
        self.output_unit = units.K
        self.beam_area = 0.00165 * units.deg**2

        self.noise = thermal_noise(
            freqs=self.freqs,
            npix=self.npix,
            nb_antennas=self.nb_antennas,
            total_obs_time=self.total_obs_time,
            integration_time=self.integration_time,
            collecting_area=self.collecting_area,
            Tsys=self.Tsys,
            output_unit=self.output_unit,
            beam_area=self.beam_area,
        )
        assert self.noise.noise_unit.is_equivalent(self.output_unit)
        assert self.noise.std[0].unit.is_equivalent(self.output_unit)
        assert self.noise.std_per_vis[0].unit.is_equivalent(self.output_unit)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):

        # check freqs units
        # issue warning if freqs fed without unit
        pytest.warns(UserWarning,
                     thermal_noise,
                     freqs=self.freqs.value,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError,
                     thermal_noise,
                     freqs=self.freqs.value * units.K,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # convert inplace if compatible unit
        _ = thermal_noise(freqs=self.freqs.to(units.MHz),
                          npix=self.npix,
                          nb_antennas=self.nb_antennas,
                          total_obs_time=self.total_obs_time,
                          integration_time=self.integration_time,
                          collecting_area=self.collecting_area,
                          Tsys=self.Tsys,
                          output_unit=self.output_unit,
                          beam_area=self.beam_area,
                          )
        
        # check total_obs_time units
        # issue warning if integration_time fed without units
        pytest.warns(UserWarning,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time.value,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # raise error if integration_time fed in wrong unit
        pytest.raises(ValueError,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time.value * units.K,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # convert inplace if compatible unit
        _ = thermal_noise(freqs=self.freqs,
                          npix=self.npix,
                          nb_antennas=self.nb_antennas,
                          total_obs_time=self.total_obs_time.to(units.s),
                          integration_time=self.integration_time,
                          collecting_area=self.collecting_area,
                          Tsys=self.Tsys,
                          output_unit=self.output_unit,
                          beam_area=self.beam_area,
                          )

        # check integration_time units
        # issue warning if integration_time fed without units
        pytest.warns(UserWarning,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time.value,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # raise error if integration_time fed in wrong unit
        pytest.raises(ValueError,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time.value * units.K,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # convert inplace if compatible unit
        _ = thermal_noise(freqs=self.freqs,
                          npix=self.npix,
                          nb_antennas=self.nb_antennas,
                          total_obs_time=self.total_obs_time,
                          integration_time=self.integration_time.to(units.hr),
                          collecting_area=self.collecting_area,
                          Tsys=self.Tsys,
                          output_unit=self.output_unit,
                          beam_area=self.beam_area,
                          )
        
        # check collecting_area units 
        # issue warning if collecting_area fed without units
        pytest.warns(UserWarning,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area.value,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # raise error if collecting_area fed in wrong unit
        pytest.raises(ValueError,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area.value * units.K,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # convert inplace if compatible unit
        _ = thermal_noise(freqs=self.freqs,
                          npix=self.npix,
                          nb_antennas=self.nb_antennas,
                          total_obs_time=self.total_obs_time,
                          integration_time=self.integration_time,
                          collecting_area=self.collecting_area.to(units.cm**2),
                          Tsys=self.Tsys,
                          output_unit=self.output_unit,
                          beam_area=self.beam_area,
                          )
        
        # check Tsys units
        # issue warning if Tsys fed without units
        pytest.warns(UserWarning,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys.value,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # raise error if Tsys fed in wrong unit
        pytest.raises(ValueError,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys.value * units.m,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area,
                     )
        # convert inplace if compatible unit
        _ = thermal_noise(freqs=self.freqs,
                          npix=self.npix,
                          nb_antennas=self.nb_antennas,
                          total_obs_time=self.total_obs_time,
                          integration_time=self.integration_time,
                          collecting_area=self.collecting_area,
                          Tsys=self.Tsys.to(units.mK),
                          output_unit=self.output_unit,
                          beam_area=self.beam_area,
                          )
        
        #check beam_area units
        # issue warning if beam_area fed without units
        pytest.warns(UserWarning,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area.value,
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError,
                     thermal_noise,
                     freqs=self.freqs,
                     npix=self.npix,
                     nb_antennas=self.nb_antennas,
                     total_obs_time=self.total_obs_time,
                     integration_time=self.integration_time,
                     collecting_area=self.collecting_area,
                     Tsys=self.Tsys,
                     output_unit=self.output_unit,
                     beam_area=self.beam_area.value * units.K,
                     )
        # convert inplace if compatible unit
        _ = thermal_noise(freqs=self.freqs,
                          npix=self.npix,
                          nb_antennas=self.nb_antennas,
                          total_obs_time=self.total_obs_time,
                          integration_time=self.integration_time,
                          collecting_area=self.collecting_area,
                          Tsys=self.Tsys,
                          output_unit=self.output_unit,
                          beam_area=self.beam_area.to(units.arcsec**2),
                          )
        
    def test_make_noise_box(self):

        nobs_daily = int(self.daily_obs_time.si / self.integration_time.si)

        # correct usage with homogeneous sampling
        uv_map = np.ones((self.noise.npix, self.noise.npix, self.noise.nfreqs)) / nobs_daily
        noise_box = self.noise.make_noise_box(
            uv_map=uv_map,
            daily_obs_time=self.daily_obs_time,
        )

        assert noise_box.shape == (self.noise.npix, self.noise.npix, self.noise.nfreqs)
        assert noise_box.unit == self.output_unit

        # if not uv input, assume homogeneous sampling
        noise_box2 = self.noise.make_noise_box(
            uv_map=None,
            daily_obs_time=self.daily_obs_time
        )

        # checks if uv_map is provided
        # issue warning it it is not provided
        pytest.warns(UserWarning,
                     self.noise.make_noise_box)

        # check uv_map shape
        # raise error if fed in wrong shape
        pytest.raises(AssertionError,
                      self.noise.make_noise_box,
                      uv_map=np.ones((self.npix, self.npix)))

        # check uvmap is daily
        pytest.raises(AssertionError,
                      self.noise.make_noise_box,
                      uv_map=np.ones((self.npix, self.npix, self.nfreqs)))

        # check daily_obs_type input
        pytest.raises(ValueError,
                      self.noise.make_noise_box,
                      uv_map=None,
                      daily_obs_time=1.*units.K)
