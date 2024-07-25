import unittest
import pytest

import numpy as np
from astropy import units, cosmology

from limstat.simulations import cosmological_signal


class test_cosmological_signal(unittest.TestCase):
    def setUp(self):

        nfreqs = 15
        df = 100e3 * units.Hz
        avg_nu = 150.e6 * units.Hz

        self.spw_window = np.arange(avg_nu.value - df.value*nfreqs//2,
                                    avg_nu.value + df.value*(nfreqs//2),
                                    step=df.value) * units.Hz
        self.npix = 50
        self.ang_res = 50. * units.arcsec

        self.k_array = np.logspace(-2, 2, 50)
        self.ps_array = self.flat_ps(self.k_array)

        self.cos = cosmology.Planck18

    def flat_ps(self, k, amp=10e-3):
        """
        Generate a flat power spectrum of amplitude amp.

        Parameters
        ----------
            k: float or array of flats
                Fourier mode in Mpc-1.
            amp: float
                Desired amplitude for the PS, in K2.

        """
        k = np.atleast_1d(k)
        return amp * np.ones(k.shape)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):

        # proper usage
        _ = cosmological_signal(
            self.flat_ps,
            self.spw_window,
            self.npix,
            self.ang_res
            )

        # check cos input
        pytest.raises(AssertionError,
                      cosmological_signal,
                      self.flat_ps,
                      self.spw_window,
                      self.npix,
                      self.ang_res,
                      cos='cos')

        # check freqs units
        # issue warning if freqs fed without unit
        pytest.warns(UserWarning,
                     cosmological_signal,
                     self.flat_ps,
                     freqs=self.spw_window.value,
                     npix=self.npix,
                     ang_res=self.ang_res,
                     )
        # raise error if freqs fed in wrong unit
        pytest.raises(ValueError,
                      cosmological_signal,
                      self.flat_ps,
                      freqs=self.spw_window.value * units.K,
                      npix=self.npix,
                      ang_res=self.ang_res,
                      )
        # convert inplace if compatible unit
        _ = cosmological_signal(
            self.flat_ps,
            freqs=self.spw_window.to(units.MHz),
            npix=self.npix,
            ang_res=self.ang_res,
            )

        # check ang_res units
        # issue warning if freqs fed without unit
        pytest.warns(UserWarning,
                     cosmological_signal,
                     self.flat_ps,
                     freqs=self.spw_window,
                     npix=self.npix,
                     ang_res=self.ang_res.value,
                     )
        # raise error if ang_res fed in wrong unit
        pytest.raises(ValueError,
                      cosmological_signal,
                      self.flat_ps,
                      freqs=self.spw_window,
                      npix=self.npix,
                      ang_res=self.ang_res.value * units.Jy,
                      )
        # convert inplace if compatible unit
        _ = cosmological_signal(
            self.flat_ps,
            freqs=self.spw_window,
            npix=self.npix,
            ang_res=self.ang_res.to(units.deg),
            )

        # check ps input
        _ = cosmological_signal(
                np.c_[self.k_array, self.ps_array].T,
                freqs=self.spw_window,
                npix=self.npix,
                ang_res=self.ang_res,
                )
        # array with too few values
        pytest.raises(AssertionError,
                      cosmological_signal,
                      [[0.1, 1.], [0.1, 1.]],
                      freqs=self.spw_window,
                      npix=self.npix,
                      ang_res=self.ang_res,
                      )
        # neither an array nor a callable
        pytest.raises(ValueError,
                      cosmological_signal,
                      'blah',
                      freqs=self.spw_window,
                      npix=self.npix,
                      ang_res=self.ang_res,
                      )

    def test_make_cosmological_signal(self):

        # proper usage
        def_universe = cosmological_signal(
            self.flat_ps,
            self.spw_window,
            self.npix,
            self.ang_res,
            verbose=True
            )
        uni = def_universe.make_universe()
        assert np.shape(uni) == (self.npix, self.npix, self.spw_window.size)
