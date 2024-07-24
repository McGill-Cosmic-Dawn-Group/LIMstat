import unittest
import pytest

import numpy as np
from astropy import units, cosmology

from ska_simulator.simulations import cosmological_signal
from ska_simulator.power_spectrum import power_spectrum


class test_power_spectrum(unittest.TestCase):
    def setUp(self):

        nfreqs = 15
        df = 100e3 * units.Hz
        avg_nu = 150.e6 * units.Hz
        self.rest_freq = 1420. * units.MHz
        self.spw_window = np.arange(avg_nu.value - df.value*nfreqs//2,
                                    avg_nu.value + df.value*(nfreqs//2),
                                    step=df.value) * units.Hz
        self.npix = 51
        self.ang_res = 10. * units.arcsec
        self.fov = (self.npix * self.ang_res).to(units.rad)
        self.freq_taper = 'bh'
        self.space_taper = 'bh'
        self.convert_data_to = None
        beam_fwhm = 0.035 * units.deg
        beam_sigma = beam_fwhm / (8 * np.log(2))**0.5
        self.beam_area = 2 * np.pi * beam_sigma**2
        self.cos = cosmology.Planck18
        self.Jy_to_K = units.brightness_temperature(
            self.spw_window, self.beam_area
        )

        # GRF cosmological signal
        gaussian_universe = cosmological_signal(
            self.flat_ps,
            self.spw_window,
            self.npix,
            self.ang_res
            )
        self.data = gaussian_universe.make_universe() * units.mK

        # gaussian psf with beam sigmal 0.02 deg
        thetax = np.linspace(
            -self.fov.to(units.deg).value/2,
            self.fov.to(units.deg).value/2,
            self.npix
        )
        thetay = thetax[:, None]
        beam_sigma = 0.02  # deg
        self.PSF = np.exp(-(thetax**2 + thetay**2)/2./beam_sigma**2)
        self.PSF = self.PSF[:, :, None] * np.ones(self.spw_window.size)

        # proper usage
        self.proper = power_spectrum(
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            verbose=False,
        )
        self.ps_cube = self.proper.FFT_crossxx()
        self.kpara, self.kperp, self.ps_2d = self.proper.compute_2D_pspec(
            ps_data=self.ps_cube
        )
        _ = self.proper.compute_spatial_fourier_modes()
        kmag_3d = np.sqrt(
            self.proper.kx[None, :, None] ** 2
            + self.proper.ky[:, None, None] ** 2
            + self.proper.k_par[None, None, :] ** 2
        )
        kmin = np.min(kmag_3d) * 2.
        kmax = np.max(kmag_3d) / 2.
        bin_edges = np.histogram_bin_edges(
            np.sort(kmag_3d.flatten()),
            bins=30,
            range=(kmin, kmax)
        )
        dk = np.diff(bin_edges).mean()
        self.kbins = np.arange(bin_edges[0]+dk/2, bin_edges[-1], step=dk)
        _, self.ps_1d = self.proper.compute_1D_pspec(
            ps_data=self.ps_cube,
            kbins=self.kbins
        )

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):

        # raise error if data has no appropriate shape
        pytest.raises(ValueError,
                      power_spectrum,
                      data=self.data[:, :, 12],
                      theta_x=self.fov,
                      theta_y=self.fov,
                      freqs=self.spw_window,
                      )
        pytest.raises(ValueError,
                      power_spectrum,
                      data=self.data.T,
                      theta_x=self.fov,
                      theta_y=self.fov,
                      freqs=self.spw_window,
                      )

        # test different units for frequencies
        test = power_spectrum(
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window.to(units.Hz),
        )
        _, ps_test = test.compute_1D_pspec()
        assert np.allclose(ps_test, self.ps_1d)
        # assume MHz if no unit given and raises warning
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window.to(units.MHz).value,
        )
        # raise ValueError if unit is not frequency
        pytest.raises(
            ValueError,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=np.ones(self.spw_window.size) * units.m,
        )

        # same for rest_freq
        test = power_spectrum(
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            rest_freq=self.rest_freq.to(units.Hz)
        )
        _, ps_test = test.compute_1D_pspec()
        assert np.allclose(ps_test, self.ps_1d)
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            rest_freq=self.rest_freq.to(units.MHz).value,
        )
        pytest.raises(
            ValueError,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            rest_freq=1.*units.m,
        )

        # test beam_area units
        test = power_spectrum(
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            beam_area=self.beam_area.to(units.sr)
        )
        _ = test.compute_1D_pspec()
        _ = test.compute_2D_pspec()
        assert np.allclose(ps_test, self.ps_1d)
        # assume steradian if no unit given and raises warning
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            beam_area=self.beam_area.to(units.sr).value
        )
        # raise ValueError if unit is not angle
        pytest.raises(
            ValueError,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            beam_area=1.*units.m,
        )

        # check data units and convert if necessary
        test = power_spectrum(
            data=self.data.to(units.K),
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            convert_data_to=units.mK
        )
        _, ps_test = test.compute_1D_pspec()
        assert np.allclose(ps_test, self.ps_1d)
        # assume MHz if no unit given and raises warning
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data.value,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
        )
        # raise warning if unit not compatible with itensity
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data.value * units.s,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
        )
        # raise warning if data is complex
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data + 1j * self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
        )

        # test different units for theta_x and theta_y
        test = power_spectrum(
            data=self.data,
            theta_x=self.fov.to(units.deg),
            theta_y=self.fov.to(units.deg),
            freqs=self.spw_window,
        )
        _, ps_test = test.compute_1D_pspec()
        assert np.allclose(ps_test, self.ps_1d)
        # assume rad if no unit given and raises warning
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data,
            theta_x=self.fov.value,
            theta_y=self.fov,
            freqs=self.spw_window,
        )
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov.value,
            freqs=self.spw_window,
        )
        # raise ValueError if unit is not angle
        pytest.raises(
            ValueError,
            power_spectrum,
            data=self.data,
            theta_x=1.*units.m,
            theta_y=self.fov,
            freqs=self.spw_window,
        )
        pytest.raises(
            ValueError,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=1.*units.m,
            freqs=self.spw_window,
        )

        # check data2 input
        # must have same shape as data
        pytest.raises(
            AssertionError,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            data2=self.data[:, :, :12]
            )
        # remove imaginary values
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            data2=self.data + 1j * self.data,
            )

        # check PSF
        # proper usage
        _ = power_spectrum(
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            PSF=self.PSF,
        )
        # must have same shape as data
        pytest.raises(
            ValueError,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            PSF=self.PSF[:, :, :12]
            )
        # remove unit
        pytest.warns(
            UserWarning,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            PSF=self.PSF * self.data.unit,
        )

    def test_FFT_crossxx(self):

        # correct usage
        ps_data = self.proper.FFT_crossxx()
        assert ps_data.unit == self.proper.pk_unit

        # def object with PSF to deconvolve
        _ = power_spectrum(
            data=self.data,
            theta_x=self.fov.to(units.rad),
            theta_y=self.fov.to(units.rad),
            freqs=self.spw_window,
            PSF=self.PSF,
        )

        # test normalisation with convolved data
        fft_data = self.proper.take_ft(self.data, axes=(0, 1))
        fft_data *= self.proper.take_ft(self.PSF, axes=(0, 1))
        convolved_data = self.proper.take_ift(fft_data, axes=(0, 1)).real \
            * ((2*np.pi)**self.data.ndim)
        test = power_spectrum(
            data=convolved_data * self.data.unit,
            theta_x=self.fov.to(units.rad),
            theta_y=self.fov.to(units.rad),
            freqs=self.spw_window,
            PSF=self.PSF,
        )
        _, ps1d = test.compute_1D_pspec(
            kbins=self.kbins
        )
        assert np.allclose(ps1d, self.ps_1d, atol=2e-3)

        # Dirac-like PSF must be like no PSF
        dirac_PSF = np.zeros(self.data.shape)
        dirac_PSF[self.npix//2, self.npix//2, :] = 1.
        deconvolved_ps = power_spectrum(
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            PSF=dirac_PSF,
        )
        deconv_ps_cube = deconvolved_ps.FFT_crossxx()
        # assert np.allclose(deconv_ps_cube, self.ps_cube)
        deconv_ps1d = deconvolved_ps.compute_1D_pspec(
            ps_data=deconv_ps_cube,
            kbins=self.kbins
        )
        assert np.allclose(deconv_ps1d[1], self.ps_1d, atol=2e-3)
        # also with taper
        pytest.raises(
            NotImplementedError,
            power_spectrum,
            data=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            PSF=dirac_PSF,
            space_taper='bh',
            freq_taper='bh'
        )
        # deconvolved_tapered_ps = power_spectrum(
        #     data=self.data,
        #     theta_x=self.fov,
        #     theta_y=self.fov,
        #     freqs=self.spw_window,
        #     PSF=dirac_PSF,
        #     space_taper='bh',
        #     freq_taper='bh'
        # )
        # deconv_tapered_ps_cube = deconvolved_tapered_ps.FFT_crossxx(
        #     deconvolve=False
        # )
        # tapered_ps = power_spectrum(
        #     data=self.data,
        #     theta_x=self.fov,
        #     theta_y=self.fov,
        #     freqs=self.spw_window,
        #     space_taper='bh',
        #     freq_taper='bh'
        # )
        # tapered_ps_cube = tapered_ps.FFT_crossxx()
        # assert np.allclose(deconv_tapered_ps_cube, tapered_ps_cube)

    def test_FFT_crossxy(self):

        # raise error if try to compute cross-spectrum without data2
        pytest.raises(
            ValueError,
            self.proper.FFT_crossxy,
        )
        # proper usage with deconvolution
        cross_ps = power_spectrum(
            data=self.data,
            data2=self.data,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
        )
        # auto spectrum and cross spectrum with twice the same
        # dataset should lead to identical results
        cross_ps_cube = cross_ps.FFT_crossxy()
        assert np.allclose(cross_ps_cube, self.ps_cube)
        # assert np.allclose(cross_ps_cube, self.ps_cube)
        _, test_1d = cross_ps.compute_1D_pspec(
            ps_data=cross_ps_cube,
            kbins=self.kbins)
        assert np.allclose(test_1d, self.ps_1d)
        # proper usage with deconvolution
        # Dirac-like PSF must be like no PSF
        dirac_PSF = np.zeros(self.data.shape)
        dirac_PSF[self.npix//2, self.npix//2, :] = 1.
        deconvolved_cross_ps = power_spectrum(
            data=self.data,
            data2=self.data,
            theta_x=self.fov.to(units.rad),
            theta_y=self.fov.to(units.rad),
            freqs=self.spw_window,
            PSF=dirac_PSF,
        )
        deconv_ps_cube = deconvolved_cross_ps.FFT_crossxy()
        # assert np.allclose(deconv_ps_cube, self.ps_cube)
        _, deconv_ps1d = deconvolved_cross_ps.compute_1D_pspec(
            ps_data=deconv_ps_cube,
            kbins=self.kbins
        )
        assert np.allclose(deconv_ps1d, self.ps_1d, atol=2e-3)

    def test_taper(self):

        # test taper options
        # with dspec method as input
        test = power_spectrum(
            data=self.data,
            theta_x=self.fov.to(units.deg),
            theta_y=self.fov.to(units.deg),
            freqs=self.spw_window,
            freq_taper='bh',
            space_taper='bh',
            verbose=True
        )
        _, ps_test = test.compute_1D_pspec()

        # raise value error if input not compatible with dspec
        pytest.raises(
            ValueError,
            power_spectrum,
            data=self.data,
            theta_x=self.fov.to(units.deg),
            theta_y=self.fov.to(units.deg),
            freqs=self.spw_window,
            freq_taper='blah',
            space_taper='bh'
        )
        pytest.raises(
            ValueError,
            power_spectrum,
            data=self.data,
            theta_x=self.fov.to(units.deg),
            theta_y=self.fov.to(units.deg),
            freqs=self.spw_window,
            freq_taper='bh',
            space_taper='blah'
        )

    def test_compute_2D_pspec(self):

        # check outputs for equivalent calls
        # input ps_data or not
        _, _, pspec2d_test = self.proper.compute_2D_pspec(self.ps_cube)
        _, _, pspec2d_test2 = self.proper.compute_2D_pspec()
        assert np.allclose(pspec2d_test, pspec2d_test2)
        # input kperp_bin and kpar_bin or not
        _, _, pspec2d_test3 = self.proper.compute_2D_pspec(
            k_perp_bin=self.kperp,
            k_par_bin=self.kpara
        )
        assert np.allclose(pspec2d_test, pspec2d_test3)

        # checks on kperp_bin
        pytest.raises(
            AssertionError,
            self.proper.compute_2D_pspec,
            k_perp_bin=[]
        )
        # checks on kpar_bin
        pytest.raises(
            AssertionError,
            self.proper.compute_2D_pspec,
            k_par_bin=[]
        )

        # if ps_data has no unit, assume it is self.data_unit**2 Mpc3
        # and raise warning
        pytest.warns(UserWarning,
                     self.proper.compute_2D_pspec,
                     ps_data=self.ps_cube.value)
        # if ps_data has imaginary terms, keep only real terms
        # and raise warning
        pytest.warns(UserWarning,
                     self.proper.compute_2D_pspec,
                     ps_data=self.ps_cube + 1j * self.ps_cube)
        # raise assertion error if wrong shape
        pytest.raises(AssertionError,
                      self.proper.compute_2D_pspec,
                      ps_data=self.ps_cube[:, :, 2])

    def test_compute_1d_from_2d(self):

        # check result is equivalent to direct spherical average
        _, ps_test = self.proper.compute_1D_pspec(
            kbins=self.kbins[::3],
            ps_data=self.ps_cube,
        )
        _, ps_test2 = self.proper.compute_1d_from_2d(
            ps_data=self.ps_cube,
            kbins=self.kbins[::3]
        )
        # assert np.allclose(ps_test, ps_test2)

        # checks on kbins
        # no input
        _ = self.proper.compute_1d_from_2d()
        # wrong input: dk < 0
        pytest.raises(
            AssertionError,
            self.proper.compute_1d_from_2d,
            kbins=[2., 1.]
        )
        # checks on kpar_bin: size(bin_edges) <= 1
        pytest.raises(
            AssertionError,
            self.proper.compute_1d_from_2d,
            kbins=[1.]
        )

    def test_compute_1D_pspec(self):

        # check outputs for equivalent calls
        # input ps_data or not
        ktest, pspec1d_test = self.proper.compute_1D_pspec(self.ps_cube)
        _, pspec1d_test2 = self.proper.compute_1D_pspec()
        assert np.allclose(pspec1d_test, pspec1d_test2)

        # wrong kbins input: dk < 0
        pytest.raises(
            AssertionError,
            self.proper.compute_1D_pspec,
            kbins=[2., 1.]
        )
        # checks on kbins: size(bin_edges) <= 1
        pytest.raises(
            AssertionError,
            self.proper.compute_1D_pspec,
            kbins=[1.]
        )

        # if ps_data has no unit, assume it is self.data_unit**2 Mpc3
        # and raise warning
        pytest.warns(UserWarning,
                     self.proper.compute_1D_pspec,
                     ps_data=self.ps_cube)
        # if ps_data has imaginary terms, keep only real terms
        # and raise warning
        pytest.warns(UserWarning,
                     self.proper.compute_1D_pspec,
                     ps_data=self.ps_cube + 1j * self.ps_cube)
        # raise assertion error if wrong shape
        pytest.raises(AssertionError,
                      self.proper.compute_1D_pspec,
                      ps_data=self.ps_cube[:, :, 2])

        # get dimensionless ps
        _, dimless_test = self.proper.compute_1D_pspec(
            ps_data=self.ps_cube,
            dimensionless=True)
        assert np.allclose(dimless_test, pspec1d_test*ktest**3/4./np.pi)

    def test_take_ft(self):

        # check if iFT of FT gives data
        fft_data = self.proper.take_ft(data=self.data)
        data_bis = self.proper.take_ift(ft_data=fft_data*((2*np.pi)**fft_data.ndim))
        assert np.allclose(self.data, data_bis)

        # raise valueerror for non 3d array
        pytest.raises(ValueError,
                      self.proper.take_ft,
                      data=self.data[..., 12])
        pytest.raises(ValueError,
                      self.proper.take_ift,
                      ft_data=fft_data[..., 12])

        # check if iFT of FT gives data, enven in only 1d
        fft_data = self.proper.take_ft(data=self.data, axes=[2])
        data_bis = self.proper.take_ift(ft_data=fft_data*((2*np.pi)**fft_data.ndim) , axes=[2])
        assert np.allclose(self.data, data_bis)

    def test_conversions(self):
        # proper usage takes mK input and output mK
        data_jybeam = self.data.to(units.Jy/units.beam,
                                   equivalencies=self.Jy_to_K)
        test = power_spectrum(
            data=data_jybeam,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            verbose=False,
            convert_data_to=units.mK,
            beam_area=self.beam_area,
        )
        _, _, ps_2d = test.compute_2D_pspec()
        assert np.allclose(ps_2d, self.ps_2d)
        _, ps_1d = self.proper.compute_1D_pspec(
            kbins=self.kbins
        )
        assert np.allclose(ps_1d, self.ps_1d)

        # cannot convert if no beam_area
        pytest.raises(
            ValueError,
            power_spectrum,
            data=data_jybeam,
            theta_x=self.fov,
            theta_y=self.fov,
            freqs=self.spw_window,
            verbose=False,
            convert_data_to=units.mK,
            beam_area=None,
        )

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

