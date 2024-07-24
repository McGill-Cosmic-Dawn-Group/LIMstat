import healpy as hp
import numpy as np
import warnings
from astropy import units, cosmology, constants
from scipy.integrate import quad
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.stats import binned_statistic_2d

from . import utils

try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError:
    HAVE_NUMBA = False


class cosmological_signal(object):
    """Generate observation box for given power spectrum.

        Parameters
        ----------
            ps: list/array of floats or method
                Power spectrum of the cosmological signal.
            freqs: list or array of floats
                Frequencies along the spectral window considered.
                Must be given with units or Hz are assumed.
            npix: int
                Number of pixels in one direction on the sky plane.
            ang_res: float
                Angular resolution of the instrument.
                Must be given with units or arcsec are assumed.
            cos: astropy.cosmology object
                Cosmology to use.
                Default is astropy.cosmology.Planck18.
            verbose: boolean
                Whether to print status reports.
                Default is False.
    """

    def __init__(
        self,
        ps,
        freqs,
        npix,
        ang_res,
        cos=cosmology.Planck18,
        verbose=False,
    ):

        # Check PS input
        if isinstance(ps, np.ndarray) or isinstance(ps, list):
            # here you interpolate
            assert np.shape(ps)[1] > 2, \
                "Need at least 3 values to interpolate power spectrum. " \
                f"Shape is {np.shape(ps)}, should be (2, nk)."
            k_theory = ps[0]
            p_theory = ps[1]
            self.ps = interp1d(k_theory, p_theory, bounds_error=False, fill_value=1e-10)
        elif callable(ps):
            self.ps = ps  # [mk^2*Mpc^3]
        else:
            raise ValueError('ps must be an array or a callable.')

        # Check cosmology
        assert isinstance(cos, cosmology.Cosmology), \
            'cos must be an astropy.Cosmology object.'

        # Check frequency units
        freqs = utils.check_unit(freqs, units.Hz, 'frequencies')
        self.freqs = np.atleast_1d(freqs)
        self.mean_freq = np.mean(self.freqs)
        self.freq_res = np.diff(freqs).mean() 
        self.nfreqs = self.freqs.size

        # Check angular resolution units
        ang_res = utils.check_unit(ang_res, units.arcsec, 'ang_res')
        self.ang_res = ang_res

        self.npix = int(npix)
        self.fov = (self.npix * self.ang_res).to(units.deg)

        # Get cosmological values
        z_array = utils.constants.f21/self.freqs - 1.
        self.avg_nu = np.mean(self.freqs)
        self.avg_z = float(np.mean(z_array))

        self.L = self.fov.to(units.rad).value \
            * cos.comoving_distance(self.avg_z).value
        self.Lz = (constants.c * (1.0 + self.avg_z) ** 2
                   / cos.H(self.avg_z).si / utils.constants.f21
                   * self.freq_res * self.nfreqs).to(units.Mpc).value

        self.delta_y = self.L / float(self.npix)  # sampling rate
        self.delta_x = self.L / float(self.npix)  # sampling rate
        self.delta_z = self.Lz / float(self.nfreqs)

        self.delta_ky = (2. * np.pi) / float(self.L)
        self.delta_kx = (2. * np.pi) / float(self.L)
        self.delta_kz = (2. * np.pi) / float(self.Lz)

        self.verbose = verbose

    def compute_kbox(self):
        """
        Outputs the norms of the Fourier modes in the box.

        Returns
        -------
            kbox: array of real loats
                Array of norms of Fourier modes included in
                simulated box. Shape is (npix, npix, nfreqs).
        """

        kx = (2. * np.pi) * np.fft.fftshift(
            np.fft.fftfreq(self.npix, d=self.delta_x)
        )
        ky = (2. * np.pi) * np.fft.fftshift(
            np.fft.fftfreq(self.npix, d=self.delta_y)
        )
        kz = (2. * np.pi) * np.fft.fftshift(
            np.fft.fftfreq(self.nfreqs, d=self.delta_z)
        )
        kbox = np.sqrt(
            kx[:, None, None] ** 2
            + ky[None, :, None] ** 2
            + kz[None, None, :] ** 2
        )

        return kbox

    def make_universe(self, **ps_kwargs):
        """
        Fill box with GRF with std PS/2.
        """

        # this box will hold all of the values of k for each pixel
        kbox = self.compute_kbox()  # Mpc-1

        # check if computed k values are far from those given in ps

        if isinstance(self.ps, interp1d) and (kbox.max() > self.ps.x.max()):
            warnings.warn('The range of k modes of the power spectrum provided '
                          'is much lower than of those computed for the box. '
                          'Higher k modes have been extrapolated to 0.')
        
        # store ps values along kbox
        powerbox = np.where(kbox != 0, self.ps(kbox, **ps_kwargs), 0.)  # mk2 Mpc3
        powerbox *= self.L * self.L * self.Lz  # Mpc3 * mk2 Mpc3 = mk2 mpc6

        # here's the box that will hold the random gaussian things
        means = np.zeros(kbox.shape)
        widths = np.sqrt(powerbox*0.5)  # sqrt(mk2 mpc6)= mk mpc3
        a, b = np.random.normal(
            means,
            widths,
            size=(2, self.npix, self.npix, self.nfreqs)
        )  # Mpc3

        if self.verbose:
            print("Taking iFFT of box")
        dk = (self.delta_kx*self.delta_ky*self.delta_kz)
        u = np.fft.fftshift(
            np.fft.irfftn(np.fft.ifftshift((a + b * 1j)*dk),
                          s=(kbox.shape),
                          norm='forward'))  # [mK]
        u /= (2.*np.pi)**3

        if self.verbose:
            print(f'Mean of simulated box = {np.mean(u.real):.3e}')

        return u.real


class foregrounds:
    """Class for :class:`foregrounds` objects."""

    def __init__(self, freqs, npix, ang_res,
                 beam_area,
                 RA=0.*units.deg,
                 Dec=-30.*units.deg,
                 point_sources=True, diffuse=True,
                 use_pygsm=True,
                 pygsm_year = '2008'):

        """
        Generate diffuse foregrounds and/or point sources for given FoV.

        Parameters
        ----------
            freqs: list or array of floats
                Frequencies along the spectral window considered.
                Must be given with units or Hz are assumed.
            npix: int
                Number of pixels in one direction on the sky plane.
            ang_res: float
                Angular resolution of the instrument.
                Must be given with units or arcsec are assumed.
            beam_area: astropy Quantity
                Integral of the peak-normalized synthesized beam.
            RA: float
                Right ascension of the centre of the field, in deg.
                Default is 0.
            Dec: float
                Declination of the centre of the field, in deg.
                Default is -30.
            point_sources: boolean
                Whether to include point sources or not in the simulation.
                Default is True.
            diffuse: boolean
                Whether to include diffuse foregrounds in the simulation.
                Default is True.
            use_pygsm: bool
                Whether to use pyGSM to generate the diffuse foregrounds
                or not. Default is True.
        """

        if use_pygsm:
            try:
                import pygsm
            except ModuleNotFoundError:
                warnings.warn('You do not have pygsm installed.'
                              'Setting pygsm to False.')
                use_pygsm = False

        self.use_pygsm = use_pygsm
        self.pygsm_year = pygsm_year
        # Check frequency units
        freqs = utils.check_unit(freqs, units.Hz, 'frequencies')
        self.freqs = np.atleast_1d(freqs)
        self.mean_freq = np.mean(self.freqs)
        self.freq_res = np.diff(freqs).mean()
        self.nfreqs = self.freqs.size

        # Check angular resolution units
        self.ang_res = utils.check_unit(ang_res, units.arcsec, 'ang_res')

        self.npix = int(npix)
        self.fov = (self.npix * self.ang_res).to(units.deg)

        # Beam area
        if beam_area is not None:
            beam_area = utils.comply_units(
                value=beam_area,
                default_unit=units.sr,
                quantity="beam_area",
                desired_unit=units.sr,
            ) * units.sr
        elif convert_data_to is not None:
            if ((data.unit.is_equivalent(units.mK) and
               convert_data_to.is_equivalent(units.Jy/units.beam)) or
               (data.unit.is_equivalent(units.Jy/units.beam) and
               convert_data_to.is_equivalent(units.mK))):
                raise ValueError('Missing beam_area to convert data!')
        self.beam_area = getattr(beam_area, "value", beam_area)

        self.Jy_to_K = units.brightness_temperature(
            self.freqs, beam_area
        )

        # Check units
        RA = utils.check_unit(RA, units.deg, 'RA')
        Dec = utils.check_unit(Dec, units.deg, 'declination')

        # coordinates of the pointing centre
        self.RA = RA.to(units.rad)  # hr or rad
        self.Dec = Dec.to(units.rad)  # rad

        # pixel coordinates
        # longitudes
        self.longitudes = np.linspace(
            -self.fov.to(units.rad) / 2,
            self.fov.to(units.rad) / 2,
            self.npix
        ) + self.RA.to(units.rad)
        # latitudes
        self.latitudes = np.linspace(
            -self.fov.to(units.rad) / 2,
            self.fov.to(units.rad) / 2,
            self.npix
        ) + self.Dec.to(units.rad)
        self.pixel_coordinates = np.meshgrid(self.longitudes, self.latitudes)

        # components
        self.diffuse = diffuse
        self.point_sources = point_sources

    def generate_point_sources(
        self,
        min_flux=10 * units.mJy,
        max_flux=100 * units.Jy,
        dist_params=(3.52, 0.307, -0.388, -0.0404, 0.0351, 0.006),
        n_flux_bins=501,
        ref_freq=150 * units.MHz,
        avg_spec_idx=-0.8,
        idx_std=0.05,
    ):
        """
        Generate point sources and place them on the observed sky grid.


        Default settings should provide the same distribution of fluxes as
        found in the GLEAM catalog, but it ignores source clustering.

        Parameters
        ----------
        min_flux: float
            Minimum flux of sources in Jy.
        max_flux: float
            Maximum flux of sources in Jy.
        dist_params: iterable of float
            Set of parameters describing the source count distribution.
            (These were taken from a notebook Steven Murray used to generate
            a synthetic point source catalog with the same statistics as
            GLEAM, based on Franzen+ 2018: https://arxiv.org/pdf/1604.03751.pdf)
        n_flux_bins: int
            Number of flux bins to use for generating the CDF from which we'll
            eventually draw the source fluxes.
        ref_freq: float
            Reference frequency at which the source fluxes are generated.
        avg_spec_idx: float
            Average spectral index describing each source's frequency evolution.
        idx_std: float
            Standard deviation describing the distribution of source spectral
            indices.
        """
        # Check units for inputs
        min_flux = utils.check_unit(min_flux, units.Jy, 'min_flux').value
        max_flux = utils.check_unit(max_flux, units.Jy, 'max_flux').value
        ref_freq = utils.check_unit(ref_freq, units.MHz, "ref_freq").value
        if (min_flux <= 0):
            raise ValueError("Minimum flux must be positive.")
        if max_flux <= min_flux:
            raise ValueError("Maximum flux must be greater than minimum flux.")

        # Figure out how many sources to generate.
        # I don't understand how this works, but it does.
        def dnds_franzen(flux, params):
            out = 10 ** (
                sum(p*np.log10(flux)**i for i, p in enumerate(params))
            )
            return flux**-2.5 * out

        def integrand(flux):
            return dnds_franzen(flux, dist_params)

        fluxes = np.logspace(
            np.log10(max_flux), np.log10(min_flux), n_flux_bins
        )
        density = [0]
        for i, flux in enumerate(fluxes[1:], start=1):
            density.append(
                quad(integrand, flux, fluxes[i-1])[0] + density[i-1]
            )
        density = np.array(density)
        survey_area = self.fov.to(units.rad).value ** 2
        n_src = np.random.poisson(survey_area*density.max())

        # Now compute the source fluxes.
        def cum_flux_integrand(flux):
            return flux * dnds_franzen(flux, dist_params)

        cum_flux = [0]
        for i, flux in enumerate(fluxes[1:], start=1):
            cum_flux.append(
                quad(cum_flux_integrand, flux, fluxes[i-1])[0] + cum_flux[i-1]
            )
        cum_flux = np.array(cum_flux)
        cdf = cum_flux / cum_flux.sum()
        spline = InterpolatedUnivariateSpline(cdf/cdf[-1], fluxes)
        source_fluxes = spline(np.random.uniform(size=n_src))

        # Now generate the source positions.
        colat = np.pi/2 - self.latitudes.to(units.rad).value
        lon = self.longitudes.to(units.rad).value
        theta_bounds = (colat[0], colat[-1])
        phi_bounds = (lon[0], lon[-1])
        theta, phi = utils.rand_pos(
            n_src, theta_bounds=theta_bounds, phi_bounds=phi_bounds
        )

        # Apply spectral evolution to the sources.
        spec_inds = np.random.normal(
            loc=avg_spec_idx, scale=idx_std, size=n_src
        )
        scalings = (
            self.freqs.to(units.Hz).value / ref_freq
        )[:,None] ** spec_inds[None,:]
        source_fluxes = source_fluxes[None,:] * scalings

        # Now bin up the source flux to the observed grid.
        dx = 0.5 * np.abs(colat[1] - colat[0])
        theta_bins = np.linspace(colat.min()-dx, colat.max()+dx, colat.size+1)
        phi_bins = np.linspace(lon.min()-dx, lon.max()+dx, lon.size+1)
        binned_flux = binned_statistic_2d(
            theta,
            phi,
            source_fluxes,
            statistic="sum",
            bins=(theta_bins, phi_bins),
        )[0]  # Should have shape (n_freq, n_pix, n_pix)

        # Convert to K
        conversion = (units.Jy/units.sr).to(
            units.K, equivalencies=units.brightness_temperature(self.freqs)
        )[:, None, None]
        pix_area = self.ang_res.to(units.rad).value ** 2
        return np.moveaxis(
            binned_flux * conversion / pix_area * units.K, 0, 2
        )
        # psf shape = (2048, 2048, 900) to match others

    def generate_diffuse(
            self,
            synchrotron=True,
            free_free=True,
            unresolved_pt=True,
            **kwargs
            ):
        """
        Setting up to retrieve foregrounds from pyGSM

        Parameters
        ----------
        synchrotron: boolean
            Whether to include synchrotron emission from the Galaxy
            in the diffuse foregrounds model.
            Default is True.
        free_free: boolean
            Whether to include free-free emission (bremsstrahlung)
            in the diffuse foregrounds model.
            Default is True.
        unresolved_pt: boolean
            Whether to include unresolved point sources in the diffuse
            foregrounds model.
            Default is True.
        """
        diffuse_foregounds = np.zeros((self.npix, self.npix, self.nfreqs))
        diffuse_foregounds *= units.K
        if synchrotron:
            diffuse_foregounds += self.generate_synchrotron(**kwargs)
        if free_free:
            diffuse_foregounds += self.generate_free_free(**kwargs)
        if unresolved_pt:
            diffuse_foregounds += self.generate_unresolved_point_sources(**kwargs)

        return diffuse_foregounds

    def generate_synchrotron(
            self,
            sync_params=[2.8, 0.1, 335.4], **kwargs):
        """
        Generate synchrotron emission from the Galaxy.

        Parameters
        ----------
            sync_params: tuple or list of 3 floats
                List of parameters for the synchrotron Liu et al. (2011) model
                if not using pyGSM.
                Must be a list of three floats: the mean and std of the
                spectral index distribution and the global amplitude of
                the signal, in K.

        Returns
        -------
            synchrotron_map: array of floats with unit K
                Maps of the synchrotron emission, in K, for each frequency
                in self.freqs. Shape (self.npix, self.npix, self.nfreqs).
        """
        # Check inputs
        if self.use_pygsm:
            try:
                if self.pygsm_year == '2016':
                    from pygsm import GlobalSkyModel2016
                elif self.pygsm_year == '2008':
                    from pygsm import GlobalSkyModel

            except ModuleNotFoundError:
                raise ValueError('Cannot set use pyGSM if the module '
                                 'is not installed.')
            if self.pygsm_year == '2016':
                gsm = GlobalSkyModel2016(
                    freq_unit='MHz',
                    data_unit='TCMB',
                    resolution='hi',
                    include_cmb=False,
                    )                
            elif self.pygsm_year == '2008':
                gsm = GlobalSkyModel(
                    freq_unit='MHz',
                    interpolation='pchip',
                    )
                
    
            sky_maps = gsm.generate(self.freqs.to(units.MHz).value)
            # assert sky_maps.shape[0] == self.npix ** 2, \
            #     "There is a dimension problem between your GSM map " \
            #     "and your field."
            obs_index = hp.pixelfunc.ang2pix(
                hp.get_nside(sky_maps),
                phi=self.pixel_coordinates[1].to(units.deg).value,
                theta=self.pixel_coordinates[0].to(units.deg).value,
                lonlat=True
                )

            synchrotron_map = sky_maps[:, obs_index]
            synchrotron_map = np.moveaxis(synchrotron_map, 0, 2) * units.K

        else:
            if len(sync_params) != 3:
                raise ValueError('sync_params must be a list of length 3. '
                                 'See documentation.')
            alpha_0_syn, sigma_syn, Asyn = sync_params
            if np.log10(Asyn) > 3 or np.log10(Asyn) < 1:
                warnings.warn('Check Asyn value: must be in K.')
            Asyn *= units.K
            # distribution of spectral indices
            alpha_syn = np.random.normal(
                alpha_0_syn,
                sigma_syn,
                size=(self.npix, self.npix))
            # synchrotron maps
            norm_freqs = self.freqs.to(units.MHz).value/150.
            synchrotron_map = Asyn * np.power(norm_freqs,
                                              -alpha_syn[..., None])

        return synchrotron_map

    def generate_free_free(self, ff_params=[2.15, 0.01, 33.5], **kwargs):
        """
        Generate a map of diffuse free-free emission following Liu+2011.

        Parameters
        ----------
            ff_params: tuple or list of 3 floats
                List of parameters for the free-free Liu+2011 model.
                Must be a list of three floats: the mean and std of the
                spectral index distribution and the global amplitude of
                the signal, in K.

        Returns
        -------
            free_free_map: array of floats with unit K
                Maps of the free-free emission, in K, for each frequency
                in self.freqs. Shape (self.npix, self.npix, self.nfreqs).
        """
        if len(ff_params) != 3:
            raise ValueError('ff_params must be a list of 3 parameters.')
        alpha_0_ff, sigma_ff, Aff = ff_params
        if np.log10(Aff) > 2 or np.log10(Aff) < 1:
            warnings.warn('Check Asyn value: must be in K.')
        Aff *= units.K
        # distribution of spectral indices
        alpha_ff = np.random.normal(
            alpha_0_ff,
            sigma_ff,
            size=(self.npix, self.npix)
            )
        # free-free map
        free_free_map = Aff * np.power(self.freqs.to(units.MHz).value/150.,
                                       -alpha_ff[..., None])

        return free_free_map

    def generate_unresolved_point_sources(
        self,
        min_flux=1*units.uJy,
        max_flux=100*units.mJy,
        count_index=-2,
        source_density=40000/units.sr,
        avg_n_src=None,
        ref_freq=150*units.MHz,
        mean_spec_index=-1.7,
        spec_index_width=0.3,
        use_numba=True,
        **kwargs
    ):
        """
        Generate a map of unresolved point sources.
        """
        # Here's the basic idea:
        # Define a power-law distribution for the source flux distribution.
        # Randomly generate some number of sources in each pixel given an
        # average source density.
        # For each pixel, generate the source fluxes and randomly assign
        # spectral indices to get the frequency evolution
        # Sum the contributions from every source in the pixel
        min_flux = utils.comply_units(
            min_flux, units.Jy, "min_flux", units.Jy
        )
        max_flux = utils.comply_units(
            max_flux, units.Jy, "max_flux", units.Jy
        )
        ref_freq = utils.comply_units(
            ref_freq, units.Hz, "ref_freq", units.Hz
        )
        freqs = utils.comply_units(
            self.freqs, units.Hz, "freqs", units.Hz
        )
        if avg_n_src is None:
            source_density = utils.comply_units(
                source_density, units.sr**-1, "source_density", units.sr**-1
            )
            pix_area = utils.comply_units(
                self.ang_res, units.arcsec, "ang_res", units.rad
            ) ** 2
            avg_n_src = source_density * pix_area

        if use_numba and HAVE_NUMBA:
            confusion_noise = _generate_confusion_noise(
                self.npix,
                min_flux,
                max_flux,
                count_index,
                avg_n_src,
                ref_freq,
                freqs,
                mean_spec_index,
                spec_index_width,
            )

        else:
            confusion_noise = np.zeros(
                (self.npix, self.npix, self.nfreqs), dtype=float
            )
            for i in range(self.npix):
                for j in range(self.npix):
                    n_src = int(
                        avg_n_src * np.random.normal(
                            loc=1, scale=0.1, size=1
                        )[0]
                    )
                    fluxes = utils.powerlaw(
                        min_flux, max_flux, count_index, n_src
                    )
                    indices = np.random.normal(
                        loc=mean_spec_index, scale=spec_index_width, size=n_src
                    )
                    confusion_noise[i,j] = fluxes @ (
                        freqs[None,:] / ref_freq
                    )**indices[:,None]

        return (confusion_noise * units.Jy).to(units.K, equivalencies=self.Jy_to_K)

    def generate_model(self, **kwargs):
        """
        Generate foregrounds model given class attributes.

        Returns
        -------
            foregrounds: array of floats
                Map of the foregrounds according to desired model.
                Units are K.
                Shape is (self.npix, self.npix, self.nfreqs).
        """

        foregrounds = np.zeros((self.npix, self.npix, self.nfreqs)) * units.K
        if self.diffuse:
            foregrounds += self.generate_diffuse(**kwargs)
        if self.point_sources:
            foregrounds += self.generate_point_sources(**kwargs)

        return foregrounds


class thermal_noise:

    def __init__(
        self,
        npix,
        freqs,
        nb_antennas,
        total_obs_time,
        collecting_area,
        integration_time=10.*units.s,
        Tsys=None,
        output_unit=units.Jy/units.beam,
        beam_area=None,
        T_rcv=100.*units.K
    ):

        """Container for thermal noise generator.

        Parameters
        ----------
            npix: int
                Number of pixels in one direction on the sky plane.
            freqs: list or array of floats
                Frequencies along the spectral window considered.
                Must be given with units or Hz are assumed.
            nb_antennas: int
                Number of antennas.
            total_obs_time: float
                Total integration time in obs.
            collecting_area: float
                Collecting area of a single antenna, in m2.
            integration_time: float
                Correlator integration time in sec.
            Tsys: float or array of floats
                System temperature in K. Can be either one value, then Tsys
                will be constant with frequency, or an array of nfreqs values.
                Default is None: using Haslam+2008 foreground temperature.
            output_unit: astropy unit, optional
                What units the noise simulation should be in.
                Default is Jy/sr.
            beam_area: astropy Quantity, optional
                Integral of the peak-normalized synthesized beam.
            T_rcv: astropy Quantity (temperature), optional
                Receiver temperature to be used if Tsys is not fed.

        """

        # Parameters of observation
        # Check frequency units
        freqs = utils.check_unit(freqs, units.Hz, 'frequencies')
        if np.size(freqs) <= 1:
            raise ValueError('freqs must have at least two channels.')
        self.freqs = np.atleast_1d(freqs)
        self.freq_res = np.diff(freqs).mean()
        self.nfreqs = self.freqs.size

        self.npix = int(npix)

        # Telescope parameters
        self.collecting_area = utils.check_unit(
            collecting_area,
            units.m**2,
            'collecting area'
        )
        self.nb_antennas = int(nb_antennas)
        self.nb_baselines = self.nb_antennas * (self.nb_antennas - 1) / 2.
        self.integration_time = utils.check_unit(
            integration_time,
            units.second,
            'integration time'
        )
        self.total_obs_time = utils.check_unit(
            total_obs_time,
            units.hr,
            'total observing time'
        )

        if Tsys is None:
            # self.Tsys = 180. * units.K \
            #     * (freqs.to(units.MHz).value / 180.)**(-2.6)
              # receiver temperature
            self.Tsys = T_rcv + 60. * units.K \
                * (freqs.to(units.MHz).value / 300.)**(-2.55)
        else:
            Tsys = utils.check_unit(Tsys, units.K, 'Tsys')
            if np.size(Tsys) == 1:
                self.Tsys = np.ones(self.nfreqs) * Tsys
            else:
                if np.size(Tsys) == self.nfreqs:
                    self.Tsys = Tsys
                else:
                    raise ValueError('Tsys must be either one value '
                                     'or an array of size nfreqs.')

        # output units
        self.noise_unit = output_unit
        self.beam_area = beam_area
        if self.beam_area is not None:
            self.beam_area = utils.comply_units(
                value=beam_area,
                default_unit=units.sr,
                quantity="beam_area",
                desired_unit=units.sr,
            ) * units.sr
            self.Jy_to_K = units.brightness_temperature(
                self.freqs, self.beam_area
            )
        else:
            if self.noise_unit.is_equivalent(units.K, equivalencies=self.Jy_to_K):
                raise ValueError('Must give beam_area if the required output'
                                 'unit is Jy/sr or K.')

        # compute std per visibility
        self.std_per_vis = (
            np.sqrt(2.) * constants.k_B.si * self.Tsys
            / self.collecting_area.si
            / np.sqrt(self.integration_time.si * self.freq_res.si)
            ).to(units.Jy)/units.beam  # std for total observing time
        self.std_per_vis = utils.comply_units(
            self.std_per_vis,
            self.noise_unit,
            'noise unit',
            self.noise_unit,
            equivalencies=self.Jy_to_K
        ) * self.noise_unit

        # compute std for full observation
        self.std = (
            np.sqrt(2.) * constants.k_B.si * self.Tsys
            / self.collecting_area.si
            / np.sqrt(self.nb_baselines)
            / np.sqrt(self.total_obs_time.si * self.freq_res.si)
            ).to(units.Jy)/units.beam  # std for total observing time
        self.std = utils.comply_units(
            self.std,
            self.noise_unit,
            'noise unit',
            self.noise_unit,
            equivalencies=self.Jy_to_K
        ) * self.noise_unit


    def make_noise_box(self, uv_map=None, daily_obs_time=4.*units.hr):

        """Make noise box given uv sampling.

        Parameters
        ----------
            uv_map: array
                uv map calculated from the distribution of antennas
                and observing time over a day.
                Default is None: equal uv sampling of each point.
            daily_obs_time: astropy Quantity in time
                Daily observation time, in hours.
                Default is 4 hours.

        Returns
        -------
            noise: array of floats
                Noise simulation of dimension
                (self.npix, self.npix, self.nfreqs).
        """

        daily_obs_time = utils.check_unit(
            daily_obs_time,
            units.hr,
            'daily observing time'
        )
        nobs_daily = int(daily_obs_time.si / self.integration_time.si)

        # check for uv map and define the variable
        if uv_map is None:
            warnings.warn('No uv map provided. Setting uv_map to 1s.')
            uvmap = np.ones((self.npix, self.npix, self.nfreqs)) / nobs_daily
        else:
            uvmap = np.copy(uv_map)
        # check the dimensions
        assert uvmap.shape == (self.npix, self.npix,self.nfreqs), \
            'Your uv map has the wrong shape, should be ({}, {}, {}).'.format(self.npix, 
                                                                              self.npix, 
                                                                              self.nfreqs)
        # each point covered must be sampled at least once per day
        assert np.isclose(np.min(uvmap[uvmap>0]) * nobs_daily, 1.), \
            "uv map incompatible with daily observation time."

        # rescale uv map for full observing season
        # uvmap * ndays * nobs_daily
        uvmap *= (self.total_obs_time.si / self.integration_time.si)

        # get std in Jy/beam to avoid losing frequency dependence
        std_Jy = self.std_per_vis.to(units.Jy/units.beam, 
                                    equivalencies=self.Jy_to_K)
        
        # simulate real component of noise
        noise_real = np.random.normal(
            0,
            std_Jy.value*np.sqrt(.5),
            size=(self.npix, self.npix, self.nfreqs))
        
        # simulate imaginary component of noise
        noise_imag = np.random.normal(
            0,
            std_Jy.value*np.sqrt(.5),
            size=(self.npix, self.npix, self.nfreqs))
        
        # making complex noise variable
        noise_ft = noise_real + 1j*noise_imag
        
        # apply the uv map
        # TO DO -- what happens at points where uvmap = 0?
        noise_ft = np.divide(noise_ft, np.sqrt(uvmap), where=uvmap != 0.)
        noise_ft[uvmap==0.] = 0.

        # inverse fourier transform
        noise = np.fft.ifftn(noise_ft, axes=(0, 1)).real * std_Jy.unit

        return noise.to(self.noise_unit, equivalencies=self.Jy_to_K)


def _generate_confusion_noise(*args, **kwargs):
    # no-op in case we don't have numba
    pass


if HAVE_NUMBA:

    @numba.njit
    def _generate_confusion_noise(
        n_pix,
        min_flux,
        max_flux,
        count_index,
        avg_n_src,
        ref_freq,
        freqs,
        mean_spec_index,
        spec_index_width,
    ):
        """numba-accelerated implementation for source confusion noise."""
        confusion_noise = np.zeros(
            (n_pix, n_pix, freqs.size), dtype=float
        )
        a = min_flux**count_index
        b = min_flux**count_index
        for i in range(n_pix):
            for j in range(n_pix):
                n_src = int(
                    avg_n_src * np.random.normal(loc=1, scale=0.1, size=1)[0]
                )
                src_fluxes = np.random.uniform(0, 1, n_src)
                src_fluxes = (a + (b - a)*src_fluxes) ** (1/count_index)
                spec_inds = np.random.normal(
                    loc=mean_spec_index, scale=spec_index_width, size=n_src
                ).reshape(-1, 1)
                freq_scaling = (freqs.reshape(1, -1) / ref_freq)**spec_inds
                confusion_noise[i,j] = src_fluxes @ freq_scaling
        return confusion_noise
