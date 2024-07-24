from cached_property import cached_property
import numpy as np
from astropy import constants, units, cosmology
import warnings
from scipy.stats import binned_statistic_2d
from . import utils


class power_spectrum(object):
    """Class containing tools necessary to build a power spectrum.

    This code computes the cylindrical and spherical power spectra from a
    3D cube of data. These 3D cubes are assumed to be within a small enough
    frequency range that they can be treated as coeval. We use the middle
    slice as the one which sets the comoving scales for the cube.
    """

    def __init__(
        self,
        data,
        theta_x,
        theta_y,
        freqs,
        data2=None,
        rest_freq=1420.*units.MHz,
        cosmo=cosmology.Planck18,
        freq_taper=None,
        space_taper=None,
        freq_taper_kwargs=None,
        space_taper_kwargs=None,
        convert_data_to=None,
        beam_area=None,
        PSF=None,
        verbose=False,
    ):
        """
        Initialisation of the Power_Spectrum class.

        Parameters
        ----------
            data: array of real floats.
                Array containing the data we want to take the power spectrum
                of.
                It should be fed with units equivalent to K or Jy/beam.
                The third dimension must be the frequency axis.
            theta_x: float
                Angular size along one sky-plane dimension in RADIANS.
            theta_y: float
                Angular size along the other sky-plane dimension in RADIANS.
            freqs: list or array of floats.
                List of frequencies the signal was measured on.
                Frequencies must be given in MHZ.
            data2: array of real floats.
                For cross-spectra: data2 is cross-correlated with data
                to compute the power spectrum.
                It should have the same units and shape as data.
            rest_freq: float
                Rest-frequency of the emission line in units.MHz.
                Default is 1420 for the 21cm line.
            cosmo: astropy.cosmology class
                Cosmology to use for computations.
                Default is Planck18.
            freq_taper: str, optional
                Name of the taper to use when Fourier transforming along the
                frequency axis. Must be available in scipy.signal.windows.
            space_taper: str, optional
                Name of the taper to use when Fourier transforming along the
                spatial axes. Must be a dspec.gen_window function.
            freq_taper_kwargs: dict, optional
                Extra parameters to use when calculating the frequency taper.
            space_taper_kwargs: dict, optional
                Extra parameters to use when calculating the spatial taper.
            convert_data_to: astropy unit, optional
                What units the data should be converted to on initialization.
                Default is to leave the data in whatever units it was
                provided in.
            beam_area: astropy Quantity, optional
                Integral of the peak-normalized synthesized beam.
            PSF: array of floats
                Point-spread function of the telescope.
                Should have no unit and same shape as data.
            verbose: bool
                Whether to output messages when running functions.
        """
        # ensure input shapes are consistent
        if np.ndim(data) != 3:
            raise ValueError(
                'Data must be a 3d array of dim (npix, npix, nfreqs).'
            )
        if np.shape(data)[-1] != np.size(freqs):
            raise ValueError(
                "The last axis of the data array must be the frequency axis."
            )

        self.freqs = utils.comply_units(
            value=freqs,
            default_unit=units.MHz,
            quantity="freqs",
            desired_unit=units.Hz,
        )
        self.rest_freq = utils.comply_units(
            value=rest_freq,
            default_unit=units.MHz,
            quantity="rest_freq",
            desired_unit=units.Hz,
        )

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
            self.freqs*units.Hz, beam_area
        )

        if convert_data_to is None and hasattr(data, "unit"):
            self.data_unit = data.unit
            if not self.data_unit.is_equivalent(
                units.K, equivalencies=self.Jy_to_K
            ):
                warnings.warn(f"Your data has units {self.data_unit} which "
                              "are not compatible with mK or Jy/beam...")
        else:
            self.data_unit = convert_data_to or units.mK
        self.pk_unit = self.data_unit**2 * units.Mpc ** 3

        self.data = utils.comply_units(
            value=data,
            default_unit=self.data_unit,
            quantity="data",
            desired_unit=convert_data_to,
            equivalencies=self.Jy_to_K,
        )

        if np.any(self.data.imag):
            warnings.warn(
                "Provided data is complex; taking the real part...",
                stacklevel=2,
            )
        self.data = self.data.real

        # get all the z info from the mid req
        self.mid_freq = np.mean(self.freqs)
        self.z = (self.rest_freq / self.mid_freq) - 1.

        # Figure out the angular extent of the map.
        self.theta_x = utils.comply_units(
            value=theta_x,
            default_unit=units.rad,
            quantity="theta_x",
            desired_unit=units.rad,
        )
        self.theta_y = utils.comply_units(
            value=theta_y,
            default_unit=units.rad,
            quantity="theta_y",
            desired_unit=units.rad,
        )

        self.y_npix = self.data.shape[0]
        self.x_npix = self.data.shape[1]
        self.freq_npix = self.data.shape[2]

        # these two lines give you the physical dimensions of a pixel
        # (inverse of sampling ratealong each axis)
        self.delta_thetay = self.theta_y / self.data.shape[0]
        self.delta_thetax = self.theta_x / self.data.shape[1]
        self.delta_freq = np.diff(self.freqs).mean()

        # define Fourier axes
        self.compute_eta_nu()

        self.cosmo = cosmo
        self.verbose = verbose

        self.freq_taper = None
        self.theta_x_taper = None
        self.theta_y_taper = None
        self.sky_taper = None
        self.taper = None
        self._parse_taper_info(
            freq_taper=freq_taper,
            space_taper=space_taper,
            freq_taper_kwargs=freq_taper_kwargs,
            space_taper_kwargs=space_taper_kwargs,
        )

        # checks on data2
        self.data2 = data2
        if self.data2 is not None:
            assert np.shape(self.data2) == np.shape(self.data), \
                "data2 must have same shape as data. Right now, data2 has " \
                f"shape {np.shape(self.data2)} vs. {np.shape(self.data)}"
            self.data2 = utils.comply_units(
                value=self.data2,
                default_unit=self.data_unit,
                quantity="data2",
                desired_unit=convert_data_to,
                equivalencies=self.Jy_to_K,
            )
            if np.any(self.data2.imag):
                warnings.warn(
                    "Provided data2 is complex; taking the real part...",
                    stacklevel=2,
                )
                self.data2 = self.data2.real

        # checks on PSF
        if PSF is not None:
            self.PSF = np.copy(PSF)
            if np.shape(self.PSF) != np.shape(self.data):
                raise ValueError(
                    "PSF must have same shape as data. Right now, "
                    f"PSF has shape {np.shape(self.PSF)} "
                    f"vs. {np.shape(self.data)}"
                    )
            if hasattr(self.PSF, "unit"):
                warnings.warn('Assuming PSF has unit 1.',
                              stacklevel=2)
                self.PSF = self.PSF.value
            # normalise
            dz = self.dRpara_dnu * self.delta_freq
            Lz = dz * self.freq_npix
            rpar = np.arange(0., Lz, step=dz)  # Mpc
            fft_psf = self.take_ft(self.PSF, axes=(0, 1))
            # PS normalisation accounting for PSF
            self.norm_map = np.trapz(np.abs(fft_psf)**2, rpar) / Lz  # unit 1
            # self.PSF /= np.sqrt(self.norm_map[..., None])
            if not np.allclose(self.taper, 1.):
                raise NotImplementedError('PSF-normalisation with tapering '
                                          'not properly implemented yet')
        else:
            self.PSF = None
            self.norm_map = np.ones((self.x_npix, self.y_npix))

    # If we anticipate ever updating the observing parameters of a class
    # instance, then we'll need to change these to properties instead of
    # cached properties (and also change some other things).
    @cached_property
    def dRperp_dtheta(self):
        """Conversion from radians to cMpc."""
        return self.cosmo.comoving_distance(self.z).to(units.Mpc).value

    @cached_property
    def dRperp_dOmega(self):
        """Conversion from sr to cMpc^2."""
        return self.dRperp_dtheta ** 2

    @cached_property
    def dRpara_dnu(self):
        """Conversion from Hz to cMpc."""
        return (
            constants.c * (1 + self.z)**2
            / (self.cosmo.H(self.z) * self.rest_freq)
        ).to("Mpc").value

    @cached_property
    def X2Y(self):
        """Conversion from image cube volume to cosmological volume."""
        return self.dRperp_dOmega * self.dRpara_dnu

    @cached_property
    def cosmo_volume(self):
        """Full cosmological volume of the image cube in cMpc^3."""
        # If we update this to allow for varying pixel sizes, then this will
        # need to be changed to an integral.
        n_pix = self.x_npix * self.y_npix * self.freq_npix
        return n_pix * self.volume_element

    @cached_property
    def volume_element(self):
        return (
            self.delta_thetax * self.delta_thetay * self.delta_freq * self.X2Y
        )

    def take_ft(self, data, axes=None):
        """
        Computes the FT of input data [no unit].

        Parameters
        ----------
            data: array of floats
                Data array to take the FT of.
            axes: list/array of ints
                Axes to perform the FT along.
                Default is all.

        Returns
        -------
            fft_data: array of complex noumbers
                FT of data array [no unit].
        """
        if data.ndim != 3:
            raise ValueError('Wrong data shape. Must be dim 3.')

        if axes is None:
            axes = (0, 1, 2)

        fft_data = np.fft.fftshift(
            np.fft.fftn(
                np.fft.ifftshift(data),
                axes=axes
            )
        )

        return fft_data

    def take_ift(self, ft_data, axes=None):
        """
        Computes the inverse FT of input ft_data [no unit].

        Here we use the Fourier convention used in cosmology where a
        factor of 1/(2 pi)^3 is in the inverse transform.

        Parameters
        ----------
            ft_data: array of floats
                FT data array to take the iFT of.
            axes: list/array of ints
                Axes to perform the iFT along.
                Default is all.

        Returns
        -------
            data: array of complex noumbers
                iFT of ft_data array [no unit].
        """
        if ft_data.ndim != 3:
            raise ValueError('Wrong ft_data shape. Must be dim 3.')

        if axes is None:
            axes = (0, 1, 2)
        axes = np.array(axes)

        data = np.fft.fftshift(
            np.fft.ifftn(
                np.fft.ifftshift(ft_data),
                axes=axes
            )
        )
        # TODO: add factor of Npix total here for convention.
        data /= ((2*np.pi)**ft_data.ndim)
        return data

    # TODO: change to rFFT to speed up computing times
    def FFT_crossxx(self):
        """
        Computes the FT**2 of data.

        Returns
        -------
            ps_data: array of floats.
                Power spectrum of self.data, in units of ``vis_units``^2 Mpc^3.
                All values are real numbers.
                Unit is self.data_unit **2 * units.Mpc ** 3.
        """
        if self.verbose:
            print('Taking FT of data...')

        fft_data = self.take_ft(self.data * self.taper)
        fft_data *= self.volume_element  # mK Mpc^3

        taper_norm = 1
        if isinstance(self.taper, np.ndarray):
            if self.freq_taper is not None:
                taper_norm *= np.sum(self.freq_taper ** 2) / self.freq_npix
            if self.theta_x_taper is not None:
                taper_norm *= np.sum(self.theta_x_taper ** 2) / self.x_npix
                taper_norm *= np.sum(self.theta_y_taper ** 2) / self.y_npix

        return np.real(
            np.conj(fft_data) * fft_data
        ) / (self.cosmo_volume * taper_norm) / self.norm_map[..., None] \
            * self.pk_unit

    def FFT_crossxy(self):
        """
        Computes the FT of data and data2.

        Returns
        -------
            ps_data: array of floats.
                Power spectrum of input data,
                in units of ``vis_units``^2 Mpc^3.
                All values are real numbers.
        """
        if self.verbose:
            print('Taking FT of data...')

        # check inputs
        if self.data2 is None:
            raise ValueError(
                'Need to define data2 to compute cross-spectrum.'
            )

        # fourier transform along spatial directions
        fft_data1 = self.take_ft(self.data * self.taper, axes=[0, 1])
        fft_data2 = self.take_ft(self.data2 * self.taper, axes=[0, 1])

        # Take FT along frequency axis
        fft_data1 = self.take_ft(fft_data1, axes=[2]) \
            * self.volume_element   # mK Mpc^3
        fft_data2 = self.take_ft(fft_data2, axes=[2]) \
            * self.volume_element   # mK Mpc^3

        taper_norm = 1
        if isinstance(self.taper, np.ndarray):
            if self.freq_taper is not None:
                taper_norm *= np.sum(self.freq_taper ** 2) / self.freq_npix
            if self.theta_x_taper is not None:
                taper_norm *= np.sum(self.theta_x_taper ** 2) / self.x_npix
                taper_norm *= np.sum(self.theta_y_taper ** 2) / self.y_npix

        return np.real(
            np.conj(fft_data1) * fft_data2
        ) / (self.cosmo_volume * taper_norm) / self.norm_map[..., None] \
            * self.pk_unit

    def compute_eta_nu(self):
        """Define observational Fourier axes."""
        # TODO: just make these all class properties and remove this function
        self.u_x = np.fft.fftshift(
            np.fft.fftfreq(self.x_npix, d=self.delta_thetax)
        )  # 1/rad
        self.u_y = np.fft.fftshift(
            np.fft.fftfreq(self.y_npix, d=self.delta_thetay)
        )  # 1/rad
        self.eta = np.fft.fftshift(
            np.fft.fftfreq(self.freq_npix, d=self.delta_freq)
        )  # 1/Hz = s

        self.delta_ux = np.diff(self.u_x).mean()  # 1/rad
        self.delta_uy = np.diff(self.u_y).mean()  # 1/rad
        self.delta_eta = np.diff(self.eta).mean()  # 1/Hz = s

    def _parse_taper_info(
        self,
        freq_taper=None,
        space_taper=None,
        freq_taper_kwargs=None,
        space_taper_kwargs=None,
    ):
        """Calculate the taper for doing the FT given the taper parameters."""
        taper_info = {"freq": {"type": None}, "space": {"type": None}}
        if freq_taper is None and space_taper is None:
            self.taper = 1
            self.taper_info = taper_info
            return

        try:
            from uvtools.dspec import gen_window
        except (ImportError, ModuleNotFoundError):
            warnings.warn(
                "uvtools is not installed, so no taper will be applied. "
                "Please install uvtools if you would like to use a taper."
            )
        try:
            gen_window(freq_taper, 1)
        except ValueError:
            raise ValueError("Wrong freq taper. See uvtools.dspec.gen_window"
                             " for options.")
        try:
            gen_window(space_taper, 1)
        except ValueError:
            raise ValueError("Wrong freq taper. See uvtools.dspec.gen_window"
                             " for options.")

        taper = np.ones((1, 1, 1), dtype=float)
        if freq_taper is not None:
            freq_taper_kwargs = freq_taper_kwargs or {}
            taper_info["freq"]["type"] = freq_taper
            taper_info["freq"].update(freq_taper_kwargs)
            self.freq_taper = gen_window(
                freq_taper, self.freq_npix, **freq_taper_kwargs
            )
            taper = taper * self.freq_taper[None, None, :]

        if space_taper is not None:
            space_taper_kwargs = space_taper_kwargs or {}
            taper_info["space"]["type"] = space_taper
            taper_info["space"].update(space_taper_kwargs)
            self.theta_y_taper = gen_window(
                space_taper, self.y_npix, **space_taper_kwargs
            )
            self.theta_x_taper = gen_window(
                space_taper, self.x_npix, **space_taper_kwargs
            )
            self.sky_taper = self.theta_y_taper[:, None] \
                * self.theta_x_taper[None, :]
            taper = taper * self.sky_taper[..., None]

        self.taper_info = taper_info
        self.taper = taper

    def compute_Ubox(self):
        """
        Compute norm of observational Fourier axes.
        """
        try:
            self.u_y
        except AttributeError:
            self.compute_eta_nu()
        U_box = np.sqrt(self.u_y[:, None] ** 2 + self.u_x**2)  # 1/rad
        return U_box

    def compute_spatial_fourier_modes(self):
        """
        Define cosmological Fourier axes.
        """

        _ = self.compute_Ubox()

        # compute k_par, k_perps
        self.k_par = 2 * np.pi * self.eta / self.dRpara_dnu  # 1 / Mpc
        self.kx = 2 * np.pi * self.u_x / self.dRperp_dtheta  # 1 / Mpc
        self.ky = 2 * np.pi * self.u_y / self.dRperp_dtheta  # 1 / Mpc

        # the delta's are also now z-dependent
        self.delta_kx = 2 * np.pi * self.delta_ux / self.dRperp_dtheta
        self.delta_ky = 2 * np.pi * self.delta_uy / self.dRperp_dtheta
        self.delta_kz = 2 * np.pi * self.delta_eta / self.dRpara_dnu

    def compute_2D_pspec(self, ps_data=None,
                         k_perp_bin=None, k_par_bin=None,
                         nbins_perp=30, nbins_par=30):
        """
        Compute cylindrical power spectrum of self.data.

        Can be used for general cylindrical averaging of whatever
        is fed to ps_data.

        Parameters
        ----------
            ps_data: array of floats (optional)
                3D power spectrum of self.data.
                Can be fed if it has already been computed with
                self.FFT_crossxx.
                Default is None.
            k_perp_bin: array or list of floats
                k-perpendicular bins to use.
                Units should be Mpc-1.
                All values should be positive.
                Default is None.
            k_par_bin: array or list of floats
                k-parallel bins to use.
                Units should be Mpc-1.
                All values should be positive.
                Default is None.
            nbins_perp: int
                Number of bins to use for kperp when building the cylindrical
                power spectrum.
                Default is 30. Set to kperp_bin.size if inconsistent.
            nbins_par: int
                Number of bins to use for kperp when building the cylindrical
                power spectrum.
                Default is 30. Set to k_par_bin.size if inconsistent.
        Returns
        -------
            k_par_bin: array of floats
                k-parallel bins used.
            k_perp_bin: array of floats
                k-perpendicular bins used.
            pspec_2D: 2d array of floats
                Cylindrical power spectrum in units of mK2 Mpc^3.

        """
        if ps_data is None:
            ps_data = self.FFT_crossxx()
        else:
            assert ps_data.shape == self.data.shape, \
                "Shape of ps_data does not match shape of data."
        try:
            ps_data.unit
        except AttributeError:
            warnings.warn(
                f"Assuming ps_data is in {self.pk_unit.to_string()}."
            )
        else:
            ps_data = ps_data.value

        if np.any(ps_data.imag):
            warnings.warn(
                "Provided ps_data is complex; taking the real part...",
                stacklevel=2,
            )
        ps_data = np.real(ps_data)

        self.compute_spatial_fourier_modes()  # 1 / Mpc
        kmag_perp = np.sqrt(self.kx[None, :] ** 2 + self.ky[:, None] ** 2)

        # so what we want to do here is bin each frequency chunk into
        # a 1D vector and then ouput kperp_binned vs. k_par

        # build the kperp bins if not specified
        if k_perp_bin is None:
            kperp_edges = np.histogram_bin_edges(
                kmag_perp.flatten(),
                bins=nbins_perp,
                range=(kmag_perp.min() * 2., kmag_perp.max() / 2.)
                )
            delta_k = np.diff(kperp_edges).mean()
            k_perp_bin = kperp_edges[:nbins_perp] + (0.5 * delta_k)
        else:
            kperp_edges = utils.bin_edges_from_array(k_perp_bin)
            assert np.size(kperp_edges) == np.size(k_perp_bin) + 1
        assert np.size(k_perp_bin) > 0, "Error obtaining kperp bins."
        nbins_perp = np.size(k_perp_bin)
        # magnitude of kperp modes in box in Mpc-1

        if k_par_bin is None:
            k_par_bin = np.linspace(
                2.*self.delta_kz, self.k_par.max()/2., nbins_par
            )
        assert np.size(k_par_bin) > 0, "Error obtaining/reading kpar bins."
        nbins_par = np.size(k_par_bin)
        kpar_edges = utils.bin_edges_from_array(k_par_bin)

        if self.verbose:
            print('Binning data...')
        # histogram in 2D
        # flatten arrays for np.histogram
        kmag_perp_flat = np.reshape(
            kmag_perp,
            kmag_perp.size,
            order='C'
        )
        ps_box_flat = np.reshape(
            ps_data,
            (kmag_perp.size, -1),
            order='C'
        ).flatten()
        # get pixel coordinates in Fourier space
        coords = np.meshgrid(np.abs(self.k_par), kmag_perp_flat)
        coords = [coords[0].flatten(), coords[1].flatten()]
        # histogram the power spectrum box
        ret = binned_statistic_2d(
            x=coords[0],
            y=coords[1],
            values=ps_box_flat,
            bins=[kpar_edges, kperp_edges],
            statistic='mean',
        )
        pspec_2D = ret.statistic
        pspec_2D[np.isnan(pspec_2D)] = 0.0

        return k_par_bin, np.asarray(k_perp_bin), pspec_2D

    # TODO: Change to np.histogram
    # TODO: Check accuracy and equivalence with compute_1d
    def compute_1d_from_2d(self, ps_data=None,
                           kbins=None, nbins=20,
                           nbins_cyl=50):
        """
        Compute spherical power spectrum of self.data from cylindrical one.

        Parameters
        ----------
            ps_data: array of floats (optional)
                3D power spectrum of self.data.
                Can be fed if it has already been computed with
                self.FFT_crossxx.
                Default is None.
            kbins: array or list of floats
                Spherical k-bins to use.
                Units should be Mpc-1.
                All values should be positive.
                Default is None.
            nbins: int
                Number of bins to use when building the spherical power
                spectrum. Set to kbins.size if kbins is fed.
                Default is 30.
            nbins_cyl: int
                Number of cylindrical bins to use when computing the (square)
                cylindrical power spectrum. Increase for precision (can get
                very slow). Minimum of 30 advised.
                Default is 50.
        Returns
        -------
            kbins: array of floats
                Spherical k-bins used, weighted by cell population.
            pspec: array of floats
                Spherical power spectrum in units of mK2 Mpc^3.

        """
        # define kperp and kpar bins making sure to include
        # all modes in the box
        self.compute_spatial_fourier_modes()  # 1 / Mpc
        kmag_perp = np.sqrt(self.kx[None, :] ** 2 + self.ky[:, None] ** 2)
        k_perp_bin = np.linspace(
            kmag_perp.min(),
            kmag_perp.max(),
            num=nbins_cyl,
        )
        k_par_bin = np.linspace(
            self.delta_kz,
            self.k_par.max(),
            num=nbins_cyl,
        )
        k_par_bin, k_perp_bin, pspec_2D = self.compute_2D_pspec(
            ps_data,
            k_par_bin=k_par_bin,
            k_perp_bin=k_perp_bin,
        )
        k_mag = np.sqrt(k_par_bin[:, None]**2 + k_perp_bin[None, :]**2)

        # define the spherical bins and bin edges
        if kbins is None:
            kmin = np.min(k_mag) * 2.
            kmax = np.max(k_mag) / 2.
            bin_edges = np.histogram_bin_edges(
                np.sort(k_mag.flatten()),
                bins=nbins,
                range=(kmin, kmax)
            )
        else:
            dk = np.diff(kbins).mean()
            assert dk > 0
            bin_edges = utils.bin_edges_from_array(kbins)
            assert np.size(bin_edges) == np.size(kbins) + 1
            nbins = kbins.size
        assert np.size(bin_edges) > 1, "Error obtaining kpar bins."

        pspec = np.zeros(len(bin_edges) - 1)
        weighted_k = np.zeros(len(bin_edges) - 1)
        for k in range(len(bin_edges) - 1):
            mask = (bin_edges[k] < k_mag) & (k_mag <= bin_edges[k + 1])
            if mask.any():
                pspec[k] = np.mean(pspec_2D[mask].real)  # [mk^2 Mpc^3]
                weighted_k[k] = np.mean(k_mag[mask])
        # Make sure there are no nans! If there are make them zeros.
        pspec[np.isnan(pspec)] = 0.0
        # Check empty bins
        if np.any(weighted_k == 0.):
            warnings.warn('Some empty k-bins!')

        return weighted_k, pspec

    # TODO: change to np.histogram
    def compute_1D_pspec(self, ps_data=None, kbins=None,
                         dimensionless=False, nbins=30):
        """
        Compute spherical power spectrum of self.data.

        Parameters
        ----------
            ps_data: array of floats (optional)
                3D power spectrum of data, in units of ``vis_units``^2 Mpc^3.
                If not provided, will use power spectrum of ``self.data``.
                Default is None.
            kbins: array or list of floats
                Spherical k-bins to use.
                Units should be Mpc-1.
                All values should be positive.
                Default is None.
            dimensionless: bool
                Whether to scale the output power spectrum by k3/4pi.
                Default is False.
            nbins: int
                Number of bins to use when building the spherical
                power spectrum.
                Default is 30. Set to kbins.size if inconsistent.
        Returns
        -------
            kbins: array of floats
                Spherical k-bins used, weighted by cell population.
            pspec: array of floats
                Spherical power spectrum in units of mK2 Mpc^3.

        """
        if ps_data is None:
            ps_data = self.FFT_crossxx()
        else:
            assert ps_data.shape == self.data.shape, \
                "Shape of ps_data does not match shape of data."
        try:
            ps_data.unit
        except AttributeError:
            warnings.warn(
                f"Assuming ps_data is in {self.pk_unit.to_string()}."
            )
        else:
            ps_data = ps_data.value

        if np.any(ps_data.imag):
            warnings.warn(
                "Provided ps_data is complex; taking the real part...",
                stacklevel=2,
            )
        ps_data = np.real(ps_data)

        self.compute_spatial_fourier_modes()
        kmag_3d = np.sqrt(
            self.kx[None, :, None] ** 2
            + self.ky[:, None, None] ** 2
            + self.k_par[None, None, :] ** 2
        )
        # This is actually really easy since you are collapsing to 1D.
        # You just want to use the check kmag thing to find all
        # the P(k_perp,k_par) that are in that bin and then average them
        # together. Literally just treat it like the old 2D case.

        # define the spherical bins and bin edges
        if kbins is None:
            kmin = np.min(kmag_3d) * 2.
            kmax = np.max(kmag_3d) / 2.
            bin_edges = np.histogram_bin_edges(
                np.sort(kmag_3d.flatten()),
                bins=nbins,
                range=(kmin, kmax)
            )
        else:
            dk = np.diff(kbins).mean()
            assert dk > 0
            bin_edges = utils.bin_edges_from_array(kbins)
            nbins = kbins.size
        assert np.size(bin_edges) > 1, "Error obtaining kpar bins."

        # now the pspec is in (kx,ky,kz) so we want to go through each kz
        # fourier mode and collapse the kxky into 1D to get a 2D pspec
        if self.verbose:
            print('Binning data...')
        pspec = np.zeros(len(bin_edges) - 1)
        weighted_k = np.zeros(len(bin_edges) - 1)
        for k in range(len(bin_edges) - 1):
            mask = (bin_edges[k] < kmag_3d) & (kmag_3d <= bin_edges[k + 1])
            if mask.any():
                pspec[k] = np.mean(ps_data[mask].real)  # [mk^2 Mpc^6]
                weighted_k[k] = np.mean(kmag_3d[mask])

        pspec[np.isnan(pspec)] = 0.0

        if dimensionless:
            pspec *= weighted_k**3 / 4.0 / np.pi  # [mk^2]
        # Check empty bins
        if np.any(weighted_k == 0.):
            warnings.warn('Some empty k-bins!')

        return weighted_k, pspec
