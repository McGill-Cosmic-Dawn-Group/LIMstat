from cached_property import cached_property
import numpy as np
import warnings

from astropy import constants, units, cosmology
# Tell astropy that K is equivalent to Jy/sr
units.set_enabled_equivalencies([(units.K, units.Jy/units.sr), ])


class Power_Spectrum(object):
    """Class containing tools necessary to build a power spectrum.

    This code computes the cylindricla and spherical power spectra from a
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
        rest_freq=1420*units.MHz,
        cosmo=cosmology.Planck18,
        freq_taper=None,
        space_taper=None,
        freq_taper_kwargs=None,
        space_taper_kwargs=None,
        convert_data_to=None,
        beam_area=None,
        verbose=False,
    ):
        """
        Initialisation of the Power_Spectrum class.

        Parameters
        ----------
            data: array of real floats.
                Input intensity mapping data. Data is assumed to be in units
                of mK if not provided as an ``astropy.Quantity``. The expected
                shape is ``(y_npix, x_npix, n_freq)``.
            theta_x: float
                Angular extent of the image cube in the x-direction. Assumed to
                be in radians if not provided as an ``astropy.Quantity``.
            theta_y: float
                Angular extent of the image cube in the y-direction. Assumed to
                be in radians if not provided as an ``astropy.Quantity``.
            freqs: list or array of floats.
                List of frequencies the signal was measured on.
                Frequencies are assumed to be in MHz if not provided as an
                ``astropy.Quantity``.
            rest_freq: float
                Rest-frequency of the emission line used for intensity mapping.
                Defaults to the 21-cm line. Frequency is assumed to be in MHz
                if not provided as an ``astropy.Quantity``.
            cosmo: astropy.cosmology class
                Cosmology to use for computations.
                Default is Planck18.
            freq_taper: str, optional
                Name of the taper to use when Fourier transforming along the
                frequency axis. Must be available in scipy.signal.windows.
            space_taper: str, optional
                Name of the taper to use when Fourier transforming along the
                spatial axes.
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
            verbose: bool
                Whether to output messages when running functions.
        """
        # ensure input shapes are consistent
        if data.shape[-1] != freqs.size:
            raise ValueError(
                "The last axis of the data array must be the frequency axis."
            )

        # check unit and convert if necessary
        self.freqs = comply_units(
            value=freqs,
            default_unit=units.MHz,
            quantity="freqs",
            desired_unit=units.Hz,
        )
        self.rest_freq = comply_units(
            value=rest_freq,
            default_unit=units.MHz,
            quantity="rest_freq",
            desired_unit=units.Hz,
        )

        if beam_area is not None:
            beam_area = comply_units(
                value=beam_area,
                default_unit=units.sr,
                quantity="beam_area",
                desired_unit=units.sr,
            ) * units.sr
        self.beam_area = getattr(beam_area, "value", beam_area)

        self.Jy_to_K = units.brightness_temperature(
            self.freqs*units.Hz, beam_area
        )

        if convert_data_to is None and hasattr(data, "unit"):
            self.data_unit = data.unit
        else:
            self.data_unit = convert_data_to or units.mK

        self.data = comply_units(
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
        self.theta_x = comply_units(
            value=theta_x,
            default_unit=units.rad,
            quantity="theta_x",
            desired_unit=units.rad,
        )
        self.theta_y = comply_units(
            value=theta_y,
            default_unit=units.rad,
            quantity="theta_y",
            desired_unit=units.rad,
        )

        # Take note of the shape of the data.
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
            constants.c * (1 + self.z)**2 / self.cosmo.H(self.z)
            / self.rest_freq
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

    def cosmo_FFT3(self):
        """
        Computes the Fourier transform of input data.

        Returns
        -------
            ps_data: array of floats.
                Power spectrum of input data,
                in units of ``vis_units``^2 Mpc^3.
                All values are real numbers.
        """
        if self.verbose:
            print('Taking FT of data...')
        # "observer" fourier transform
        fft_data = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(self.data * self.taper))
        ) * self.volume_element  # mK Mpc^3

        taper_norm = 1
        if isinstance(self.taper, np.ndarray):
            if self.freq_taper is not None:
                taper_norm *= np.sum(self.freq_taper ** 2) / self.freq_npix
            if self.theta_x_taper is not None:
                taper_norm *= np.sum(self.theta_x_taper ** 2) / self.x_npix
                taper_norm *= np.sum(self.theta_y_taper ** 2) / self.y_npix
        return np.real(
            np.conj(fft_data) * fft_data
        ) / (self.cosmo_volume * taper_norm)  # mK^2 Mpc^3

    def cosmo_FFT3_crossxy(self, data1, data2, PSF1=None, PSF2=None):
        """
        Computes the FT of 2 input data sets and deconvolves PSF if provided.

        Returns
        -------
            ps_data: array of floats.
                Power spectrum of input data,
                in units of ``vis_units``^2 Mpc^3.
                All values are real numbers.
        """
        if self.verbose:
            print('Taking FT of data...')
        # "observer" fourier transform
        fft_data1 = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(data1 * self.taper), axes=[0, 1])
        ) * self.volume_element  # mK Mpc^3

        fft_data2 = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(data2 * self.taper), axes=[0, 1])
        ) * self.volume_element

        if (PSF1 is not None):
            if (PSF2 is None):
                PSF2 = np.copy(PSF1)
                warnings.warn('Using same PSF for both data sets.',
                              stacklevel=2)
            if self.verbose:
                print('Deconvolving PSF...')
            fft_psf1 = np.fft.fftshift(
                np.fft.fftn(
                    np.fft.ifftshift(PSF1),
                    axes=[0, 1]
                )
            ) * self.volume_element

            fft_psf2 = np.fft.fftshift(
                np.fft.fftn(
                    np.fft.ifftshift(PSF2),
                    axes=[0, 1]
                )
            ) * self.volume_element

            idx1 = np.argwhere(fft_psf1 == 0)
            fft_psf1[idx1] = 1

            idx2 = np.argwhere(fft_psf2 == 0)
            fft_psf2[idx2] = 1

            fft_data1 /= fft_psf1
            fft_data2 /= fft_psf2

        fft_data1 = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(fft_data1 * self.taper), axes=[2])
        ) * self.volume_element  # mK Mpc^3

        fft_data2 = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(fft_data2 * self.taper), axes=[2])
        ) * self.volume_element

        taper_norm = 1
        if isinstance(self.taper, np.ndarray):
            if self.freq_taper is not None:
                taper_norm *= np.sum(self.freq_taper ** 2) / self.freq_npix
            if self.theta_x_taper is not None:
                taper_norm *= np.sum(self.theta_x_taper ** 2) / self.x_npix
                taper_norm *= np.sum(self.theta_y_taper ** 2) / self.y_npix
        return np.real(
            np.conj(fft_data1) * fft_data2
        ) / (self.cosmo_volume * taper_norm)  # mK^2 Mpc^3

    def cosmo_FFT3_cross(self, data1, data2, PSF1=None, PSF2=None):
        """
        Computes the Fourier transform of input data.

        Returns
        -------
            ps_data: array of floats.
                Power spectrum of input data,
                in units of ``vis_units``^2 Mpc^3.
                All values are real numbers.
        """
        if self.verbose:
            print('Taking FT of data...')
        # "observer" fourier transform
        fft_data1 = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(data1 * self.taper))
        ) * self.volume_element  # mK Mpc^3

        fft_data2 = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(data2 * self.taper))
        ) * self.volume_element

        if PSF1 is not None:
            print('deconvolving PSF')
            fft_psf1 = np.fft.fftshift(
                np.fft.fftn(
                    np.fft.ifftshift(PSF1)
                )
            )  # * self.volume_element

            fft_psf2 = np.fft.fftshift(
                np.fft.fftn(
                    np.fft.ifftshift(PSF2)
                )
            ) * self.volume_element

            idx1 = np.argwhere(fft_psf1 == 0)
            fft_psf1[idx1] = 1

            idx2 = np.argwhere(fft_psf2 == 0)
            fft_psf2[idx2] = 1

            fft_data1 /= fft_psf1
            fft_data2 /= fft_psf2

        else:
            pass

        taper_norm = 1

        if isinstance(self.taper, np.ndarray):
            if self.freq_taper is not None:
                taper_norm *= np.sum(self.freq_taper ** 2) / self.freq_npix
            if self.theta_x_taper is not None:
                taper_norm *= np.sum(self.theta_x_taper ** 2) / self.x_npix
                taper_norm *= np.sum(self.theta_y_taper ** 2) / self.y_npix
        return np.real(
            np.conj(fft_data1) * fft_data2
        ) / (self.cosmo_volume * taper_norm)  # mK^2 Mpc^3

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
        space_taper_kwargs=None
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
            import warnings
            warnings.warn(
                "uvtools is not installed, so no taper will be applied. "
                "Please install uvtools if you would like to use a taper."
            )

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
            self.sky_taper = self.theta_y_taper[: None] \
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

    # TODO: make this method name better match what it does.
    def compute_kperp_kpar(self):
        """
        Define cosmological Fourier axes.

        Returns
        -------
            k_perp_mag: array of floats
                List of norm of kperp vectors in the box.
        """

        U_box = self.compute_Ubox()

        # compute k_par, k_perps
        self.k_par = 2 * np.pi * self.eta / self.dRpara_dnu  # 1 / Mpc
        self.kx = 2 * np.pi * self.u_x / self.dRperp_dtheta  # 1 / Mpc
        self.ky = 2 * np.pi * self.u_y / self.dRperp_dtheta  # 1 / Mpc

        # the delta's are also now z-dependent
        self.delta_kx = 2 * np.pi * self.delta_ux / self.dRperp_dtheta
        self.delta_ky = 2 * np.pi * self.delta_uy / self.dRperp_dtheta
        self.delta_kz = 2 * np.pi * self.delta_eta / self.dRpara_dnu

        k_perp_mag = (
            2 * np.pi * np.reshape(U_box, U_box.size) / self.dRperp_dtheta
        )  # 1 / Mpc

        return k_perp_mag

    def compute_2D_pspec(self, ps_data=None,
                         k_perp_bin=None, k_par_bin=None,
                         nbins_perp=30, nbins_par=30):
        """
        Compute cylindrical power spectrum of self.data.

        Parameters
        ----------
            ps_data: array of floats (optional)
                3D power spectrum of self.data.
                Can be fed if it has already been computed with
                self.cosmo_FFT3.
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
            ps_data = self.cosmo_FFT3() * self.data_unit ** 2
        else:
            assert ps_data.shape == self.data.shape, \
                "Shape of ps_data does not match shape of data."
        try:
            ps_data.unit
        except AttributeError:
            warnings.warn(f"Assuming ps_data is in {self.data_unit}**2.")
        else:
            ps_data = ps_data.to(self.data_unit**2).value
        ps_data = np.real(ps_data)

        k_perp_mag = self.compute_kperp_kpar()  # 1 / Mpc

        # so what we want to do here is bin each frequency chunk into
        # a 1D vector and then ouput kperp_binned vs. k_par

        # build the kperp bins if not specified
        if k_perp_bin is None:
            kperp_edges = np.histogram_bin_edges(
                k_perp_mag,
                bins=nbins_perp,
                range=(k_perp_mag.min() * 2., k_perp_mag.max() / 2.)
                )
            delta_k = np.diff(kperp_edges).mean()
            k_perp_bin = kperp_edges[:nbins_perp] + (0.5 * delta_k)
        else:
            kperp_edges = bin_edges_from_array(k_perp_bin)
            assert np.size(kperp_edges) == np.size(k_perp_bin) + 1
        assert np.size(k_perp_bin) > 0, "Error obtaining kperp bins."
        nbins_perp = np.size(k_perp_bin)
        # magnitude of kperp modes in box
        kmag_perp = np.sqrt(self.kx[None, :] ** 2 + self.ky[:, None] ** 2)  # 1 / Mpc

        if k_par_bin is None:
            k_par_bin = np.linspace(
                2.*self.delta_kz, self.k_par.max()/2., nbins_par
            )
        assert np.size(k_par_bin) > 0, "Error obtaining/reading kpar bins."
        nbins_par = np.size(k_par_bin)
        kpar_edges = bin_edges_from_array(k_par_bin)

        if self.verbose:
            print('Binning data...')
        pspec_2D = np.zeros((nbins_par, nbins_perp))
        # now the pspec is in (kx,ky,kz) so we want to go through each kz
        # fourier mode and collapse the kxky into 1D to get a 2D pspec
        for i in range(k_par_bin.size):
            mask_par = (kpar_edges[i] < np.abs(self.k_par)) & \
                (np.abs(self.k_par) <= kpar_edges[i + 1])
            if not mask_par.any():
                continue
            for j in range(len(kperp_edges) - 1):
                mask_perp = (kperp_edges[j] < kmag_perp) & \
                    (kmag_perp <= kperp_edges[j + 1])
                if mask_perp.any():
                    pspec_2D[i, j] = np.mean(ps_data[mask_perp][:, mask_par])  # [mK^2 Mpc^6]

        pspec_2D[np.isnan(pspec_2D)] = 0.0

        return k_par_bin, np.asarray(k_perp_bin), pspec_2D

    def compute_kmag(self):
        """
        Compute norm of cosmological Fourier modes in box.

        Returns
        -------
            kmag: list of floats
                List of norms of cosmological Fourier modes
                in box. Size is self.data.size.
        """
        k_perp_mag = self.compute_kperp_kpar()  # 1 / Mpc
        kmag = np.sqrt(k_perp_mag[None, :] ** 2 + self.k_par[:, None] ** 2)  # 1 / Mpc

        return np.reshape(kmag, kmag.size)

    def compute_1d_from_2d(self, ps_data=None,
                           kbins=None, nbins=30,
                           nbins_cyl=50):
        """
        Compute spherical power spectrum of self.data from cylindrical one.

        Parameters
        ----------
            ps_data: array of floats (optional)
                3D power spectrum of self.data.
                Can be fed if it has already been computed with
                self.cosmo_FFT3.
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
                cylindrical power spectrum. Increase for precision
                (can get very slow). Minimum of 30 advised.
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
        k_perp_mag = self.compute_kperp_kpar()
        k_perp_bin = np.linspace(
            k_perp_mag.min(),
            k_perp_mag.max(),
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
            bin_edges = bin_edges_from_array(kbins)
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
                Number of bins to use when building
                the spherical power spectrum.
                Default is 30. Set to kbins.size if inconsistent.
        Returns
        -------
            kbins: array of floats
                Spherical k-bins used, weighted by cell population.
            pspec: array of floats
                Spherical power spectrum in units of mK2 Mpc^3.

        """
        if ps_data is None:
            ps_data = self.cosmo_FFT3() * self.data_unit ** 2
        else:
            assert ps_data.shape == self.data.shape, \
                "Shape of ps_data does not match shape of data."
        try:
            ps_data.unit
        except AttributeError:
            warnings.warn(f"Assuming ps_data is in {self.data_unit}**2.")
        else:
            ps_data = ps_data.to(self.data_unit**2).value

        _ = self.compute_kperp_kpar()
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
            bin_edges = bin_edges_from_array(kbins)
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
            pspec *= weighted_k**3 * pspec / 4.0 / np.pi  # [mk^2]
        # Check empty bins
        if np.any(weighted_k == 0.):
            warnings.warn('Some empty k-bins!')

        return weighted_k, pspec

    def compute_dimensionless_1D_pspec(self):
        """
        Compute dimensionless spherical power spectrum.
        """
        warnings.warn('Has been replaced by an option in compute_1D_pspec.')
        return
        # k_modes, ps_1d = self.compute_1D_pspec()
        # dimensionless_pspec = k_modes**3 * ps_1d / 4.0 / np.pi  # [mk^2]
        # return dimensionless_pspec


def convert_units(signal, freq, FWHM, from_to='bt_i'):
    '''Helper function which is basically just a wrapper
    of the astropy equivalencies for brightness temperature.

    If the desired conversion involves Jy/beam_area then you must
    specify a beam, otherwise you could ask for Jy/str.


    Parameters:
            signal (array): signal, must have astropy units of
                            either brightness_temperature (e.g. mK)
                            or intensity (e.g. Jy/str, Jy/beam_area)
            freq (float): The frequency of the observed signal,
                            must have astropy units of frequency
            FWHM (float): The FWHM of the beam. This is then converted
                            to sigma of the beam to calculate the beam area
                            must have astropy units of angular extent
                            (e.g. degrees, arcsec)

    Returns:
            binary_sum (arr): Converted form of 'signal' parameter in
                                new desired astropy units
    '''

    beam_sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    beam_area = 2*np.pi*(beam_sigma)**2
    equiv = units.brightness_temperature(freq)

    if from_to == 'bt_i':
        signal_convert = (signal).to(units.Jy/beam_area, equivalencies=equiv)

    if from_to == 'i_bt':
        signal_convert = (signal).to(units.mK, equivalencies=equiv)

    return signal_convert


def bin_edges_from_array(arr):
    """
    Obtain bin edges from existing bin-array.

    Parameters
    ----------
        arr: 1d array
            List of bin centres.
    Returns
    -------
        arr: 1d array
            List of bin edges.
            Size is arr.size + 1.
    """
    # checks on inputs
    assert np.ndim(arr) == 1. and np.size(arr) > 1, \
        "arr must be a list of values."

    dx = np.diff(arr).mean()
    assert dx > 0, "arr must be sorted in increasing order."
    assert np.allclose(np.diff(arr), dx), \
        "arr must be regularly spaced."

    bin_edges = np.arange(
        arr.min()-dx/2,
        arr.max()+dx,
        step=dx)
    assert np.size(bin_edges) == np.size(arr) + 1

    return bin_edges


def check_unit(value, unit, name):

    """
    Check if input has correct unit.

    Parameters
    ----------
        value: object
            Input to check the unit for.
        unit: astropy.units object or str
            Required unit for input.
        name: str
            Name of the input (for error message).
    Returns
    -------
        value: object
            Input converted to appropriate unit if necessary.
    """

    try:
        value.unit
    except AttributeError:
        warnings.warn(f'Assuming {name} given in {unit}...')
        value *= unit

    try:
        value = value.to(unit)
    except units.UnitConversionError:
        raise ValueError(f'Unit of {name} is not compatible with '
                         f'required {unit}... Aborting.')
    else:
        value = value.to(unit)
    return value


def comply_units(
        value,
        default_unit,
        quantity,
        desired_unit=None,
        equivalencies=None
        ):
    """
    Check the units of the input and optionally convert to another type.

    Parameters
    ----------
        value: np.ndarray, astropy.Quantity, or float
            Quantity to perform unit-checking (and optional conversion) on.
        default_unit: astropy unit
            Unit to use in case ``value`` is not an ``astropy.Quantity``.
        quantity: str
            Name of the quantity being unit-checked, for logging purposes.
        desired_unit: astropy unit, optional
            If provided, specifies the unit that ``value``
            should be converted to.
        equivalencies: list, optional
            List of equivalencies for converting between ``value.unit`` (or
            ``default_unit``) and ``desired_unit``.

    Returns
    -------
        checked_value: np.ndarray or float
            Provided quantity in the desired units.
    """
    if not hasattr(value, "unit"):
        msg = f"Provided {quantity} does not have units. "
        msg += f"Assuming the input is provided in {str(default_unit)}."
        warnings.warn(msg, stacklevel=2)
        value = value * default_unit

    if not value.unit.is_equivalent(
        default_unit, equivalencies=equivalencies
    ):
        raise ValueError(f"{quantity} does not have the right units!")

    if desired_unit is not None:
        if not desired_unit.is_equivalent(
            value.unit, equivalencies=equivalencies
        ):
            raise ValueError(
                f"{quantity} cannot be converted to {str(desired_unit)}."
            )
        equivalencies = equivalencies or []
        value = value.to(desired_unit, equivalencies=equivalencies)

    return value.value
