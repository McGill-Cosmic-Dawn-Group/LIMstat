import healpy as hp
import numpy as np
import warnings
from astropy import units, cosmology, constants
from scipy.integrate import quad
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.stats import binned_statistic_2d
from deprecated import deprecated
import pickle
import matplotlib.pyplot as plt
from . import utils

import time

try:
	import numba
	HAVE_NUMBA = True
except ModuleNotFoundError:
	HAVE_NUMBA = False


class cosmological_signal(object):
	"""Generate observation box for given power spectrum.

		Parameters
		----------
			ps: list/array of floats or function
				Power spectrum of the cosmological signal.
			cosmo_units: instance of the cosmo_units class
				Contains information about the survey volume and resolution in units of Mpc.
			verbose: boolean
				Whether to print status reports.
				Default is False.
	"""

	def __init__(
		self,
		ps,
		cosmo_units,
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
			self.ps = interp1d(k_theory, p_theory, fill_value="extrapolate")
		elif callable(ps):
			self.ps = ps  # [mk^2*Mpc^3]
		else:
			raise ValueError('ps must be an array or a callable.')

		self.volume = cosmo_units.cosmo_volume.value

		self.delta_y = cosmo_units.delta_y.value # sampling rate y
		self.delta_x = cosmo_units.delta_x.value  # sampling rate x
		self.delta_z = cosmo_units.delta_z.value # sampling rate z

		self.delta_ky = cosmo_units.delta_ky.value
		self.delta_kx = cosmo_units.delta_kx.value
		self.delta_kz = cosmo_units.delta_k_par.value

		self.y_npix = cosmo_units.y_npix
		self.x_npix = cosmo_units.x_npix
		self.z_npix = cosmo_units.z_npix

		self.verbose = verbose

	def compute_kbox(self):
		"""
		Outputs the norms of the Fourier modes in the box.

		Returns
		-------
			kbox: array of real floats
				Array of norms of Fourier modes included in
				simulated box. Shape is (npix, npix, nfreqs).
		"""

		kx = (2. * np.pi) * np.fft.fftshift(
			np.fft.fftfreq(self.x_npix, d=self.delta_x)
		)
		ky = (2. * np.pi) * np.fft.fftshift(
			np.fft.fftfreq(self.y_npix, d=self.delta_y)
		)
		kz = (2. * np.pi) * np.fft.fftshift(
			np.fft.fftfreq(self.z_npix, d=self.delta_z)
		)
		kbox = np.sqrt(
			kx[:, None, None] ** 2
			+ ky[None, :, None] ** 2
			+ kz[None, None, :] ** 2
		)

		return kbox

	def sigma_box(self,r, **ps_kwargs):
		"""
		Generate a function that 
		returns the decorrelation function sigma for a given k and r(k).'''
		
		Returns
		-------
		sigma_box: array of floats
			Array of decorrelation stdev, sigma.
		"""
		# if r(k) = 0 then sigma = 100*np.pi (arbitary large number) 
		# if r(k) = 1 then sigma = 0
		# else sigma = sqrt(log(r(k))**(-2))

		self.make_universe(**ps_kwargs)

		rbox = np.where(self.kbox != 0, r(self.kbox), 0.) 

		# Make sure that r has values between -1 and 2
		if np.any(np.abs(rbox) > 1) == True:
			raise ValueError('r(k) must be between -1 and 1.')


		sigma_box = np.zeros_like(rbox)
		sigma_box = np.where(rbox == 0, 100*np.pi, 0.)

		i,j,k = np.where((rbox != 0) & (np.abs(rbox) != 1))
		sigma_box[i,j,k] =  np.sqrt(np.log(np.abs(rbox[i,j,k])**(-2)))

		return sigma_box
	
	def make_correlated_universes(self, r, ps_2 = None, **ps_kwargs):
		sigma_box = self.sigma_box(r, **ps_kwargs)
		
		sign_box =  np.where(self.kbox != 0, np.sign(r(self.kbox)), 0.)
		phi_box = np.random.normal(0,sigma_box)


		if ps_2 is not None:
			# Check PS input
			if isinstance(ps_2, np.ndarray) or isinstance(ps_2, list):
				# here you interpolate
				assert np.shape(ps_2)[1] > 2, \
					"Need at least 3 values to interpolate power spectrum. " \
					f"Shape is {np.shape(ps_2)}, should be (2, nk)."
				k_theory = ps_2[0]
				p_theory = ps_2[1] 
				self.ps_2 = interp1d(k_theory, p_theory, fill_value="extrapolate")
			elif callable(ps_2):
				self.ps_2 = ps_2  # [mk^2*Mpc^3]
			else:
				raise ValueError('ps must be an array or a callable.')

			#if either spectra are 0 in the box, raise error
			ps2_box = np.where(self.kbox != 0, self.ps_2(self.kbox, **ps_kwargs), 0.)  # mk2 Mpc3
			ps_box = np.where(self.kbox != 0, self.ps(self.kbox, **ps_kwargs), 0.)  # mk2 Mpc3
			
			ratio = ps2_box/ps_box
			#if ratio nan set to 0
			ratio = np.where(np.isnan(ratio), 0, ratio)

			f_k_box = np.sqrt(ratio)

			self.T_tilde_2 = f_k_box * sign_box * np.exp(1j*phi_box) * self.T_tilde    


		else:
			if self.verbose:
				print("Generating 2nd box with same PS as 1st box")
			
			self.T_tilde_2 = sign_box * np.exp(1j*phi_box) * self.T_tilde    
		
		
		if self.verbose:
			print("Taking iFFT of 2nd box")

		self.u2 = np.fft.fftshift(
			np.fft.irfftn(np.fft.ifftshift((self.T_tilde_2)*self.dk),
						  s=(self.kbox.shape),
						  norm='forward'))  # [mK]
		self.u2 /= (2.*np.pi)**3

		if self.verbose:
			print(f'Mean of simulated box = {np.mean(self.u2.real):.3e}')

		return self.u.real, self.u2.real

	def make_universe(self, **ps_kwargs):
		"""
		Fill box with GRF with std PS/2.
		"""

		# this box will hold all of the values of k for each pixel
		self.kbox = self.compute_kbox()  # Mpc-1
		# store ps values along kbox
		powerbox = np.where(self.kbox != 0, self.ps(self.kbox, **ps_kwargs), 0.)  # mk2 Mpc3
		powerbox *= self.volume  # Mpc3 * mk2 Mpc3 = mk2 mpc6
		
		# here's the box that will hold the random gaussian things
		means = np.zeros(self.kbox.shape)
		widths = np.sqrt(powerbox*0.5)  # sqrt(mk2 mpc6)= mk mpc3
		a, b = np.random.normal(
			means,
			widths,
			size=(2, self.y_npix, self.x_npix, self.z_npix)
		)  # Mpc3

		self.T_tilde = a + b * 1j

		if self.verbose:
			print("Taking iFFT of box")
		self.dk = (self.delta_kx*self.delta_ky*self.delta_kz)
		self.u = np.fft.fftshift(
			np.fft.irfftn(np.fft.ifftshift((self.T_tilde)*self.dk),
						  s=(self.kbox.shape),
						   norm='forward'
						  )
						  )  # [mK]
		self.u /= (2.*np.pi)**3

		if self.verbose:
			print(f'Mean of simulated box = {np.mean(self.u.real):.3e}')

		return self.u.real


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
		Generate diffuse foregrounds and/or point sources for given FoV for 21cm observations.

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
		self.nfreqs = self.freqs.size
		self.mean_freq = np.mean(self.freqs)
		self.freq_res = np.diff(freqs).mean()

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
			GLEAM, based on Franzen+ 2018: https://arxiv.org/pdf/1812.00666.pdf)
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
			diffuse_foregounds += self.generate_point_sources(**kwargs)

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

	
	
	@deprecated(version='1.0', reason="This function has been replaced by the generate point sources method. Simply set the flux bounds lower to generate psource background.")
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


class CO_interlopers:  
	""" Class to generate CO interloper foregrounds for sub-mm observations."""

	def __init__(
		self,
		freqs,
		cosmo_units,
		ang_res,
		npix, 
		verbose = True):

		"""Container for CO interloper foregrounds generator.

		Parameters
		----------
			cosmo_units: astropy.cosmology
				Instance of the cosmo_units class. 
			freqs: list or array of floats
				Frequencies along the spectral window considered.
				Must be given with units or Hz are assumed.
			ang_res: float
				Angular resolution of the sim box.
		"""

		
		self.freqs = utils.check_unit(freqs, units.Hz, 'frequencies')
		self.delta_nu = np.abs(self.freqs[1] - self.freqs[0])
		self.cu = cosmo_units
		self.ang_res  = ang_res.to(units.rad)
		self.npix   = npix
		self.verbose = verbose
		self.co_restfreqs = {'CO 1-0': 115.27 * units.GHz,
		   'CO 2-1':230.54 * units.GHz,
		   'CO 3-2': 345.80 *units.GHz,
			'CO 4-3': 416.04 * units.GHz,
			'CO 5-4': 576.27 * units.GHz,   
			'CO 6-5': 691.47 * units.GHz,
		}
			# 'CO 7-6': 806.65 * units.GHz,
			# 'CO 8-7': 921.80 * units.GHz,
			# 'CO 9-8': 1036.91 * units.GHz,}
		if self.verbose:
			print('initializing CO interloper foregrounds generator...')

	def find_line_redshifts(self):
		""" Here we want to find which lines are seen in each frequency channel given the lines'
		 rest frequency and find the corresponding emission redshifts."""

		self.observed_line_redsfhits = {}

		for line in self.co_restfreqs.keys():
			rest_freq = self.co_restfreqs[line].to(units.Hz)
			z_emit = (rest_freq/self.freqs) -1
			self.observed_line_redsfhits[line] = z_emit
	

	
	def generate_intensity(self):

		self.find_line_redshifts()

		# Get the luminosity function parameters for this line
		# Load data (deserialize)
		with open('/Users/hannahfronenberg/Documents/GitHub/SKA_SDC3/ska_simulator/ska_simulator/CO_interlopers/CO_Schecter_Params.pkl', 'rb') as handle:
			Schecter_Params = pickle.load(handle)

		# DefineL/L* bins in log space
		log_ll_star = np.linspace(-3,1,100) 
		delta_log_llstar = log_ll_star[1]-log_ll_star[0]

		#For each line, draw an average number of sources per lum bin at each redshift
		ave_Ns = {} 

		for line in self.co_restfreqs.keys():
			z = self.observed_line_redsfhits[line]

			#Get the luminosity function parameters for this line
			alpha = Schecter_Params[line]['alpha'](z)
			phi_star = np.where( z > 0 , 10**(Schecter_Params[line]['phi'](z)),0.)
			# generate Schecter function for this line at every redshift
			phi = np.log(10) * phi_star[np.newaxis,:] * np.exp(-(10**(log_ll_star[:,np.newaxis])))* ((10**log_ll_star[:,np.newaxis])**(alpha[np.newaxis,:] + 1)) * delta_log_llstar # [Mpc -3] 
			ave_Ns[line] = phi  * self.cu.volume_element.value #[Mpc^3 Mpc^-3]

		
		#Now we want to draw a luminosity for each source in each bin
		lum_box = np.zeros(((self.npix*self.npix), len(self.freqs)))
		
		for line in self.co_restfreqs.keys():

			if self.verbose:
				print(line)

			z = self.observed_line_redsfhits[line]
			start_time = time.time()
			random_samples = np.random.poisson(ave_Ns[line], (self.cu.x_npix*self.cu.y_npix,len(log_ll_star),len(z)))
			end_time = time.time()
		
			log_L_star = Schecter_Params[line]['L'](z)
			start_time = time.time()
			lum = 10**(log_ll_star[np.newaxis,:,np.newaxis] + log_L_star[np.newaxis,np.newaxis,:]) * random_samples
			

			lum = np.sum(lum, axis=1) #[Jy km s^-1 Mpc^2]

			#now unit conversion to [Jy/sr]
			D_l = self.cu.cosmo.luminosity_distance(z)
			conversion = self.co_restfreqs[line].to('Hz').value /(4*np.pi*(D_l.value**2) *3e5 * self.delta_nu.value * self.ang_res.value**2) #to Jy/sr
			lum *= conversion[np.newaxis,:]# [Jy/sr]
			lum = lum*units.Jy/units.sr 
			lum = lum.to(units.uK, equivalencies=units.brightness_temperature(self.freqs)) #convert to uK

			lum_box += lum.value
		lum_box = lum_box.reshape((self.cu.x_npix,self.cu.y_npix,len(self.freqs)))
		
		return lum_box * units.uK
	
	#maybe add a function to generate an intensity cube for one line of interest

	

class thermal_noise:

	def __init__(
		self,
		npix,
		freqs,
		nb_antennas,
		total_obs_time,
		collecting_area,
		integration_time=10.*units.s,
		uv_map=None,
		Tsys=None,
		output_unit=units.Jy/units.beam,
		beam_area=None,
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
			uv_map: array
				uv map calculated from the distribution of antennas
				and observing time.
				Default is None. Must provide a uv map to make noise box.
			Tsys: float or array of floats
				System temperature in K. Can be either one value, then Tsys
				will be constant with frequency, or an array of nfreqs values.
				Default is None: using Haslam+2008 foreground temperature.
			output_unit: astropy unit, optional
				What units the noise simulation should be in.
				Default is Jy/sr.
			beam_area: astropy Quantity, optional
				Integral of the peak-normalized synthesized beam.

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
			self.Tsys = 180. * units.K \
				* (freqs.to(units.MHz).value / 180.)**(-2.6)
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

		# assigning the uv map
		self.uv_map = uv_map

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
			2. * constants.k_B.si * self.Tsys
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


	def make_noise_box(self, uv_map=None):

		# check for uv map and define the variable
		if (uv_map is None) and (self.uv_map is None):
			warnings.warn('No uv map provided. Setting uv_map to 1s.')
			uvmap = np.ones((self.npix, self.npix, self.nfreqs))
		elif uv_map is None:
			uvmap = self.uv_map
		else:
			uvmap = uv_map

		# get std in Jy/beam to avoid losing frequency dependence
		std_Jy = self.std.to(units.Jy/units.beam, 
									equivalencies=self.Jy_to_K)
		
		# simulate real component of noise
		noise_real = np.random.normal(
			0,
			std_Jy.value/np.sqrt(2),
			size=(self.npix, self.npix, self.nfreqs))
		
		# simulate imaginary component of noise
		noise_imag = np.random.normal(
			0,
			std_Jy.value/np.sqrt(2),
			size=(self.npix, self.npix, self.nfreqs))
		
		# making complex noise variable
		noise_ft = noise_real + 1j*noise_imag
		
		# check the dimensions
		assert uvmap.shape == noise_ft.shape, \
			'Your uv map has the wrong shape. It should should have the shape ({}, {}, {}).'.format(self.npix, 
																									self.npix, 
																									self.nfreqs)
		# apply the uv map
		# TO DO -- what happens at points where uv_map = 0?
		noise_ft = np.divide(noise_ft, np.sqrt(uvmap), where=uvmap != 0.)
		noise_ft[uvmap==0.] = 0.

		# inverse fourier transform
		noise = np.fft.fftshift(
					np.fft.ifftn(
						np.fft.ifftshift(noise_ft),
						norm='forward',  # forward removes any normalization in ifft
						axes=(0, 1)
					)
				).real
		
		# normalization
		noise /= np.sqrt(noise.size)
		noise *= std_Jy.unit
		
		return noise.to(self.noise_unit, equivalencies=self.Jy_to_K)


def _generate_confusion_noise(*args, **kwargs):
	# no-op in case we don't have numba
	pass

# is this sopposed to be nested in generate_confusion_noise?
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
