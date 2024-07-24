from cached_property import cached_property


import numpy as np
from tqdm import tqdm
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter1d

from astropy import constants, units, cosmology
import warnings
from . import utils


class window_function(object):

	"""
	Class containing tools necessary to compute window functions and window
	function estimated power spectra. 

	This code computes window functions for a given point spread function (PSF)
	for both cylindrical and spherical power spectra. The PSF is assumed to be that 
	of a small enough frequency chunk of data such that

	One thing that is maybe useful is that if you are using the HERA_hack_FG code 
	to generate the M_matrix, there is a useful method to find out the npix_col and npix_row. 
	run the observation class method "sky_shape()". The first output is npix_col, the second is npix_row. 

	This computs the window function for a single 8 MHz chunk of data (i.e when it is appropriate to make a coeval approximation)

	Note to self: try to implement a function that auto-bins in bayesian block binning

	"""
	
	def __init__(self, 
		  		PSF,
				cosmo_units,
				PSF_2 = None,
				freq_taper=None, 
				space_taper=None, 
				freq_taper_kwargs=None, 
				space_taper_kwargs=None, 
				verbose = False,
	):
		
		"""
        Initialisation of the Power_Spectrum class.

        Parameters
        ----------
		PSF : array_like
			3D array of the point spread function. The last axis must be the frequency axis.
		pspec: instance of the Power_Spectrum class
		cosmo_units : instance of the Cosmo_Conversions class
			Instance of the Cosmo_Conversions class containing the cosmological conversion factors.
		PSF_2 : array_like, optional
			3D array of the point spread function for the second map. The last axis must be the frequency axis.
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
		verbose : bool, optional
			Whether to print out extra information during computation. Default is False.
		"""

		# ensure input shapes are consistent
		if np.ndim(PSF) != 3:
			raise ValueError('Data must be a 3d array of dim (npix, npix, nfreqs).')

		if np.shape(PSF)[-1] != cosmo_units.z_npix:
			raise ValueError("The last axis of the data array must be the frequency axis.")	    
		
		self.PSF = PSF
		self.theta_x = (cosmo_units.theta_x)
		self.theta_y = (cosmo_units.theta_y)

		self.delta_x = cosmo_units.delta_x.value
		self.delta_y = cosmo_units.delta_y.value
		self.delta_z = cosmo_units.delta_z.value

		self.y_npix = cosmo_units.y_npix
		self.x_npix = cosmo_units.y_npix
		self.z_npix = cosmo_units.z_npix

		self.Lx = cosmo_units.Lx.value
		self.Ly = cosmo_units.Ly.value
		self.Lz = cosmo_units.Lz.value
		
		self.cosmo_volume = cosmo_units.cosmo_volume.value
	
		self.verbose = verbose

		if PSF_2 is not None:
			self.PSF_2 = PSF_2
		else:
			self.PSF_2 = None


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

		if self.sky_taper is not None:
		# raise warning that sky 
			warnings.warn("Only frequency tapering is permitted. Sky tapering is not yet implemented. Please set sky_taper to None.")

		print('Window function class initialised.')


	def _parse_taper_info(self,
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
				freq_taper, self.z_npix, **freq_taper_kwargs
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

	def compute_Gtilde(self):
		
		"""
		Compute the Gtilde matrix for the given PSF. 

		"""
		# compute the Gtilde matrix
		if self.verbose:
			print('Gtilde matrix computed.')
		x = np.fft.fftshift(np.fft.fftfreq(self.PSF.shape[0], d = 1/(self.theta_x)))
		y = np.fft.fftshift(np.fft.fftfreq(self.PSF.shape[0], d = 1/(self.theta_y)))

		dx = x[1] - x[0]
		dy = y[1] - y[0]
		
		if self.PSF_2 is not None:
			self.Gtilde = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.PSF*self.taper,axes = (0,1)), axes = (0,1)), axes = (0,1)) *dx*dy
			self.Gtilde_2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.PSF_2*self.taper,axes = (0,1)), axes = (0,1)), axes = (0,1))*dx*dy

			return self.Gtilde, self.Gtilde_2
		else:
			self.Gtilde = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.PSF*self.taper,axes = (0,1)), axes = (0,1)), axes = (0,1))*dx*dy
			
			
			return self.Gtilde

	def compute_norm3D(self):

		"""
		Compute the normalization of the power spectrum. 

		Returns
		-------
		norm3D : float
			The normalization of the power spectrum in (kx,ky,kx) bins.

		"""

		self.compute_Gtilde()

		taper_norm = 1
		if isinstance(self.taper, np.ndarray):
			if self.freq_taper is not None:
				taper_norm *= np.sum(self.freq_taper ** 2) / self.z_npix
			if self.theta_x_taper is not None:
				taper_norm *= np.sum(self.theta_x_taper ** 2) / self.x_npix
				taper_norm *= np.sum(self.theta_y_taper ** 2) / self.y_npix

		if self.PSF_2 is not None:
			Gtilde_squared = (self.Gtilde * np.conj(self.Gtilde_2))
		else:
			Gtilde_squared = (self.Gtilde * np.conj(self.Gtilde))
		
		Gtilde_squared /= taper_norm
		norm = (1/self.Lz) * (np.sum(Gtilde_squared, axis = 2)* self.delta_z) #normalization factor for the 3D window function
		norm = np.asarray(norm)
		#stack norm on itself len(spw_window) times
		norm = np.stack((norm,)*Gtilde_squared.shape[2], axis = 2)
		return norm
	
### EVENTUALLY WRAP THE BINNING STUFF UP NICELY HERE! 
	# def compute_norm2D(self):
	# 	compute_norm3D()


	# def compute_norm1D(self):
	# 	compute_norm3D()

	def compute_kmodes(self):

		"""
		Compute the Fourier grid. 

		"""
		self.kx = np.fft.fftshift(
		np.fft.fftfreq(self.x_npix, d=self.delta_x))  # 1/Mpc
		self.kx *= 2*np.pi
		
		self.ky = np.fft.fftshift(
			np.fft.fftfreq(self.y_npix, d=self.delta_y))  # 1/Mpc
		self.ky *= 2*np.pi
		
		self.k_par = np.fft.fftshift(
			np.fft.fftfreq(self.z_npix, d=self.delta_z)  )# 1/Mpc
		self.k_par *= 2*np.pi

		return self.kx, self.ky, self.k_par
	

	
	def bin_Gtilde_kperp(self, PSF_UV, kperp_cutoff = None):

		"""
		Bin the Gtilde matrix in k_perp and k_parallel. 

		"""
		self.compute_kmodes()

		kmag_perp = np.sqrt(self.kx[None, :] ** 2 + self.ky[:, None] ** 2)

		#TODO: make this random 95 more general. Is there a rule for how fine you can bin this without gaps? 
		if kperp_cutoff is not None:
			self.kperp_bin = np.linspace((np.min(kmag_perp)),kperp_cutoff,95)
		else:
			self.kperp_bin = np.linspace((np.min(kmag_perp)),(np.max(kmag_perp)),95)


		kpar_bin = self.k_par + np.diff(self.k_par)[0]/2
		#add one value to the beginning of k_par_bin
		kpar_bin = np.insert(kpar_bin,0,self.k_par[0] - np.diff(self.k_par)[0]/2)


		kmag_perp_flat = np.reshape(
					kmag_perp,
					kmag_perp.size,
					order='C'
				)

		gauss_fft_flat = np.reshape(
			PSF_UV,
			(kmag_perp.size, -1),
			order='C'
		).flatten('C')

		# get pixel coordinates in Fourier space
		coords = np.meshgrid(self.k_par,kmag_perp_flat)
		coords = [coords[0].flatten(), coords[1].flatten()]

		# histogram the power spectrum box
		ret_real = binned_statistic_2d(
			x=coords[0], # k_par
			y=coords[1], # k_perp
			values=gauss_fft_flat.real,
			bins=[kpar_bin,self.kperp_bin],
			statistic='mean',
		)

		# histogram the power spectrum box
		ret_imag = binned_statistic_2d(
			x=coords[0], # k_par
			y=coords[1], # k_perp
			values=gauss_fft_flat.imag,
			bins=[kpar_bin,self.kperp_bin],
			statistic='mean',
		)

		G_tilde_binned = ret_real.statistic + 1j*ret_imag.statistic

		return G_tilde_binned 


	
	def padding_Gtilde(self):

		"""
		Pad the Gtilde matrix with zeros to expand the Fourier grid in the k_parallel direction. 

		"""
		self.compute_Gtilde()

		if self.PSF_2 is not None:
			G_tilde_binned = self.bin_Gtilde_kperp(self.Gtilde, kperp_cutoff = None)
			G_tilde_2_binned =self.bin_Gtilde_kperp(self.Gtilde_2, kperp_cutoff = None)

			self.Gtilde_pad = np.pad(G_tilde_binned, ((self.z_npix, self.z_npix), (0,0)), mode = 'constant', constant_values = 0)
			self.Gtilde_2_pad = np.pad(G_tilde_2_binned, ((self.z_npix, self.z_npix), (0,0)), mode = 'constant', constant_values = 0)
			
			return self.Gtilde_pad, self.Gtilde_2_pad

		else:
			G_tilde_binned = self.bin_Gtilde_kperp(self.Gtilde, kperp_cutoff = None)
			
			self.Gtilde_pad = np.pad(G_tilde_binned, ((self.z_npix, self.z_npix), (0,0)), mode = 'constant', constant_values = 0)
			return self.Gtilde_pad
	
	def compute_Wpar(self):
		
		"""
		Compute the window function in the parallel direction. 

		"""
		self.padding_Gtilde()

		taper_norm = 1
		if isinstance(self.taper, np.ndarray):
			if self.freq_taper is not None:
				taper_norm *= np.sum(self.freq_taper ** 2) / self.z_npix
			if self.theta_x_taper is not None:
				taper_norm *= np.sum(self.theta_x_taper ** 2) / self.x_npix
				taper_norm *= np.sum(self.theta_y_taper ** 2) / self.y_npix
		
		if self.PSF_2 is not None:
			F_par = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(self.Gtilde_pad, axes = 0), axis = 0), axes = 0) * self.delta_z
			F_par_2 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(self.Gtilde_2_pad, axes = 0), axis = 0), axes = 0) * self.delta_z
			self.k_par_long = np.fft.fftshift(np.fft.fftfreq(self.Gtilde_pad.shape[0], self.delta_z))
			self.k_par_long *= 2*np.pi
			
			self.Wpar_kperp_kz = ((F_par*np.conj(F_par_2)) / taper_norm).T
			return self.Wpar_kperp_kz.real
		
		else:
			F_par = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(self.Gtilde_pad, axes = 0), axis = 0), axes = 0) * self.delta_z
			self.k_par_long = np.fft.fftshift(np.fft.fftfreq(self.Gtilde_pad.shape[0], self.delta_z))
			self.k_par_long *= 2*np.pi

			self.Wpar_kperp_kz = ((F_par*np.conj(F_par))/taper_norm).T
			return self.Wpar_kperp_kz.real
		

	
	def build_Wkper_kkz(self):


		"""
		Build the window function in the k_perp and k_parallel directions. 

		"""
		self.compute_Wpar()
		nperp = self.Wpar_kperp_kz.shape[0]
		npar = self.Wpar_kperp_kz.shape[1]


		self.W_kperp_kkz = np.zeros((nperp,npar,npar), dtype = complex)

		for i in tqdm(range(nperp)):
			for j in range(npar):
				for k in range(npar):
					idx = (j - k) + npar//2
					if idx < 0 or idx >= npar:
						continue
					self.W_kperp_kkz[i,j,k] = self.Wpar_kperp_kz[i,idx]
	
	
	
	def compute_1D_window(self,k, k_prime, *args,norm = True, smoothing = False, sigma = 1, **kwargs):

		"""
		Compute the 1D window function in k_perp and k_parallel directions. 

		Parameters
		----------
		k : array_like
			k bins for which you want to calculate the window functions.
		k_prime : array_like
			k grid for which you want to calculate the window functions.
		*args : list
		norm : bool, optional
			Normalize the window function. Default is True.
		smoothing : bool, optional
			Apply a Gaussian smoothing to the window function. Default is False.
		sigma : float, optional
			The standard deviation of the Gaussian smoothing kernel. Default is 1.
		**kwargs : dict
			Additional keyword arguments to pass to the Gaussian smoothing function.

		Returns
		-------
		W_k_kprime : array_like
			The 1D window functions.

		"""
		self.build_Wkper_kkz()



		nperp = self.W_kperp_kkz.shape[0]
		npar = self.W_kperp_kkz.shape[1]
		#get k bin edges 
		k_edges = k + np.diff(k)[0]/2
		k_edges = np.insert(k_edges,0,k[0] - np.diff(k)[0]/2)
		
		### warnings.filterwarnings("This only works for evenly linearly spaced k bins")

		#make this len(k_edges) and have the 0 row be the garbage bin
		self.W_k_kprime = np.zeros((len(k_edges), len(k_prime)))

		for i in tqdm(range(nperp)):
			for j in range(npar):

				k_mag = np.sqrt(self.kperp_bin[i]**2 + self.k_par_long[j]**2)
				# check with k bin it falls into
				idx_1 = np.digitize(k_mag, k_edges)
				if idx_1 > len(k):
					continue	
				for l in range(npar):
					k_prime_mag = np.sqrt(self.kperp_bin[i]**2 + self.k_par_long[l]**2)
					idx_2 = np.digitize(k_prime_mag, k_prime)
					if idx_2 >= len(k_prime):
						continue
			
					self.W_k_kprime[idx_1, idx_2] += self.W_kperp_kkz[i,j,l].real
		if norm:
			N = np.sum(self.W_k_kprime, axis = 1) 
			N *= np.diff(k_prime)[0]
			self.W_k_kprime = self.W_k_kprime/N[:,None]
			# W_k_kprime now has units of Mpc

		if smoothing:
			self.W_k_kprime = gaussian_filter1d(self.W_k_kprime, sigma = sigma, axis = 1)
		
		#remove the garbage bin
		return k_prime, self.W_k_kprime[1:,:]
	
