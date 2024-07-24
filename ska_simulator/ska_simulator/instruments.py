
import numpy as np
from astropy import units
from matplotlib import pyplot as plt

def gaussian_2D_cube(x,y,sigma_x,sigma_y,mu_x,mu_y, normalization = 'area'):
	""" Code for generating 2D Gaussian cube. 
	
		Parameters
		----------
		x: float
			x coordinate of pixel.
		y: float
			y coordinate of pixel.
		sigma_x: float
			Standard deviation of the Gaussian in the x direction.
		sigma_y: float
			Standard deviation of the Gaussian in the y direction.
		mu_x: float
			Mean of the Gaussian in the x direction.
		mu_y: float
			Mean of the Gaussian in the y direction.
	"""

	xx, yy = np.meshgrid(x,y, sparse = True)
	#broadcasting of xx in the freq direction	
	xx = xx[:,:,np.newaxis]
	#broadcasting of yy in the freq direction	
	yy = yy[:,:,np.newaxis]

	mu_x = mu_x[np.newaxis,np.newaxis,:]
	mu_y = mu_y[np.newaxis,np.newaxis,:]
	sigma_x = sigma_x[np.newaxis,np.newaxis,:]
	sigma_y = sigma_y[np.newaxis,np.newaxis,:]

	gaussian_box =  np.exp(-((((xx-mu_x)**2)/(2*(sigma_x**2))) + (((yy-mu_y)**2)/(2*(sigma_y**2)))))

	if normalization == 'area':
		area = np.sum(gaussian_box, axis = (0,1))
		gaussian_box /= area
		return gaussian_box

	elif normalization == 'peak':
		return gaussian_box/np.max(gaussian_box)

def single_dish_instrument(sky, 
						cosmo_units, 
						freqs, 
						dish_diameter, 
						omega_pix_box, 
						normalization = 'area', 
						noise = False, 
						sigma_pix = None, 
						t_obs = None):

	""" Function that simulates a single dish instrument. This creates the PSF and convolves it with the sky.
		It then adds noise to the convolved image.
		
		Parameters
		----------
		sky: 2D array
			The sky to be observed.
		cosmo_units: CosmoUnits object
			An instance of the cosmo_units object.
		freqs: array
			The frequencies to be observed at in GHz.
		dish_diameter: float
			The diameter of the dish in meters.
		noise: bool
			Whether or not to include noise.
		omega_pix_box: float
			The solid angle of a pixel in steradians.
		sigma_pix: float
			The per pixel noise in Jy/ sr s^-1/2.
		t_obs: float
			The observation time in hours.
										"""
	if sky.shape[-1] != len(freqs):
		raise ValueError('The sky box must have freq axis on the last axis.')
	sky_box = sky.value
	sky_unit = sky.unit
	#convert all units to SI
	dish_diameter = dish_diameter.to(units.m).value
	omega_pix_box = omega_pix_box.to(units.sr).value
	freqs = freqs.to(units.Hz).value
	nfreqs = freqs.shape[0]
	npix = sky.shape[0]

   #diffaction limited angular resolution in radians 
	beamFWHM = (3e8/(freqs[nfreqs//2]))/dish_diameter #resolution for each frequency channel in radians
	beamFWHM = np.repeat(beamFWHM, nfreqs)
	mu = np.zeros(len(freqs)) #mean of the gaussian in the frequency direction
	
	x = np.fft.fftshift(np.fft.fftfreq(sky.shape[0], d = 1/(np.sqrt(omega_pix_box)*npix)))
	y = np.fft.fftshift(np.fft.fftfreq(sky.shape[0], d = 1/(np.sqrt(omega_pix_box)*npix)))
		
	dx = x[1] - x[0]
	dy = y[1] - y[0]
	gaussian_cube = gaussian_2D_cube(x,y,beamFWHM,beamFWHM,mu,mu, normalization = normalization)
	gaussian_cube /= dx*dy 


	sky_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky_box, axes = (0,1)), axes = (0,1)), axes = (0,1)) *dx*dy
	gauss_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gaussian_cube, axes = (0,1)), axes = (0,1)), axes = (0,1)) *dx*dy
	k = np.fft.fftshift(np.fft.fftfreq(sky.shape[0], d = dx))
	dk = k[1] - k[0]

	
	product = sky_fft * gauss_fft 
	new_sky = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(product, axes = (0,1)), axes = (0,1)), axes = (0,1)) *dk*dk*(npix**2)


	if noise == True:
		t_obs = t_obs.to(units.s)
		sigma_pix = sigma_pix.to('Jy s^(1/2)/sr')

		#instrument pixel size
		omega_pix_inst = (np.pi*(beamFWHM**2))/(4*np.log(2)) #sr

		# 1: find sigma_pix for the given pixel size from sigma_big = sigma_small / root(n)
		npix_in_pix = omega_pix_box/omega_pix_inst #num pix in pix 
		sigma = (sigma_pix)/np.sqrt(npix_in_pix) # this is still in Jy/sr s^1/2
		# calculate time observing each pixel
		omega_surv = omega_pix_box*cosmo_units.x_npix*cosmo_units.y_npix
		t_pix = t_obs*(omega_pix_box/omega_surv)
		sigma_rms = sigma/(np.sqrt(t_pix)) #Jr/sr --> convert to K
		#convert to K
		equiv = units.brightness_temperature(freqs*units.Hz)
		sigma_rms = sigma_rms.to(units.K, equivalencies=equiv)
		noise_unit = sigma_rms.unit

		# check if noise_unit and sky unit are the same
		if noise_unit != sky_unit:
			raise ValueError('The units of the noise and the sky are not the same. Sky has units of {} and noise has units of {}.'.format(sky_unit, noise_unit))
		
		#draw random map and add to convolved sky!
		noise_map = np.random.normal(0,sigma_rms.value, (cosmo_units.x_npix,cosmo_units.y_npix, len(freqs)))
		new_sky += noise_map

		return np.real(new_sky) * sky_unit , gaussian_cube, noise_map * sky_unit

		 
	else:	
		return np.real(new_sky) * sky_unit, gaussian_cube, noise_map * sky_unit

# Make a few wrapper functions for common single dish instruments (e.g. CCAT-p, CONCERTO, TIME, etc.)




