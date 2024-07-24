from cached_property import cached_property

import os
import numpy as np
import scipy.constants as sc
from astropy import constants, units, cosmology
import warnings
from . import utils


class cosmo_units(object):
    """
    Class containing methods to convert survey volume to cosmological units. 

    This also defines some analogous quanitites in Fourier space (e.g. dk).
    """
        
    def __init__(self,
            theta_x,
            theta_y,
            x_npix,
            y_npix,
            freqs,
            rest_freq=1420.*units.MHz,
            cosmo=cosmology.Planck18,
            verbose = False,
            ):
            
        """
        Initialisation of the Power_Spectrum class.

        Parameters
        ----------
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
            verbose: bool
                Whether to output messages when running functions.
        """

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
        self.y_npix = y_npix
        self.x_npix = x_npix
        self.freq_npix = self.freqs.shape[0]

		# these two lines give you the physical dimensions of a pixel
		# (inverse of sampling ratealong each axis)
        self.delta_thetay = self.theta_y / self.y_npix
        self.delta_thetax = self.theta_x / self.x_npix
        self.delta_freq = np.diff(self.freqs).mean()

        self.cosmo = cosmo
        self.verbose = verbose
	


### Conversion functions 

    @cached_property
    def dRperp_dtheta(self):
        """Conversion from radians to Mpc."""
        return self.cosmo.comoving_distance(self.z).to(units.Mpc)

    @cached_property
    def dRperp_dOmega(self):
        """Conversion from sr to Mpc^2."""
        return self.dRperp_dtheta ** 2

    @cached_property
    def dRpara_dnu(self):
        """Conversion from Hz to Mpc."""
        return (
            constants.c * (1 + self.z)**2
            / (self.cosmo.H(self.z) * self.rest_freq)
        ).to("Mpc")

    @cached_property
    def X2Y(self):
        """Conversion from image cube volume to cosmological volume."""
        return self.dRperp_dOmega * self.dRpara_dnu


    ### Volume Properties 
    @cached_property
    def cosmo_volume(self):
        """Full cosmological volume of the image cube in Mpc^3."""
        # If we update this to allow for varying pixel sizes, then this will
        # need to be changed to an integral.
        n_pix = self.x_npix * self.y_npix * self.freq_npix
        return n_pix * self.volume_element

    @cached_property
    def volume_element(self):
        """ Cosmological volume element in Mpc^3. """
        return (
            self.delta_thetax * self.delta_thetay * self.delta_freq * self.X2Y
        )


    ### Length Properties 
    @cached_property
    def delta_x(self): 
        """ X line element in Mpc"""
        return self.delta_thetax * self.dRperp_dtheta
    @cached_property
    def delta_y(self): 
        """ Y line element in Mpc"""
        return self.delta_thetay * self.dRperp_dtheta

    @cached_property
    def delta_z(self): 
        """ Z line element in Mpc"""
        return self.delta_freq * self.dRpara_dnu

    @cached_property
    def Lx(self): 
        """ Length of side X inin Mpc"""
        return self.delta_x * self.x_npix

    @cached_property
    def Ly(self):
        """ Length of side Y in in Mpc"""
        return self.delta_y * self.y_npix

    @cached_property
    def Lz(self): 
        """ Length of side Z in in Mpc"""
        return self.delta_z * self.freq_npix

    #### Fourier Properties 
    @cached_property
    def delta_k_par(self):
        """ k_par line element in 1/Mpc"""
        return 2 * np.pi / self.Lz	

    @cached_property
    def delta_k_perp(self):
        """ k_perp line element in 1/Mpc^2"""
        return (2*np.pi)**2 / np.sqrt((self.Lx*self.Ly))

    @cached_property
    def delta_k_x(self):
        """ k_x line element in 1/Mpc"""
        return (2*np.pi) / (self.Lx)

    @cached_property
    def delta_k_y(self):
        """ k_y line element in 1/Mpc """
        return (2*np.pi) / (self.Ly)
    


    def print_quantities(self):
        print('Lx:', self.Lx,
              'Ly:', self.Ly,
              'Lz:', self.Lz,
              'Box Volume:', self.cosmo_volume,
              'delta_x:', self.delta_x,
              'delta_y:', self.delta_y,
              'delta_z:', self.delta_z,
              'Volume Element:', self.volume_element,
              'delta_k_par:', self.delta_k_par,
              'delta_k_perp,',self.delta_k_perp,
              'delta_k_x:',self.delta_k_x,
              'delta_k_y:',self.delta_k_y,
               sep='\n')
