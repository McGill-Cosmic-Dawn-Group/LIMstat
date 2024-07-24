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
            x_npix,
            y_npix,
            z_npix = None, 
            Lx = None,
            Ly = None,
            Lz = None,
            theta_x = None,
            theta_y = None,
            freqs = None,
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
        # Check cosmology
        assert isinstance(cosmo, cosmology.Cosmology), \
            'cosmo must be an astropy.Cosmology object.'
        
        self.cosmo = cosmo
        self.x_npix = x_npix
        self.y_npix = y_npix

        if theta_x is not None:

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

            self.z_npix = self.freqs.shape[0]

            self.delta_thetay = self.theta_y / self.y_npix
            self.delta_thetax = self.theta_x / self.x_npix
            self.delta_freq = np.diff(self.freqs).mean()

            self.dRpara_dnu = (constants.c * (1 + self.z)**2/ (self.cosmo.H(self.z) * self.rest_freq)).to("Mpc")
            self.dRperp_dtheta = self.cosmo.comoving_distance(self.z).to(units.Mpc)

            self.Lx = self.theta_x * self.dRperp_dtheta
            self.Ly = self.theta_y * self.dRperp_dtheta
            self.Lz = (max(self.freqs) - min(self.freqs)) *  self.dRpara_dnu
        else:
            self.Lx = Lx
            self.Ly = Ly
            self.Lz  = Lz

            self.z_npix = z_npix


        # # these two lines give you the physical dimensions of a pixel
        # # (inverse of sampling ratealong each axis)
        # self.delta_thetay = self.theta_y / self.y_npix
        # self.delta_thetax = self.theta_x / self.x_npix
        # self.delta_freq = np.diff(self.freqs).mean()

        self.verbose = verbose
	

   ### Length Properties 
    @cached_property
    def delta_x(self): 
        """ X line element in Mpc"""
        return self.Lx / self.x_npix
    
    @cached_property
    def delta_y(self): 
        """ Y line element in Mpc"""
        return self.Ly / self.y_npix

    @cached_property
    def delta_z(self): 
        """ Z line element in Mpc"""
        return self.Lz / self.z_npix

    ### Volume Properties 
    @cached_property
    def cosmo_volume(self):
        """Full cosmological volume of the image cube in Mpc^3."""
        return self.Lx * self.Ly * self.Lz 

    @cached_property
    def volume_element(self):
        """ Cosmological volume element in Mpc^3. """
        return self.delta_x * self.delta_y * self.delta_z 
 
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
    def delta_kx(self):
        """ kx line element in 1/Mpc"""
        return (2*np.pi) / (self.Lx)

    @cached_property
    def delta_ky(self):
        """ ky line element in 1/Mpc """
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
              'delta_k_x:',self.delta_kx,
              'delta_k_y:',self.delta_ky,
               sep='\n')



