from cached_property import cached_property

import os
import numpy as np
import scipy.constants as sc
from astropy import constants, units, cosmology
from astropy.cosmology import units as cunits
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
            redshift = None,
            rest_freq=1420.*units.MHz,
            cosmo=cosmology.Planck18,
            little_h=False,
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
            redshift: float
                Redshift at the box centre, if defined from cosmological space.
            rest_freq: float
                Rest-frequency of the emission line in units.MHz.
                Default is 1420 for the 21cm line.
            cosmo: astropy.cosmology class
                Cosmology to use for computations.
                Default is Planck18.
            little_h: bool
                Whether to use Mpc/h or not.
                Default is False.
            verbose: bool
                Whether to output messages when running functions.
        """
        # Check cosmology
        assert isinstance(cosmo, cosmology.Cosmology), \
            'cosmo must be an astropy.Cosmology object.'
        
        self.cosmo = cosmo
        self.x_npix = x_npix
        self.y_npix = y_npix
        self.little_h = bool(little_h)
        self.rest_freq = utils.comply_units(
            value=rest_freq,
            default_unit=units.MHz,
            quantity="rest_freq",
            desired_unit=units.Hz,
        )

        if theta_x is not None:

            self.freqs = utils.comply_units(
                value=freqs,
                default_unit=units.MHz,
                quantity="freqs",
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

            self.delta_thetay = self.theta_y.value / self.y_npix
            self.delta_thetax = self.theta_x.value / self.x_npix
            self.delta_freq = np.diff(self.freqs).mean()

            self.dRpara_dnu = (constants.c * (1 + self.z)**2/ (self.cosmo.H(self.z).si * self.rest_freq)).to("Mpc/Hz")
            self.dRperp_dtheta = self.cosmo.comoving_distance(self.z).to(units.Mpc)
            if self.little_h:
                self.dRperp_dtheta = self.dRperp_dtheta.to(
                    units.Mpc/cunits.littleh,
                    cunits.with_H0(self.cosmo.H0)
                )
                self.dRpara_dnu = self.dRpara_dnu.to(
                    units.Mpc/cunits.littleh/units.Hz,
                    cunits.with_H0(self.cosmo.H0)
                )
            self.Lx = self.theta_x.value * self.dRperp_dtheta
            self.Ly = self.theta_y.value * self.dRperp_dtheta
            self.Lz = (max(self.freqs) - min(self.freqs)) *  self.dRpara_dnu

        else:
            if redshift is None:
                raise ValueError('Must specify redshift along with box length.')
            self.z = float(redshift)
            self.Lx = utils.comply_units(
                value=Lx,
                default_unit=units.Mpc,
                quantity="Lx",
                desired_unit=units.Mpc,
            )
            self.Ly = utils.comply_units(
                value=Ly,
                default_unit=units.Mpc,
                quantity="Ly",
                desired_unit=units.Mpc,
            )
            self.Lz = utils.comply_units(
                value=Lz,
                default_unit=units.Mpc,
                quantity="Lz",
                desired_unit=units.Mpc,
            )
            self.z_npix = int(z_npix)

            self.dRperp_dtheta = self.cosmo.comoving_distance(self.z).to(units.Mpc)
            self.theta_x = self.Lx / self.dRperp_dtheta
            self.theta_y = self.Ly / self.dRperp_dtheta
            self.delta_thetax = self.theta_x / self.x_npix
            self.delta_thetay = self.theta_y / self.y_npix

            self.dRpara_dnu = (constants.c * (1 + self.z)**2/ (self.cosmo.H(self.z).si * self.rest_freq)).to("Mpc/Hz")
            bandwidth = self.Lz / self.dRpara_dnu
            self.mid_freq = self.rest_freq / (1. + self.z)
            fmin = self.mid_freq - bandwidth/2.
            fmax = self.mid_freq + bandwidth/2.
            self.freqs = np.linspace(fmin.value, fmax.value, self.z_npix) * units.Hz
            self.delta_freq = np.diff(self.freqs.value).mean() * units.Hz

        if self.little_h:
            self.Lx = self.Lx.to(
                units.Mpc/cunits.littleh,
                cunits.with_H0(self.cosmo.H0)
            )
            self.Ly = self.Ly.to(
                units.Mpc/cunits.littleh,
                cunits.with_H0(self.cosmo.H0)
            )
            self.Lz = self.Lz.to(
                units.Mpc/cunits.littleh,
                cunits.with_H0(self.cosmo.H0)
            )

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



