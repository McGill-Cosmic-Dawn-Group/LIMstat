import sys
import os
import os.path
import numpy as np
import numpy.linalg as la
import numpy.random as ra
import pandas as pd
import healpy as hp
import astroquery
import scipy.constants as sc
from astroquery.vizier import Vizier
import HERA_hack_FG

####CODE FOR ADDING FOREGROUNDS####

#1. Galactic Synchrotron Portion
#2. Free-Free emission
#3. Point sources
#4. Unresolved point sources

#pseudocode for including foregrounds in the sky vector


"""	The contiribution of the foreground will depend on the observation you want to make
(in the sky you are looking,at what freq, etc...) when in the universe's history,

 """

class foregrounds(object):

	def __init__(self, obs,*args, path_to_maps = None,**kwargs):


		self.freq = obs.freq # observed frequency in MHz
		self.Npix = len(obs.observable_coordinates()) # total number of pixels in the map
		self.observable_coordinates = obs.observable_coordinates() # the coordinates of the observed patch of the sky
		obs.necessary_times()
		self.latitude = obs.latitude #latitutde of the instrument
		self.times = obs.times #integration time steps
		self.Nt = obs.Nt #number of integrations
		self.position  = obs.position
		self.beam_width = obs.beam_width # beam FWHM
		self.path = path_to_maps

		#inheret observation properties like where in the sky/at what freq the observation is made.

	def compute_synchro_pygsm(self):

		'''
		This method computes galactic synchrotron emission by
		findng the relevant pixels from the pygsm model
		based on the portion of observed sky, and the oberving frequency

		see Oliveira-Costa et. al., (2008) and Zheng et. al., (2016)
		'''

		## MAKE THIS READ FROM pygdsm.bin

		nside = 1024
		#Reads in the file for the frequecny in question
		diffuse_synchrotron = np.fromfile('pygdsm_%sMHz.bin'%int(self.freq), dtype=np.float32)

		obs_index = hp.pixelfunc.ang2pix(nside, self.observable_coordinates[:,0],self.observable_coordinates[:,1])

		self.gal_emission = []

		for i in range(len(obs_index)):
			self.gal_emission.append(diffuse_synchrotron[obs_index[i]])


		return self.gal_emission ## THIS IS IN KELVIN

	def compute_synchro(self):

		'''
		This method computes a statistically accurate
		galactric synchrotron model published in Liu et. al., (2011)
		'''


		alpha_0_syn = 2.8
		sigma_syn = 0.1
		Asyn = 335.4 #K

		pixel_flux_syn = []

		alpha_syn = np.random.normal(alpha_0_syn,sigma_syn,self.Npix)

		for i in range(self.Npix):
			flux = Asyn *((self.freq/150)**(-alpha_syn[i]))

			pixel_flux_syn.append(flux)

		self.gal_emission = np.asarray(pixel_flux_syn)
		return self.gal_emission # in Kelvin




	def compute_bremsstrauhlung(self):

		'''
		This method computes a map of diffuse free-free
		emission from a model published in Liu et. al., (2011)
		'''

		alpha_0_ff = 2.15
		sigma_ff = 0.01
		Aff = 33.5 #K

		pixel_flux_ff = []

		alpha_ff = np.random.normal(alpha_0_ff,sigma_ff,self.Npix)


		for i in range(self.Npix):
			flux = Aff*(self.freq/150)**(-alpha_ff[i])
			pixel_flux_ff.append(flux)

		self.free_free = np.asarray(pixel_flux_ff)

		return self.free_free # THIS IS IN KELVIN

	def compute_omega(self): #this is for temp brightness conversion

		phi = self.observable_coordinates[:,1]

		min_indices = np.where(phi == min(phi))

		upper_index = max(min_indices[0])+1

		theta_res = np.abs(np.cos(self.observable_coordinates[1,0])-np.cos(self.observable_coordinates[0,0]))
		phi_res = self.observable_coordinates[upper_index,1]- self.observable_coordinates[1,1]

		self.omega_pix = float(theta_res*phi_res)


	def compute_unres_point_sources(self,n_sources):




		gamma = 1.75

		def dnds(s):
			return 4.*(s/880)**(-gamma)

		s = np.arange(8,100,1) #maybe make this an argument

		pdf = np.asarray([s,dnds(s)]) #0 is s, 1 is dnds
		prob = pdf[1]/float(sum(pdf[1]))
		cum_prob = np.cumsum(prob)

		def gen_fluxes(N):
			R = ra.uniform(0, 1, N)
			#Here we first find the bin interval that random number lies in min(cum_prob[])
			#then we find the flux who's index is that cum_prob
			#repat for all r in R
			return [int(s[np.argwhere(cum_prob == min(cum_prob[(cum_prob - r) > 0]))]) for r in R]

		alpha_0 = 2.5
		sigma = 0.5

		self.compute_omega()

		factor = 1.4e-6*((self.freq/150)**(-2))*(self.omega_pix**(-1))

		pixel_flux = []

		for i in range(self.Npix):
			alpha = np.random.normal(alpha_0,sigma,n_sources)
			S_star = gen_fluxes(n_sources)
			sum_fluxes = 0

			for j in range(n_sources-1):
				sum_fluxes += factor*S_star[j]*(self.freq/150)**(-alpha[j])

			pixel_flux.append(sum_fluxes/n_sources)

		self.sources = np.asarray(pixel_flux)

		return self.sources # THIS IS IN KELVIN


	def diffuse_fg(self,n_sources,pygsm): #should also have data input

		if pygsm == True:

			self.fg_map = self.compute_synchro_pygsm() + self.compute_bremsstrauhlung() + self.compute_unres_point_sources(n_sources)

			return self.fg_map

		elif pygsm == False:


			self.fg_map = self.compute_synchro() + self.compute_bremsstrauhlung() + self.compute_unres_point_sources(n_sources)

			return self.fg_map
