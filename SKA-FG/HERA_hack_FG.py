#these are for the telescope and observation
import numpy as np 
import numpy.linalg as la
from scipy import signal
import scipy.constants as sc
# from timeit import default_timer as timer
import pandas as pd
import matplotlib.pyplot as plt

'''

'''


class telescope(object):
	"""
	Object to store the properties of the telescope, including:
	---Telescope location
	---Array configuration
	---Primary beam
	"""
	def __init__(self, ant_locs, latitude, channel_width, beam_width, beam='gaussian'):
		self.ant_locs = ant_locs # (Nants,2) sized array # relative position in meters 
		self.latitude = latitude # degrees this is the central latitute of the HERA strip 
		self.latitude *= np.pi / 180. # radians
		self.channel_width = channel_width # assume Hz
		
		if beam != 'gaussian':
			raise NotImplementedError()
		else:
			self.beam = beam
			self.beam_width = beam_width # FWHM in degrees
			self.beam_width *= np.pi / 180. # FWHM in radians


	def compute_2D_bls(self): #DO NOTE RETURN
		"""
		Computes the 2D ("uv plane") baselines for a local coordinate
		system when the observations started.
		"""
		N_ants = self.ant_locs.shape[0]

		N_bls = int(N_ants * (N_ants - 1) / 2.) #this is the total number of baselines, redundant included
		self.bls = np.zeros((N_bls,2)) # initialize a list holding the length of all the baselines 
		k = 0 #initialize this k variable
		for i in range(N_ants):
			ant_i = self.ant_locs[i]
			for j in range(i+1,N_ants):
				ant_j = self.ant_locs[j]
				self.bls[k] = ant_i - ant_j # this subtracts each coordinate from the other [0,0]-[1,1]
				k += 1 #add k every time you identify a baseline 
				
				
	def remove_redundant_bls(self): #DO NOT RETURN
		"""
		Removes redundant baselines. Information is preserved by
		storing a multiplicity array.
		"""
		self.compute_2D_bls()
		self.bls, self.ucounts = np.unique(self.bls, axis=0, return_counts=True) #picking out only unique bls
		self.N_bls= len(self.bls) # number of unique pairs of baselines

	def compute_celestial_bls(self):#REUTRN bls_celestial
		"""
		Computes a 3D distribution of baselines in a coordinate
		system fixed to the celestial sphere.
		"""
		# First define the rotation matrix that gets baselines
		# to the correct latitude for the array
		#rotate about the z axis 
		self.remove_redundant_bls()
		
		co_lat = (np.pi / 2.) - self.latitude 
		cos_co_lat = np.cos(co_lat)
		sin_co_lat = np.sin(co_lat)
		lat_rot_matrix = np.array([[1., 0., 0.], #X
								  [0., cos_co_lat, -sin_co_lat], #Y
								  [0., sin_co_lat, cos_co_lat]]) #Z


		# Add a z coordinate to the 2D baseline vectors 
		self.bls_celestial = np.vstack((self.bls.T, np.zeros(self.N_bls)))
	   
		# Rotate them! 
		self.bls_celestial = np.dot(lat_rot_matrix, self.bls_celestial) #3 x N_bls array
		
		return self.bls_celestial 

class observation(object):
	"""
	Object that stores all the sky information 
	"""

	def __init__(self, telescope, n_days, freq, delta_t, corners, beam_sigma_cutoff, sky_shape ,norm,pbeam):
		self.times = None # in days
		self.position = None# observable corners (theta, phi)
		self.bl_times = None
		self.norm = None
		self.corners = corners# corners has to be the four corner of the sky (theta, phi) [4,2]
		self.beam_sigma_cutoff = beam_sigma_cutoff
		self.telescope = telescope
		self.n_days = n_days # Number of cycles of the observation
		self.delta_t = delta_t #length of time steps 
		#self.effarea = effarea # Effective area of an antenna
		self.normalization = norm #indicate whether to include normalization in mapping
		self.primary_beam = pbeam #indicate whether to include pbeam in Amatrix
		
		
		self.beam_width = telescope.beam_width
		self.beam = telescope.beam
		self.ant_locs = telescope.ant_locs
		self.latitude = telescope.latitude
		self.channel_width = telescope.channel_width
		self.bls_celestial = telescope.compute_celestial_bls()
		self.ucounts = telescope.ucounts # number of unique baselines
		
		self.freq = freq #MHz
		self.Tsys = 100+(120*((self.freq/150)**(-2.55)))

		self.npix_theta = sky_shape[0]
		self.npix_phi = sky_shape[1]

		self.Nbl = self.bls_celestial.shape[1]

		
	# def observable_coordinates(self): #RETURN self.position
	# 	"""
	# 	Find the observable coordinates given the four corners of a patch
	# 	"""
	# 	s = self.corners.shape
	# 	if s[0] != 4:
	# 		raise ValueError('Four coordinates should be provided')
	# 	elif s[1] != 2:
	# 		raise ValueError('Input has to be RA & Dec')
			
	# 	beam_minus, beam_plus = np.array([-1, +1])*self.beam_width*self.beam_sigma_cutoff
	# 	min_corner = np.min(self.corners, axis=0)
	# 	max_corner = np.max(self.corners, axis=0)
		 
	# 	min_obsbound = self.latitude + beam_minus #Here we're computing the size of the observing window
	# 	max_obsbound = self.latitude + beam_plus  #bigger beam_width or sigma_cutoff means a larger observing window 
	# 	#print(min_obsbound*180./np.pi,max_obsbound*180./np.pi)
	# 	# Now convert from latitude (measured from the equatorial plane up)
	# 	# to the polar angle (measured down from the z axis)
	# 	# Note how "max" and "min" swap places here
	# 	max_swap = max_obsbound
	# 	max_obsbound = np.pi / 2. - min_obsbound 
	# 	min_obsbound = np.pi / 2. - max_swap
		

	# 	if max_corner[0]*(np.pi/180.) < min_obsbound or min_corner[0]*(np.pi/180.) > max_obsbound:
	# 		raise ValueError('Requested Region is not observable')
	# 	else:
	# 		thetas = np.arange(min_obsbound, max_obsbound, self.resol) 

	# 		phis = np.arange((min_corner[1]*(np.pi/180.)), (max_corner[1]*(np.pi/180.)), self.resol)
		
	# 		self.position = np.concatenate(np.dstack(np.meshgrid(thetas, phis)))
	# 		self.Npix = self.position.shape[0]		
	# 	return self.position

	def observable_coordinates(self):


		s = self.corners.shape
		if s[0] != 4:
			raise ValueError('Four coordinates should be provided')
		elif s[1] != 2:
			raise ValueError('Input has to be RA & Dec')
			
		beam_minus, beam_plus = np.array([-1, +1])*self.beam_width*self.beam_sigma_cutoff
		min_corner = np.min(self.corners, axis=0)
		max_corner = np.max(self.corners, axis=0)
		 
		# min_obsbound = self.latitude + beam_minus #Here we're computing the size of the observing window
		# max_obsbound = self.latitude + beam_plus  #bigger beam_width or sigma_cutoff means a larger observing window 


		#print(min_obsbound*180./np.pi,max_obsbound*180./np.pi)
		# Now convert from latitude (measured from the equatorial plane up)
		# to the polar angle (measured down from the z axis)
		# Note how "max" and "min" swap places here

		# max_swap = max_obsbound
		# max_obsbound = np.pi / 2. - min_obsbound 
		# min_obsbound = np.pi / 2. - max_swap
		
		min_obsbound = min_corner[0]*(np.pi/180)
		max_obsbound = max_corner[0]*(np.pi/180)
		

		if max_corner[0]*(np.pi/180.) < min_obsbound or min_corner[0]*(np.pi/180.) > max_obsbound:
			raise ValueError('Requested Region is not observable')
		else:
			thetas = np.linspace(min_obsbound, max_obsbound, self.npix_theta) 

			phis = np.linspace((min_corner[1]*(np.pi/180.)), (max_corner[1]*(np.pi/180.)), self.npix_phi)
		
			self.position = np.concatenate(np.dstack(np.meshgrid(thetas, phis)))
			self.Npix = self.position.shape[0]	

		return self.position



	def sky_shapes(self):

		self.observable_coordinates()

		x = self.observable_coordinates()[:,1] #phi
		y = self.observable_coordinates()[:,0]#theta

		null = np.zeros(len(x))
		df_check = pd.DataFrame.from_dict(np.array([x,y,null]).T)

		df_check.columns = ['phi','theta','temp']

		pivotted_obs_check= df_check.pivot('theta','phi','temp')
		self.sky_shape = pivotted_obs_check.shape

		return self.sky_shape

	def necessary_times(self): #REUTRN self.times
		"""
		Assuming Phi = 0h at phi coordinate of observable region(self.potion), 
		Figure out time when the telescope scanning and observable region overlaps
		"""

		if self.position is not None:
			pass
		else: ### and if else about psource fg
			self.position = self.observable_coordinates() 
			
		time_length = np.abs(self.position[self.position.shape[0]-1,1]-self.position[0,1])/ (np.pi*2.)#fraction of day you observe the sky in question [day]
		self.times = np.arange(0., time_length, self.delta_t)#units here odn't make sense
		self.Nt = len(self.times) #this is the number of times the telescope makes an observation
		return self.times # in fraction of circle
	

	def rotate_bls(self): #DO NOT RETURN
		"""
		Rotates the baselines to their correct positions given
		the observation times. Results are stored in a bl-time array.
		"""

		if self.times is not None:
			pass
		else:
			self.times = self.necessary_times()
		
		# Radians rotated since beginning of observation    
		phis = (2. * np.pi * self.times) 
		
		# Define rotation matrices about z axis
		cos_phis = np.cos(phis)
		sin_phis = np.sin(phis)
		
		
		time_rot_matrices = np.zeros((self.times.shape[0], 3, 3))
		time_rot_matrices[:,-1,-1] = 1.
		for i in range(self.times.shape[0]): 
			time_rot_matrices[i,0,0] = cos_phis[i]
			time_rot_matrices[i,0,1] = -sin_phis[i]
			time_rot_matrices[i,1,0] = sin_phis[i]
			time_rot_matrices[i,1,1] = cos_phis[i]

		# Multiply baselines by rotation matrices
		# Result is a N_times x 3 x N_bls matrix
		
		self.bl_times = np.dot(time_rot_matrices, self.bls_celestial)
		self.bl_times = np.moveaxis(self.bl_times, -1, 0) # now N_bls x N_t x 3
		
		self.bl_times = np.reshape(self.bl_times, (-1,3)) #TESTED
		# now N_bls N_t x 3, where we cycle through times more quickly
		
	def convert_to_3d(self): #RETURN transformed
		"""
		Project the patch of the sky (Equatorial coord) onto 
		the baseline coordinate (3D) for calibration
		"""
		
		if self.position is not None:
			pass
		else:
			self.observable_coordinates()
		
		thetas = self.position[:,0] 
		phis = self.position[:,1]
		#print(thetas[:10]*180./np.pi,phis[:10]*180./np.pi)
		transformed = np.zeros((self.Npix, 3))
		
		for i in range(self.Npix): 
			transformed[i,0] = np.sin(thetas[i])*np.cos(phis[i])#X
			transformed[i,1] = np.sin(thetas[i])*np.sin(phis[i])#Y
			transformed[i,2] = np.cos(thetas[i])#Z
			
		return transformed 
		
	
	def compute_bdotr(self, psource_data): #DO NOT RETURN psources is a nsources x 4 array  
		"""
		Given an array of times and sky positions,
		computes an (N_times,N_pos) array
		containing b-dot r.
		"""

		if self.bl_times is not None: 
			pass
		else:
			self.rotate_bls()

		position3d = self.convert_to_3d()
		 

		if psource_data is not None: 
			self.psources= np.fromfile('psource_data.bin', dtype=np.float32)
			position3d = np.concatenate((position3d, self.psources[:,1:]), axis=0)

		else:
			pass 

		# (N_bls N_t x 3  ) (3 x npix)

		# Result is a N_t N_bl x N_pix (+ N_souces) array
		# Cycles through time more rapidly than baselines

		self.bdotr = np.dot(self.bl_times, position3d.T)

	def compute_beam(self,psource_beam): #DO NOT RETURN
		"""
		Compute Primary Beam assuming all antenna has an identical beam with the fixed sky.
		"""

		if self.times is not None:
			pass
		else:
			self.times = self.necessary_times()

		phis = (2. * np.pi * self.times) + self.position[0,1] #convert back to from times to phi, shifting back to initial phi coord
	   
		# nsources = len(psources)
		# print(nsources)

		# phi = np.concatenate((self.position[:,1],),axis = 0)

		if self.beam == 'gaussian':
			primary = np.zeros((self.Nt, (self.Npix)))
			
			co_lat = np.pi / 2. - self.latitude

			for i in range(self.Nt): #compute the elements of pbeam
				primary[i] = np.exp(-((self.position[:,1]-phis[i])**2 +(self.position[:,0]-co_lat)**2) / float(self.beam_width**2))# 2D gaussian beam (N_position,2)
				
			if psource_beam is not None: 
				# concatenate the psource primary here
				psource_primary = np.fromfile('psource_beam.bin', dtype=np.float32)
				primary = np.concatenate((primary, psource_primary), axis=1)
			else:
				pass



		else:
			raise NotImplementedError()
		
		# assume primary beam is same for all baselines
		# Want self.pbeam to have shape N_t N_bl x Npix
		# where we cycle rapidly through time

		# self.primary = primary

		self.pbeam = np.vstack([primary] * self.Nbl) #to replace the for loop pbeam
		
	
	def compute_Amat(self,psource_data, psource_beam): #DO NOT RETURN
		"""
		Compute A matrix
		"""
		if self.times is not None:
			pass
		else:   
			self.times = self.necessary_times()
		
		self.sky_shapes()
			
		self.compute_beam(psource_beam)
		self.compute_bdotr(psource_data)
		
		wavelength = sc.c/float(self.freq*1e6) # in m

		exponent = np.exp((-1j * 2 * np.pi*self.bdotr)/ float(wavelength))

		## A has shape of Nt Nbl x N_pix
		## cycling through time more rapidly
		# pix_size = self.resol ** 2

		self.delta_phi = self.position[self.sky_shape[1],1]-self.position[0,1]
		self.delta_theta = self.position[1,0]-self.position[0,0]

	
		if self.primary_beam == True: #the compute Amat with pbeam 
			
			self.Amat = exponent*self.pbeam*self.delta_theta*self.delta_phi
			
		elif self.primary_beam == False: # this computes Amat without the pbeam
			self.Amat = exponent*self.delta_phi*self.delta_theta
		else: 
			raise ValueError('You should indicate the use of a primary beam')
		
		# tau = (self.n_days*self.delta_t*self.Nt*86400)# total integration time in s 
		self.noise_rms =  self.Tsys / np.sqrt(2*self.Nt*self.n_days*self.delta_t* self.channel_width)
		self.invN = np.diag(np.repeat(self.ucounts, self.Nt)) * (1/float((self.noise_rms)**2)) #Nt N_unique_bls x Nt N_unique_bls diagonal array

	def compute_general_A(self,psource_data,psource_beam):

		"""
		Compute A matrix
		"""
		if self.times is not None:
			pass
		else:   
			self.times = self.necessary_times()
		
		self.sky_shapes()
			
		self.compute_beam(psource_beam)
		self.compute_bdotr(psource_data)
		
		exponent = (-1j * 2 * np.pi*self.bdotr)/(3e8)
		## A has shape of Nt Nbl x N_pix
		## cycling through time more rapidly
		# pix_size = self.resol ** 2

		self.delta_phi = self.position[self.sky_shape[1],1]-self.position[0,1]
		self.delta_theta = self.position[1,0]-self.position[0,0]

	
		if self.primary_beam == True: #the compute Amat with pbeam 
			
			self.A_exp = exponent
			self.A_beam = self.pbeam*self.delta_theta*self.delta_phi

		elif self.primary_beam == False: # this computes Amat without the pbeam
			self.Amat_exponentg = exponent
			self.A_beam = self.delta_phi*self.delta_theta
		else: 
			raise ValueError('You should indicate the use of a primary beam')
		
		# tau = (self.n_days*self.delta_t*self.Nt*86400)# total integration time in s 
		self.noise_rms_general =  1 / np.sqrt(2*self.Nt*self.n_days*self.delta_t* self.channel_width)
		self.invN_general = np.repeat(self.ucounts, self.Nt) * (1/float((self.noise_rms_general)**2)) #Nt N_unique_bls x Nt N_unique_bls diagonal array


	

	def compute_vis(self,vec,psource_data, psource_beam): #DO NOT RETURN
		"""
		Compute visibility from given vector
		"""
		self.compute_Amat(psource_data, psource_beam)

		if psource_data is not None: 
			vec = np.concatenate((vec,self.psources[:,0]), axis = 0) 

		else: 
			pass
		

		self.Adotx = np.dot(self.Amat, vec) 

	def compute_normalization(self,psource_data, psource_beam): #DO NOT RETURN

		 ## compute [A^*t N^{-1} A]^{-1}
		self.compute_Amat(psource_data, psource_beam)
						  
		AtA = ((np.conj(self.Amat)).T).dot(self.invN).dot(self.Amat) 
		
		# print(self.Amat.shape)
		#here we are setting all the non-diagonal elements of AtA to 0
		
		diagAtA = np.diag(AtA) #pick out the main diagonal
	
		matrix_diagAtA = np.diag(diagAtA) #make that diagonal into a diagonal matrix

		self.norm = la.inv(matrix_diagAtA) #take the diagonal matrix and take the inverse 
		# print(self.norm.shape)

	def generate_map_noise(self,psource_data, psource_beam):#DO NOT RETURN
		"""
		Draw Gaussian random white noise from noise rms
		Returns to normalized noise prediction
		"""

		if self.norm is not None:
			pass
		else:
			self.compute_normalization(psource_data, psource_beam)
				   

		self.my_noise = np.random.normal(0, 0.091704138,self.Nt*self.Nbl) + np.random.normal(0,0.091704138,self.Nt*self.Nbl)*(1j)
		if self.normalization == True: #uses the normalization in the estimator


			self.noise = np.dot(self.norm,(np.conj(self.Amat)).T).dot(self.invN).dot(self.my_noise)

			return self.noise

		else: 

			self.noise = (np.conj(self.Amat)).T.dot(self.invN).dot(self.my_noise)

			return self.noise

	
	def single_pix_convolve_map(self,pix,vec,psource_data, psource_beam):

		# start = timer()

		if self.norm is not None:
			pass
		else:
			self.compute_normalization(psource_data, psource_beam)


		self.compute_vis(vec,psource_data,psource_beam)

		if self.normalization == True: 
			self.map = (((np.conj(self.Amat)).T)[pix]).dot(self.invN).dot(self.Adotx)

		else: 
		
			self.map = (((np.conj(self.Amat)).T)[pix]).dot(self.invN).dot(self.Adotx)

			return self.map
		# end = timer()
		# runtime = end-start
		# return runtime 

	def compute_M(self,psource_data,psource_beam):

		if self.norm is not None:
			pass 
		else: 
			self.compute_normalization(psource_data,psource_beam)

		self.compute_Amat(psource_data,psource_beam)

		if self.normalization == True:
			self.Mmat = np.dot(self.norm,((np.conj(self.Amat)).T).dot(self.invN).dot(self.Amat))

		else:
			self.Mmat = ((np.conj(self.Amat)).T).dot(self.invN).dot(self.Amat)

			



			# print(self.norm[1])
	
	def convolve_map(self,vec,psource_data,psource_beam ): #DO NOT RETURN
		
		
		"""
		Normalized Sky Prediction
		"""

		#start timine here 
		#start = timer()

		if self.norm is not None:
			pass
		else:
			self.compute_normalization(psource_data, psource_beam)

		self.compute_vis(vec,psource_data, psource_beam)
	
		
		if self.normalization == True: #uses the normalization in the estimator

			
			self.map = np.dot(self.norm,((np.conj(self.Amat)).T).dot(self.invN).dot(self.Adotx))
			
			#end timer here
			# return self.map 
			# end = timer()
			# runtime = end-start
			# return runtime
			
		else: #leaves out the normalization 


			self.map = ((np.conj(self.Amat)).T).dot(self.invN).dot(self.Adotx)
		
			# return self.map
			# end = timer()
			# runtime = end-start
			# return runtime 
			

		