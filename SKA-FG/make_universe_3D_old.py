import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math


class universe(object):
	"""docstring for ClassName"""
	def __init__(self, ps, row_npix,col_npix,aisle_npix, Ly,Lx,Lz, nbins, z_mid):

		
		self.z_mid = z_mid
		self.nbins = nbins

		if isinstance(ps,np.ndarray) or isinstance(ps,tuple):
			#here you interpolate


			k_theory = ps[0]
			p_theory = ps[1] 


			self.ps = interp1d(k_theory, p_theory, fill_value="extrapolate")

		else:
			self.ps = ps #[mk^2*Mpc^3]

		self.row_npix = row_npix
		self.col_npix = col_npix
		self.aisle_npix = aisle_npix
		self.Lx = Lx
		self.Ly = Ly
		self.Lz = Lz

		self.delta_y = self.Ly/np.float(self.row_npix) #sampling rate
		self.delta_x = self.Lx/np.float(self.col_npix) #sampling rate 
		self.delta_z = self.Lz/np.float(self.aisle_npix)

		print(self.Lx,self.Ly,self.Lz)

		self.delta_ky = (2*np.pi)/np.float(self.Ly)
		self.delta_kx = (2*np.pi)/np.float(self.Lx)
		self.delta_kz = (2*np.pi)/np.float(self.Lz)


	def compute_k(self):

	
		self.kx = np.fft.fftshift(np.fft.fftfreq(self.col_npix, d = self.delta_x)) 
		self.ky = np.fft.fftshift(np.fft.fftfreq(self.row_npix, d = self.delta_y))
		self.kz = np.fft.fftshift(np.fft.fftfreq(self.aisle_npix, d = self.delta_z))


		self.ky *= -2*np.pi
		self.kx *= -2*np.pi
		self.kz *= -2*np.pi


		k = []

		#Careful! need to fix a y and cycle through x to get the right concentric rings in your kbox!

		for i in range(len(self.ky)): 
			for j in range(len(self.kx)):
				for z in range(len(self.kz)):
					k.append(np.sqrt(self.kx[j]**2 + self.ky[i]**2 + self.kz[z]**2)) 


		self.k = np.asarray(k)
		
		self.ksorted= np.asarray(sorted(self.k))

		
	def compute_theory_spec(self):

		self.compute_k_2D()

		self.theory_spec =  self.ps(self.ksorted)

	def compute_array_populated_box(self): 

		'''  need to make a method for if the theory psepc in an array, you interpolate to find the k's'''

	def compute_kbox(self):
		self.compute_k()

		self.kbox = np.reshape(self.k,(self.row_npix,self.col_npix,self.aisle_npix)) 

	
	def make_universe(self):

		#this box will hold all of the values of k for each pixel
		
		self.compute_kbox()
		#here's the box that will hold the random gaussian things

		fft_box = np.zeros_like(self.kbox, dtype = complex)
	
		for i in range(self.row_npix):
			for j in range(self.col_npix): 
				for z in range(self.aisle_npix):
					if self.kbox[i,j,z] == 0:
						a = np.random.normal(0,np.sqrt((self.ps(0.000000001)*self.Lx*self.Ly*self.Lz)/2))
						b = np.random.normal(0,np.sqrt((self.ps(0.000000001)*self.Lx*self.Ly*self.Lz)/2)) # [mk Mpc^3]
						fft_box[i,j,z] = np.complex(a,b)
						pass
					else:
						a = np.random.normal(0,np.sqrt((self.ps(self.kbox[i,j,z])*self.Lx*self.Ly*self.Lz)/2))
						b = np.random.normal(0,np.sqrt((self.ps(self.kbox[i,j,z])*self.Lx*self.Ly*self.Lz)/2)) # [mk Mpc^3]
						fft_box[i,j,z] = np.complex(a,b)

######## DO THE DECORR ########

		bin_edges = np.histogram_bin_edges(self.ksorted, bins = self.nbins)

		half_bin = (bin_edges[1]-bin_edges[0])/2


		self.kmodes = bin_edges[:self.nbins]#bin_edges[:self.nbins]+half_delta_bin


	######### IMPOSE SYMMETRY CONDITIONS SO THAT IFT IS REAL #########

		for i in range(self.row_npix):
			for j in range(self.col_npix):
				for z in range(self.aisle_npix):

					if self.row_npix % 2 == 0: # if rows even 
						mirror_i = (self.row_npix - i) % self.row_npix
					else: # if rows odd
						mirror_i = (self.row_npix-1)-i

					if self.col_npix % 2 == 0: #if cols even
						mirror_j = (self.col_npix - j ) % self.col_npix
					else: #if cols odd
						mirror_j = (self.col_npix-1)-j

					if self.aisle_npix % 2 == 0: #if cols even
						mirror_z = (self.aisle_npix - z ) % self.aisle_npix
					else: #if cols odd
						mirror_z = (self.aisle_npix-1)-z

					if mirror_i == i and mirror_j == j and mirror_z == z: 
						# must be real
						fft_box[i,j,z] = np.real(fft_box[i,j,z]) #only number equal to its own complex conjugate are real numbers
					else:
					# copy complex conjugate from mirror coordinates to current coord
						fft_box[i,j,z] = np.conjugate(fft_box[mirror_i,mirror_j,mirror_z])

	########################################################################
		
		self.u = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(fft_box * (self.delta_ky*self.delta_kx*self.delta_kz)))) #[mk]


		self.u /= (2.*np.pi)**3

		
		self.universe = (np.real(self.u)) - np.mean(np.real(self.u))

		# return self.universe, self.universe_decorr