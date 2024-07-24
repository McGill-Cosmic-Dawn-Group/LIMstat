import numpy as np
import matplotlib.pyplot as plt


class Power_Spectrum(object):
    """docstring for Power_Spectrum
"""
    def __init__(self, data = None , Ly = None, Lx = None, nbins = None, kmax = None):
        
        self.data = data #box/cube data set
        self.Lx = Lx # physical length scale of data.shape[0] side
        self.Ly = Ly # physical length scale of data.shape[1] side
        self.nbins = nbins # number of bins, 
        # self.log = log #indicate whether to use log spacing (to be implemented later)
        self.row_npix = self.data.shape[0]
        self.col_npix = self.data.shape[1]
        self.cutoff_k = kmax


        #these two lines give you the physical dimensions of a pixel (inverse of sampling ratealong each axis)
        self.delta_y = (self.Ly/self.data.shape[0]) # size of y pixel
        self.delta_x = (self.Lx/self.data.shape[1]) # size of x pixel

        self.delta_ky = (2*np.pi)/self.Ly
        self.delta_kx = (2*np.pi)/self.Lx


        #ADD if len shape is 3 then add delta z

    def cosmo_FFT2(self): 
        ''' computes the fourier transform of a 2D field of mean 0'''
        
        self.data = self.data - np.mean(self.data)

        self.fft_data = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.data* (self.delta_x*self.delta_y))))# [mk mpc^2]
        self.ps_data = (np.conj(self.fft_data))*self.fft_data # [mk^2 mpc^4]

    def compute_k_2D(self):

        self.kx = np.fft.fftshift(np.fft.fftfreq(self.col_npix, d = self.delta_x)) 
        self.ky = np.fft.fftshift(np.fft.fftfreq(self.row_npix, d = self.delta_y))

        self.ky *= 2*np.pi
        self.kx *= 2*np.pi


        k = []

        for i in range(len(self.ky)): 
            for j in range(len(self.kx)):
                k.append(np.sqrt(self.kx[j]**2 + self.ky[i]**2)) 

        self.k = np.asarray(k)



    def compute_kbox(self):
        self.compute_k_2D()

        self.kbox = np.reshape(self.k,(self.row_npix,self.col_npix)) 


    def compute_pspec(self):

        self.cosmo_FFT2() 
        self.compute_k_2D()

        if self.cutoff_k == None: 
            bin_edges = np.histogram_bin_edges(self.k, bins = self.nbins)
            print(bin_edges)
        else: 
            bin_edges = np.linspace(min(self.k),self.cutoff_k,self.nbins+1)

        delta_k = bin_edges[1]-bin_edges[0]

        self.kmodes = bin_edges[:self.nbins]+(0.5*delta_k)

        a = np.zeros(len(bin_edges)-1) #holds real stuff..here you need to take the number of BINS not bin edges! # you alwaysneed an extra edge than you have bin!

        #c holds, in each element, the number of pixels 
        c = np.zeros_like(a)

        for i in range(self.data.shape[0]) : 
            for j in range(self.data.shape[1]):
                kx = ((j-(self.data.shape[0]/2))*self.delta_kx)#need to multiply by kdelta to get your k units
                ky = ((i-(self.data.shape[1]/2))*self.delta_ky)
                kmag = np.sqrt((kx**2) + (ky**2))
                for k in range(len(bin_edges)-1): #make sure that you speed this up by not considering already binned ps's
                    if bin_edges[k] < kmag <= bin_edges[k+1]: 
                        a[k] += np.real(self.ps_data[j,i])
                        c[k] += 1
                        break
        
        arg = np.argwhere(np.isnan(a)), np.where(c == 0) # Make sure there are no nans! If there are make them zeros. Also make sure you never divide by 0!
        if len(arg) > 0:
            for i in range(len(arg)):
                a[arg[i]] = 0
                c[arg[i]]=1
        else:
            pass
       

        T_tilde = a/c

        volume = self.Lx*self.Ly 

        self.pk = T_tilde/volume #[mk^2*Mpc^2]

        return self.kmodes,self.pk

    def compute_dimensionless_2D_pspec(self):

        self.compute_2D_pspec()

        self.power =  ((self.kmodes**2) * self.pk)/(2*np.pi) #[mk^2]

        return self.kmodes, self.power




