import numpy as np
from astropy import units
from scipy import stats

class fast_interferometer(object):

    ''' A class to simulate the output of an interferometer.'''

    def __init__(self,
                 ant_locs, 
                 theta_x,
                 theta_y,
                 x_npix,
                 y_npix,
                 T_sys = None,
                 t_obs = None,
                 bandwidth = None,
                 ):

        self.ants = ant_locs.to(units.m).value
        self.theta_x = theta_x.to(units.rad).value
        self.theta_y = theta_y.to(units.rad).value
        self.x_npix = x_npix
        self.y_npix = y_npix
            
        if T_sys is not None:
            self.T_sys = T_sys.to(units.K).value
        if t_obs is not None:
            self.t_obs = t_obs.to(units.s).value
        if bandwidth is not None:
            self.bandwidth = bandwidth.to(units.Hz).value

        # if RA_patch is not None:
        #     hours_per_day = (RA_patch.to(units.deg).value/15.2) * units.h #earth rotates 15.2 deg per hour
        #     self.t_obs = (hours_per_day * Ndays).to(units.s) #t_obs converted to seconds

        """ 
        Initialize the class with the telescope specifications.

        Parameters
        ----------
        ant_locs : array_like
            The coordinates for the locations of the antennas in meters. The shape should be (n_ants, 2). 
        theta_x : float
            The field of view in the x direction in degrees. You need to tack on astropy units to this variable.
        theta_y : float
            The field of view in the y direction in degrees.  You need to tack on astropy units to this variable.
        x_npix : int
            The number of pixels you want in the final uv grid in the x direction.
        y_npix : int
            The number of pixels you want in the final uv grid in the y direction.
        T_sys : array_like
            The system temperature in K. If None, then noiseless observation.
        t_obs : float
            The observation time in hours. If None, then noiseless observation.
        """


    def get_bls(self):
        '''Get the unique baselines from the antenna locations.
        Returns
        -------
        Nothing :) '''

        n_ants = self.ants.shape[0]
        n_bls = int(n_ants * (n_ants - 1) / 2) # this is the number of baselines

        bls = np.zeros((n_bls,2)) # initialize a list holding the length of all the baselines 
        k = 0 #initialize this k variable
        for i in range(n_ants):
            ant_i = self.ants[i]
            for j in range(i+1, n_ants):
                ant_j = self.ants[j]
                bls[k] = ant_i - ant_j # this subtracts each coordinate from the other [0,0]-[1,1]
                k += 1 #add k every time you identify a baseline 
     
        total_bls = np.concatenate((bls,-bls)) #this is the total number of baselines
        self.unique_bls, self.counts = np.unique(total_bls,axis=0, return_counts = True) #this is the list of unique baselines
        return self.unique_bls
    def get_uvmap(self, freq):
        '''Get the uv map of the interferometer. This does NOT do rotation synthesis, it is the instantaneous uv coverage.
        Parameters
        ----------
        freq : float
            The frequency of observation in MHz, GHz, etc...
        Returns
        -------
        uv_map : array_like
            The uv coverage of the interferometer.
        '''
        self.get_bls()

        frequency = freq.to(units.Hz).value
        lambda_ = 3e8 / frequency

        L = np.sin(self.theta_x)
        M = np.sin(self.theta_y)

        u = np.fft.fftshift(np.fft.fftfreq(self.x_npix+1, d=L/(self.x_npix+1)))
        v = np.fft.fftshift(np.fft.fftfreq(self.y_npix+1, d=M/(self.y_npix+1)))

        self.dl = L / self.x_npix
        self.dm = M / self.y_npix

        self.du = u[1] - u[0]
        self.dv = v[1] - v[0]
    
        u_coords,v_coords = self.unique_bls[:,0]/lambda_ , self.unique_bls[:,1]/lambda_

        binned_uv = stats.binned_statistic_2d(v_coords,u_coords, np.ones((len(self.unique_bls[:,0]))),
                                        statistic='mean',
                                        bins=[v, u])
        
        binned_counts = stats.binned_statistic_2d(v_coords,u_coords, self.counts,
                                        statistic='sum',
                                        bins=[v, u])
        
        # get the number of measurments per uv bin (this is from redundant baselines)
        self.count_map = binned_counts.statistic
        #set all nans to 0 
        self.count_map[np.isnan(binned_counts.statistic)] = 0

        uv_map = binned_uv.statistic
        #set all nans to 0 
        uv_map[np.isnan(binned_uv.statistic)] = 0

        return uv_map

    def get_psf(self, freq):
        '''Get the point spread function of the interferometer.
        Parameters
        ----------
        freq : float
            The frequency of observation in MHz, GHz, etc...
        Returns
        -------
        psf : array_like
            The point spread function of the interferometer.
        '''
        uv_map = self.get_uvmap(freq)
        psf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uv_map,axes=(0,1))))*(self.du*self.dv)
        
        return psf
    
    def get_dirty_map(self, sky_map, freq,*args, noise = False, redundancy = True, **kwargs):
        '''Get the dirty map of the sky.
        Parameters
        ----------
        sky_map : array_like
            The sky map to be observed in K.
        freq : float
            The frequency of observation in MHz, GHz, etc...
        
        Returns
        -------
        dirty_map : array_like
            The dirty map of the sky in K.
        '''

        uv_map = self.get_uvmap(freq)    
        sky_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky_map,axes=(0,1)))) * (self.dl*self.dm) #fft of sky map
        dirty_uv = np.multiply(uv_map, sky_fft)

        if noise:
            noise = self.compute_noise()
        
            if redundancy: 
                redundant_noise = np.where(self.count_map !=0, noise/np.sqrt(self.count_map),0)
                a = np.random.normal(0, redundant_noise, (self.x_npix, self.y_npix))
                b = np.random.normal(0, redundant_noise, (self.x_npix, self.y_npix))
            else:
                a = np.random.normal(0, noise, (self.x_npix, self.y_npix))
                b = np.random.normal(0, noise, (self.x_npix, self.y_npix))
            
            noise_map = a + (1j*b)
            noise_map = np.where(uv_map != 0, noise_map,0)

            dirty_uv += noise_map
        else: 
            pass 

        dirty_map = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(dirty_uv,axes=(0,1)))) * self.du*self.dv* (self.x_npix*self.y_npix)
        return dirty_map.real
    
    def get_noise_map(self, freq, *args, redundancy = True, **kwargs):
        self.get_uvmap(freq)
        noise = self.compute_noise()

        if redundancy:
            redundant_noise = np.where(self.count_map !=0, noise/np.sqrt(self.count_map),0)
            a = np.random.normal(0, redundant_noise, (self.x_npix, self.y_npix))
            b = np.random.normal(0, redundant_noise, (self.x_npix, self.y_npix))
        else:
            a = np.random.normal(0, noise, (self.x_npix, self.y_npix))
            b = np.random.normal(0, noise, (self.x_npix, self.y_npix))

        noise_map = (a + (1j*b))/np.sqrt(2)
      
        position_noise_map = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(noise_map,axes=(0,1)))) * self.du*self.dv * (self.x_npix*self.y_npix)

        return position_noise_map.real 

    def compute_noise(self):
        '''Compute the noise variance of the instrument.
        Returns
        -------
        noise : float
            The standard deviation of the instrument noise.'''
        if self.T_sys is None or self.t_obs is None:
            raise ValueError('T_sys and t_obs must be set to compute noise.')
       
        try:
            Tsys = self.T_sys.value
        except AttributeError:
            Tsys = self.T_sys
        solid_angle = self.theta_x * self.theta_y
        noise = (Tsys *solid_angle)/(np.sqrt(self.bandwidth * self.t_obs)) 
        return noise

