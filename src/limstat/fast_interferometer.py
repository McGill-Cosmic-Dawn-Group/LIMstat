import numpy as np
from astropy import units
from scipy import stats
import scipy.interpolate 
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


    def get_bls(self, freq):
        #GOOD KEEP THIS
        '''Get the unique baselines from the antenna locations and calculate the uv coordinates.
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
     
        total_bls = np.concatenate((bls,-bls)) #this is the total number of baselines (including in the negative direction)

        #make sure -0 = 0
        total_bls = np.where(total_bls == 0, 0, total_bls)
        #check that the total number of baselines is correct
        if len(total_bls) != ((n_ants * (n_ants - 1))):
            raise ValueError('The total number of baselines is not correct.')
        else:
            pass

        self.unique_bls, self.counts = np.unique(total_bls, axis=0, return_counts = True) #this is the list of unique baselines

        frequency = freq.to(units.Hz).value
        lambda_ = 3e8 / frequency

        self.u_coords = self.unique_bls[:,0]/lambda_
        self.v_coords = self.unique_bls[:,1]/lambda_

        return self.unique_bls

    def get_uvmap_halfwave(self, freq):
        # GOOD KEEP THIS
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
        self.get_bls(freq)
        self.du = 0.5
        self.dv = 0.5
        self.u_grid = np.arange(-self.u_coords.max(),self.u_coords.max()+self.du,self.du)
        self.v_grid = np.arange(-self.v_coords.max(),self.v_coords.max()+self.dv,self.dv)
        
        length_v = len(self.v_grid)
        length_u = len(self.u_grid)
        
        if length_u <=1:
            self.array_layout = "ns_only"
        elif length_v <=1:
            self.array_layout = "ew_only"
        else:
            self.array_layout = "2d"
            
        if self.array_layout == "2d":
            binned_uv = stats.binned_statistic_2d(self.v_coords,self.u_coords, np.ones((len(self.unique_bls[:,0]))),
                                            statistic='mean',
                                            bins=[self.v_grid, self.u_grid])

            binned_counts = stats.binned_statistic_2d(self.v_coords,self.u_coords, self.counts,
                                            statistic='sum',
                                            bins=[self.v_grid, self.u_grid])
            
            # get the number of measurments per uv bin (this is from redundant baselines)
            self.count_map = binned_counts.statistic
            #set all nans to 0 
            self.count_map[np.isnan(binned_counts.statistic)] = 0

            uv_map = binned_uv.statistic
            #set all nans to 0 
            uv_map[np.isnan(binned_uv.statistic)] = 0
            
            print(uv_map.shape)

        if self.array_layout == "ns_only":
            #make the binned uv map a 2D array using the u from sky_coords
            u,_,_,_,_,_, = self.sky_uv_coords()
        
            binned_uv = stats.binned_statistic_2d(self.v_coords,self.u_coords, np.ones((len(self.unique_bls[:,0]))),
                                statistic='mean',
                                bins=[self.v_grid, u])
            
            binned_counts = stats.binned_statistic_2d(self.v_coords,self.u_coords, self.counts,
                                            statistic='sum',
                                            bins=[self.v_grid, u])
            
            # get the number of measurments per uv bin (this is from redundant baselines)
            self.count_map = binned_counts.statistic
            #set all nans to 0 
            self.count_map[np.isnan(binned_counts.statistic)] = 0

            uv_map = binned_uv.statistic
            #set all nans to 0 
            uv_map[np.isnan(binned_uv.statistic)] = 0

        # TODO: need to figure out transposes and whatnot. 
        elif self.array_layout == "ew_only":
            #make the binned uv map a 2D array using the v from sky_coords
            _,v,_,_,_,_, = self.sky_uv_coords()
            binned_uv = stats.binned_statistic_2d(self.v_coords,self.u_coords, np.ones((len(self.unique_bls[:,0]))),
                                statistic='mean',
                                bins=[v, self.u_grid])
            
            binned_counts = stats.binned_statistic_2d(self.v_coords,self.u_coords, self.counts,
                                            statistic='sum',
                                            bins=[v, self.u_grid])
            
            # get the number of measurments per uv bin (this is from redundant baselines)
            self.count_map = binned_counts.statistic
            #set all nans to 0 
            self.count_map[np.isnan(binned_counts.statistic)] = 0

            uv_map = binned_uv.statistic
            #set all nans to 0 
            uv_map[np.isnan(binned_uv.statistic)] = 0

        else:
            pass
        
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
        uv_map = self.get_uvmap_halfwave(freq)
        npix_0 = uv_map.shape[0]
        npix_1 = uv_map.shape[1]
        psf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uv_map,axes=(0,1))))*(self.du*self.dv* npix_0* npix_1)

        psf = self.interp_dirty_map_to_sky(psf)
        
        return psf.real

    def sky_uv_coords(self):
        #GOOD KEEP THIS
        '''Get the uv coordinates of the sky map.
        Parameters
        ----------
        sky_map : array_like
            The sky map to be observed in K.
        Returns
        -------
        uv_coords : array_like
            The uv coordinates of the sky map.
        '''
        L = np.sin(self.theta_x)
        M = np.sin(self.theta_y)

        dl = L / self.x_npix
        dm = M / self.y_npix

        l = np.arange(-L/2, L/2, dl)
        m = np.arange(-M/2, M/2, dm)

        u = np.fft.fftshift(np.fft.fftfreq(len(l), d=dl))
        v = np.fft.fftshift(np.fft.fftfreq(len(m), d=dm))

        return u, v, l,m, dl, dm

    def sky_fft_to_uv(self,sky_map):
        #GOOD KEEP THIS
        """FFT sky_map to UV; same scaling as get_dirty_map."""

        _,_,_,_, dl, dm = self.sky_uv_coords()

        sky_fft = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(sky_map, axes=(0, 1)))
        ) * (dl * dm)

        return sky_fft

    def interp_sky_fft_to_halfwave(self, sky_map, freq, method="linear", fill_value=0.0):
        #GOOD KEEP THIS
        """
        Interpolate model visibilities onto the half-wave grid used by get_uvmap_halfwave.
        Uses self.u_grid and self.v_grid (set by get_uvmap_halfwave). Target (u,v)
        are bin centres so V_hw matches uv_map.shape.
        """
        uv_map = self.get_uvmap_halfwave(freq)  # sets self.u_grid, self.v_grid, uv_map
        sky_fft = self.sky_fft_to_uv(sky_map)
        u_nat, v_nat, _, _, _,_= self.sky_uv_coords()

        # Targets tied to self.u_grid / self.v_grid (same bins as uv_map)
        if self.array_layout == "2d":
            self.u_hw = 0.5 * (self.u_grid[:-1] + self.u_grid[1:])
            self.v_hw = 0.5 * (self.v_grid[:-1] + self.v_grid[1:])
        
        elif self.array_layout == "ns_only":
            self.u_hw = u_nat[:-1]
            self.v_hw = 0.5 * (self.v_grid[:-1] + self.v_grid[1:])
        elif self.array_layout == "ew_only":
            self.u_hw = 0.5 * (self.u_grid[:-1] + self.u_grid[1:])
            self.v_hw = v_nat[:-1]
        else:
            pass

        interp_re = scipy.interpolate.RegularGridInterpolator(
            (v_nat, u_nat),
            sky_fft.real,
            method=method,
            bounds_error=False,
            fill_value=fill_value,
        )
        interp_im = scipy.interpolate.RegularGridInterpolator(
            (v_nat, u_nat),
            sky_fft.imag,
            method=method,
            bounds_error=False,
            fill_value=fill_value,
        )
        VV, UU = np.meshgrid(self.v_hw, self.u_hw, indexing="ij")
        pts = np.column_stack([VV.ravel(), UU.ravel()])
        V_hw = (interp_re(pts) + 1j * interp_im(pts)).reshape(len(self.v_hw), len(self.u_hw))
        
        return V_hw, uv_map

    def interp_dirty_map_to_sky(self, dirty_map):
        '''Interp dirty uv to sky grid.
        Parameters
        ----------
        dirty_uv : array_like
            The dirty uv to be interpolated to the sky grid.
        Returns
        -------
        sky_map : array_like
            The sky map in K.
        '''
        #Target l , m grid (i.e. sky grid)
        _, _, l_sky, m_sky, _, _ = self.sky_uv_coords()

        #dual to hw grid 
        l_hw = np.fft.fftshift(np.fft.fftfreq(len(self.u_hw), d=self.du))
        m_hw = np.fft.fftshift(np.fft.fftfreq(len(self.v_hw), d=self.dv))

        #interp dirty uv to sky grid
        interp = scipy.interpolate.RegularGridInterpolator(
            (m_hw, l_hw),   # v then u / row then col — match dirty_hw axes
            dirty_map,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        MM, LL = np.meshgrid(m_sky, l_sky, indexing="ij")
        pts = np.column_stack([MM.ravel(), LL.ravel()])

        nx = len(l_sky)
        ny = len(m_sky)

        sky_map = interp(pts).reshape(ny, nx)

        return sky_map
    
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
        V_hw, uv_map = self.interp_sky_fft_to_halfwave(sky_map, freq)

        dirty_uv = np.multiply(uv_map, V_hw)
        npix_0 = dirty_uv.shape[0]
        npix_1 = dirty_uv.shape[1]
        if noise:
            noise = self.compute_noise()
        
            if redundancy: 
                redundant_noise = np.where(self.count_map !=0, noise/np.sqrt(self.count_map),0)
                a = np.random.normal(0, redundant_noise, (npix_0, npix_1))
                b = np.random.normal(0, redundant_noise, (npix_0, npix_1))
            else:
                a = np.random.normal(0, noise, (npix_0, npix_1))
                b = np.random.normal(0, noise, (npix_0, npix_1))
            
            noise_map = a + (1j*b)
            noise_map = np.where(uv_map != 0, noise_map,0)

            dirty_uv += noise_map
        else: 
            pass 
        #fft factor
        npix_interp = len(V_hw[1]) * len(V_hw[0])
        dirty_map = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(dirty_uv,axes=(0,1)))) * self.du*self.dv* npix_interp
        ## Need to then crop/interp to the old sky grid.
        dirty_map = self.interp_dirty_map_to_sky(dirty_map)
        
        return dirty_map.real
    
    def get_noise_map(self, freq, *args, redundancy = True, **kwargs):
        uv_map = self.get_uvmap_halfwave(freq)
        noise = self.compute_noise()
        npix_0 = uv_map.shape[0]
        npix_1 = uv_map.shape[1]

        if redundancy:
            redundant_noise = np.where(self.count_map !=0, noise/np.sqrt(self.count_map),0)
            a = np.random.normal(0, redundant_noise, (npix_0, npix_1))
            b = np.random.normal(0, redundant_noise, (npix_0, npix_1))
        else:
            a = np.random.normal(0, noise, (npix_0, npix_1))
            b = np.random.normal(0, noise, (npix_0, npix_1))

        noise_map = (a + (1j*b))/np.sqrt(2)
      
        position_noise_map = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(noise_map,axes=(0,1)))) * self.du*self.dv * npix_0* npix_1
        position_noise_map = self.interp_dirty_map_to_sky(position_noise_map)
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

