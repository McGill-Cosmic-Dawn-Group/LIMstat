import numpy as np
from scipy.interpolate import interp1d


class universe(object):
    """docstring for ClassName"""

    def __init__(
        self,
        ps,
        row_npix,
        col_npix,
        aisle_npix,
        Ly,
        Lx,
        Lz,
        z_mid,
        verbose=False,
    ):

        self.z_mid = z_mid

        if isinstance(ps, np.ndarray) or isinstance(ps, tuple):
            # here you interpolate

            k_theory = ps[0]
            p_theory = ps[1]

            self.ps = interp1d(k_theory, p_theory, fill_value="extrapolate")

        else:
            self.ps = ps  # [mk^2*Mpc^3]

        self.row_npix = row_npix
        self.col_npix = col_npix
        self.aisle_npix = aisle_npix
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        self.delta_y = self.Ly / np.float(self.row_npix)  # sampling rate
        self.delta_x = self.Lx / np.float(self.col_npix)  # sampling rate
        self.delta_z = self.Lz / np.float(self.aisle_npix)
        # print(self.Lx,self.Ly,self.Lz)

        self.delta_ky = (2 * np.pi) / np.float(self.Ly)
        self.delta_kx = (2 * np.pi) / np.float(self.Lx)
        self.delta_kz = (2 * np.pi) / np.float(self.Lz)

        self.verbose = verbose

    def compute_k(self):

        kx = (2. * np.pi) * np.fft.fftshift(
            np.fft.fftfreq(self.col_npix, d=self.delta_x)
        )
        ky = (2. * np.pi) * np.fft.fftshift(
            np.fft.fftfreq(self.row_npix, d=self.delta_y)
        )
        kz = (2. * np.pi) * np.fft.fftshift(
            np.fft.fftfreq(self.aisle_npix, d=self.delta_z)
        )

        return kx, ky, kz

    def compute_theory_spec(self):

        self.compute_k_2D()

        self.theory_spec = self.ps(self.ksorted)

    def compute_array_populated_box(self):

        """
        need to make a method for if the theory psepc in an array,
        you interpolate to find the k's
        """

    def compute_kbox(self):
        kx, ky, kz = self.compute_k()
        kbox = np.sqrt(
            kx[:, None, None] ** 2
            + ky[None, :, None] ** 2
            + kz[None, None, :] ** 2

        )

        return kbox

    def make_universe(self, **ps_kwargs):

        # this box will hold all of the values of k for each pixel
        if self.verbose:
            print("Getting k box...")
        kbox = self.compute_kbox()  # Mpc-1
        # store ps values along kbox
        powerbox = np.zeros(kbox.shape)
        powerbox[kbox != 0] = self.ps(kbox[kbox != 0], **ps_kwargs)  # mk2 Mpc3
        powerbox *= self.Lx * self.Ly * self.Lz  # Mpc3 * mk2 Mpc3 = mk2 mpc6 

        # here's the box that will hold the random gaussian things
        if self.verbose:
            print("Generating fft box")
        means = np.zeros(kbox.shape)
        widths = np.sqrt(powerbox*0.5) # sqrt(mk2 mpc6)= mk mpc3
        a, b = np.random.normal(
            means,
            widths,
            size=(2, self.row_npix, self.col_npix, self.aisle_npix)
        )  # Mpc3


        if self.verbose:
            print("Taking iFFT of box")
        dk = (self.delta_kx*self.delta_ky*self.delta_kz)
        u = np.fft.fftshift(
            np.fft.irfftn(np.fft.ifftshift((a + b * 1j)*dk),
                          s=(kbox.shape), 
                          norm= 'forward'))  # [mK]
        
        u /= ((2*np.pi)**3)

        universe = (np.real(u))

        print('mean',np.mean(np.real(u)))

        return universe
