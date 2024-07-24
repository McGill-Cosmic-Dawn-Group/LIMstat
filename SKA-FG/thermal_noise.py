import numpy as np
from astropy import units
import warnings


class thermal_noise:

    def __init__(
        self,
        npix,
        ang_res,
        freqs,
        nb_antennas,
        integration_time,
        Tsys=None,
    ):

        """Container for thermal noise generator.

        Parameters
        ----------
            npix: int
                Number of pixels in one direction on the sky plane.
            ang_res: float
                Angular resolution of the instrument.
                Must be given with units or arcsec are assumed.
            freqs: list or array of floats
                Frequencies along the spectral window considered.
                Must be given with units or Hz are assumed.
            nb_antennas: int
                Number of antennas.
            integration_time: float
                Total integration time in sec.
            Tsys: float
                System temperature in K.
                Default is None: using Haslam+2008 foreground temperature.

        """

        # Parameters of observation
        # Check frequency units
        freqs = check_unit(freqs, units.Hz, 'frequencies')
        self.freqs = np.atleast_1d(freqs)
        self.freq_res = np.diff(freqs).mean()
        self.nfreqs = self.freqs.size

        # Check angular resolution units
        ang_res = check_unit(ang_res, units.arcsec, 'ang_res')
        self.ang_res = ang_res

        self.npix = int(npix)

        # Telescope parameters
        self.nb_antennas = int(nb_antennas)
        self.nb_baselines = self.nb_antennas * (self.nb_antennas - 1)
        self.integration_time = check_unit(integration_time, units.second, 'integration time')
        if Tsys is None:
            self.Tsys = 180. * units.K * (freqs.to(units.MHz).value / 180.)**(-2.6)
        else:
            self.Tsys = check_unit(Tsys, units.K, 'Tsys')

    def make_noise_box(self):

        std_noise = self.Tsys / np.sqrt(self.nb_baselines * self.integration_time * self.freq_res)
        noise = np.random.normal(
            0,
            std_noise.value,
            size=(self.npix, self.npix, self.nfreqs)) * units.K
        return noise


def check_unit(value, unit, name):

    """
    Check if input has correct unit.

    Parameters
    ----------
        value: object
            Input to check the unit for.
        unit: astropy.units object or str
            Required unit for input.
        name: str
            Name of the input (for error message).
    Returns
    -------
        value: object
            Input converted to appropriate unit if necessary.
    """

    try:
        value.unit
    except AttributeError:
        warnings.warn(f'Assuming {name} given in {unit}...')
        value *= unit

    try:
        value = value.to(unit)
    except units.UnitConversionError:
        raise ValueError(f'Unit of {name} is not compatible with '
                         f'required {unit}... Aborting.')
    else:
        value = value.to(unit)
    return value
