import numpy as np
import warnings
from astropy import units


class constants:
    """
    Fundamental constants and conversion constants
    in ** SI ** units.

    f21 : 21cm frequency in Hz
    """
    # 21cm rest-frame frequency
    f21 = 1420 * units.MHz


def bin_edges_from_array(arr):
    """
    Obtain bin edges from existing bin-array.

    Parameters
    ----------
        arr: 1d array
            List of bin centres.
    Returns
    -------
        arr: 1d array
            List of bin edges.
            Size is arr.size + 1.
    """
    # checks on inputs
    assert np.ndim(arr) == 1. and np.size(arr) > 1, \
        "arr must be a list of values."

    dx = np.diff(arr).mean()
    assert dx > 0, "arr must be sorted in increasing order."
    assert np.allclose(np.diff(arr), dx), \
        "arr must be regularly spaced."

    bin_edges = np.arange(
        arr.min()-dx/2,
        arr.max()+dx,
        step=dx)
    assert np.size(bin_edges) == np.size(arr) + 1

    return bin_edges


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
        value = value * unit

    try:
        value = value.to(unit)
    except units.UnitConversionError:
        raise ValueError(f'Unit of {name} is not compatible with '
                         f'required {unit}... Aborting.')
    else:
        value = value.to(unit)
    return value


def check_units(value, units, name):

    """
    Check if input has correct unit among list of possibilities.

    Parameters
    ----------
        value: object
            Input to check the unit for.
        units: list of astropy.units object or str
            Possible units for input.
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
        raise ValueError(f"{name} doesn't have astropy units! "
                         "Please add them!")


def convert_units(signal, freq, FWHM, from_to='bt_i'):
    '''Helper function which is basically just a wrapper
    of the astropy equivalencies for brightness temperature.

    If the desired conversion involves Jy/beam_area then you must
    specify a beam, otherwise you could ask for Jy/str.

    Parameters
    ----------
        signal (array): signal, must have astropy units of
                        either brightness_temperature (e.g. mK)
                        or intensity (e.g. Jy/str, Jy/beam_area)
        freq (float): The frequency of the observed signal,
                        must have astropy units of frequency
        FWHM (float): The FWHM of the beam. This is then converted
                        to sigma of the beam to calculate the beam area
                        must have astropy units of angular extent
                        (e.g. degrees, arcsec)
        fromt_to (str): TODO

    Returns
    -------
        binary_sum (arr): Converted form of 'signal' parameter in
                            new desired astropy units
    '''

    beam_sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    beam_area = 2*np.pi*beam_sigma**2
    equiv = units.brightness_temperature(freq)

    if from_to == 'bt_i':
        signal_convert = (signal).to(units.Jy/beam_area, equivalencies=equiv)
    elif from_to == 'i_bt':
        signal_convert = (signal).to(units.mK, equivalencies=equiv)
    else:
        raise ValueError('Wrong input for from_to.')

    return signal_convert


def comply_units(
        value,
        default_unit,
        quantity,
        desired_unit=None,
        equivalencies=None
        ):
    """
    Check the units of the input and optionally convert to another type.

    Parameters
    ----------
        value: np.ndarray, astropy.Quantity, or float
            Quantity to perform unit-checking (and optional conversion) on.
        default_unit: astropy unit
            Unit to use in case ``value`` is not an ``astropy.Quantity``.
        quantity: str
            Name of the quantity being unit-checked, for logging purposes.
        desired_unit: astropy unit, optional
            If provided, specifies the unit that ``value``
            should be converted to.
        equivalencies: list, optional
            List of equivalencies for converting between ``value.unit`` (or
            ``default_unit``) and ``desired_unit``.

    Returns
    -------
        checked_value: np.ndarray or float
            Provided quantity in the desired units.
    """
    if not hasattr(value, "unit"):
        msg = f"Provided {quantity} does not have units. "
        msg += f"Assuming the input is provided in {str(default_unit)}."
        warnings.warn(msg, stacklevel=2)
        value = value * default_unit

    if not value.unit.is_equivalent(
        default_unit, equivalencies=equivalencies
    ):
        raise ValueError(f"{quantity} does not have the right units! has "
                         f"{value.unit}, uncompatible with {default_unit}")

    if desired_unit is not None:
        if not desired_unit.is_equivalent(
            value.unit, equivalencies=equivalencies
        ):
            raise ValueError(
                f"{quantity} cannot be converted to {str(desired_unit)}."
            )
        equivalencies = equivalencies or []
        value = value.to(desired_unit, equivalencies=equivalencies)

    return value.value


def powerlaw(a, b, idx, size=1):
    """Generate random numbers with a power-law distribution.

    Taken from stackoverflow:
    https://stackoverflow.com/questions/31114330/python-generating-random-numbers-from-a-power-law-distribution
    Parameters
    ----------
    a
        Minimum value generated by the distribution.
    b
        Maximum value generated by the distribution.
    idx
        Power-law index.
    size
        Number of samples to generate.

    Returns
    -------
    samples
        Draws from the power-law distribution.
    """
    samples = np.random.uniform(size=size)
    return (a**idx + (b**idx - a**idx)*samples) ** (1/idx)

def rand_pos(n_src, theta_bounds=(0,np.pi), phi_bounds=(0,2*np.pi)):
    """Generate random positions uniformly on the sphere.

    Parameters
    ----------
    n_src: int
        Number of sources to generate positions for.
    theta_bounds: tuple of float
        Range of co-latitudes on which to generate source positions.
    phi_bounds: tuple of float
        Range of longitudes on which to generate source positions.

    Returns
    -------
    theta, phi
        Co-latitude and longitude of source positions, in radians.
    """
    phi = np.random.uniform(*phi_bounds, n_src)
    c1 = np.cos(theta_bounds[0])
    c2 = np.cos(theta_bounds[1])
    cos_theta = np.random.uniform(c1, c2, n_src)
    return np.arccos(cos_theta), phi

def f2z(f, rest_freq=constants.f21):
    """
    Converts a frequency into a redshift for a given rest-frame frequency.

    Parameters
    ----------
        f : astropy.Quantity with unit
            Input frequency.
            If it has no unit, MHz are assumed.
        rest_freq: astropy.Quantity with unit
            Rest-frame frequency.
            If it has no unit, MHz are assumed.
            Default is 21cm.

    Returns
    -------
        z: float
            Redshift.
    """
    rest_freq = comply_units(
        value=rest_freq,
        default_unit=units.MHz,
        quantity="rest freq",
        desired_unit=units.MHz,
    )
    f = comply_units(
        value=f,
        default_unit=units.MHz,
        quantity="input freq",
        desired_unit=units.MHz,
    )
    return (rest_freq / f) - 1.


def z2f(z, rest_freq=constants.f21, unit='MHz'):
    """
    Converts a redshift into a frequency for a given rest-frame frequency.

    Parameters
    ----------
        z: float
            Input redshift.
        rest_freq: astropy.Quantity with unit
            Rest-frame frequency.
            If it has no unit, MHz are assumed.
            Default is 21cm.
        unit: str
            Desired frequency for the output.
            Options are MHz and Hz.
            Default is MHz.

    Returns
    -------
        f : float
            Redshifted frequency in required units.
    """
    rest_freq = comply_units(
        value=rest_freq,
        default_unit=units.MHz,
        quantity="rest freq",
        desired_unit=units.MHz if unit == 'MHz' else units.Hz,
    )

    return rest_freq / (1. + z)
