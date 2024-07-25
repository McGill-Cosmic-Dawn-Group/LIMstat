"""
__init__.py file for limstat
"""
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("limstat")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

from . import utils
from .simulations import foregrounds, cosmological_signal, thermal_noise
from .power_spectrum import power_spectrum as ps

del version
del PackageNotFoundError
