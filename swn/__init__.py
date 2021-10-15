"""Surface water network."""

__license__ = 'BSD'
__author__ = 'Mike Taves'
__email__ = 'mwtoews@gmail.com'
__credits__ = ['Brioch Hemmings']

__all__ = ["SurfaceWaterNetwork", "MfSfrNetwork", "SwnMf6", "SwnModflow"]

from swn._version import version as __version__
from swn.core import SurfaceWaterNetwork
from swn.modflow import MfSfrNetwork, SwnMf6, SwnModflow
