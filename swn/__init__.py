# -*- coding: utf-8 -*-
"""Surface water network."""
__version__ = '1.0dev'
__license__ = 'BSD'
__author__ = 'Mike Taves'
__email__ = 'mwtoews@gmail.com'
__credits__ = ['Brioch Hemmings']

__all__ = ["SurfaceWaterNetwork", "MfSfrNetwork"]

from swn.core import SurfaceWaterNetwork
from swn.modflow import MfSfrNetwork
