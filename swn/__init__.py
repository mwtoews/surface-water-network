# -*- coding: utf-8 -*-
"""Surface water network."""
__version__ = '1.0'
__license__ = 'BSD'
__author__ = 'Mike Taves'
__email__ = 'mwtoews@gmail.com'
__credits__ = ['Brioch Hemmings']

from swn import base, file, logger, modflow, spatial, util  # noqa: F401

from swn.base import SurfaceWaterNetwork  # noqa: F401
from swn.modflow import MfSfrNetwork  # noqa: F401
