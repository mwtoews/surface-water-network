"""Surface water network."""

__license__ = "BSD"
__author__ = "Mike Taves"
__email__ = "mwtoews@gmail.com"
__credits__ = ["Brioch Hemmings"]

__all__ = [
    "SurfaceWaterNetwork",
    "SwnMf6",
    "SwnModflow",
    "__version__",
    "core",
    "file",
    "modflow",
    "spatial",
]

from . import core, file, modflow, spatial
from ._version import version as __version__
from .core import SurfaceWaterNetwork
from .modflow import SwnMf6, SwnModflow
