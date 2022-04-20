"""Compatibility module."""
import contextlib
import warnings

import numpy as np
import shapely
from packaging.version import Version

NUMPY_GE_121 = Version(np.__version__) >= Version("1.21")

if shapely is not None:
    SHAPELY_GE_20 = Version(shapely.__version__) >= Version("2.0")
    SHAPELY_LT_18 = Version(shapely.__version__) < Version("1.8")
else:
    SHAPELY_GE_20 = False
    SHAPELY_LT_18 = False

shapely_warning = None
if shapely is not None:
    try:
        from shapely.errors import ShapelyDeprecationWarning as shapely_warning
    except ImportError:
        pass

if shapely_warning is not None and not SHAPELY_GE_20:

    @contextlib.contextmanager
    def ignore_shapely_warnings_for_object_array():

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Iteration|The array interface|__len__",
                shapely_warning,
            )
            if NUMPY_GE_121:
                # warning from numpy for existing Shapely releases (this is
                # fixed with Shapely 1.8)
                warnings.filterwarnings(
                    "ignore",
                    "An exception was ignored while fetching",
                    DeprecationWarning,
                )
            yield


elif SHAPELY_LT_18 and NUMPY_GE_121:

    @contextlib.contextmanager
    def ignore_shapely_warnings_for_object_array():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "An exception was ignored while fetching",
                DeprecationWarning,
            )
            yield


else:

    @contextlib.contextmanager
    def ignore_shapely_warnings_for_object_array():
        yield
