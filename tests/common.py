"""Common code for testing."""

import re
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

datadir = Path("tests") / "data"

PANDAS_VERSION_TUPLE = tuple(int(x) for x in re.findall(r"\d+", pd.__version__))
