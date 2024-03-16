"""Common code for testing."""
import re
import sys
from importlib import metadata
from pathlib import Path

import geopandas
import pandas as pd
import pytest
from shapely import wkt

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = False
    plt = None

if matplotlib and sys.platform == "darwin":
    matplotlib.use("qt5agg")


# Import this local package for tests
#  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import swn  # noqa: E402
from swn.compat import ignore_shapely_warnings_for_object_array

PANDAS_VESRSION_TUPLE = tuple(int(x) for x in re.findall(r"\d+", pd.__version__))

datadir = Path("tests") / "data"


# https://commons.wikimedia.org/wiki/File:Flussordnung_(Strahler).svg
fluss_gs = geopandas.GeoSeries(
    wkt.loads(
        """\
MULTILINESTRING(
    (380 490, 370 420), (300 460, 370 420), (370 420, 420 330),
    (190 250, 280 270), (225 180, 280 270), (280 270, 420 330),
    (420 330, 584 250), (520 220, 584 250), (584 250, 710 160),
    (740 270, 710 160), (735 350, 740 270), (880 320, 740 270),
    (925 370, 880 320), (974 300, 880 320), (760 460, 735 350),
    (650 430, 735 350), (710 160, 770 100), (700  90, 770 100),
    (770 100, 820  40))
"""
    ).geoms
)


@pytest.fixture
def fluss_n():
    return swn.SurfaceWaterNetwork.from_lines(fluss_gs)


@pytest.fixture(scope="session", autouse=True)
def coastal_lines_gdf():
    gdf = geopandas.read_file(datadir / "DN2_Coastal_strahler1z_stream_vf.shp")
    return gdf.set_index("nzsegment")


@pytest.fixture(scope="module")
def coastal_polygons_gdf(coastal_lines_gdf):
    polygons = geopandas.read_file(datadir / "DN2_Coastal_strahler1_vf.shp")
    polygons.set_index("nzsegment", inplace=True)
    # repair the shapefile by filling in the missing data
    for segnum in [3046737, 3047026, 3047906, 3048995, 3049065]:
        line = coastal_lines_gdf.loc[segnum]
        poly_d = {
            "HydroID": line["HydroID"],
            "GridID": 0,
            "OBJECTID": 0,
            "nzreach_re": line["nzreach_re"],
            "Shape_Leng": 0.0,
            "Shape_Area": 0.0,
            "Area": 0.0,
            "X84": 0.0,
            "Y84": 0.0,
            "geometry": line["geometry"].centroid.buffer(20.0, 1),
            # wkt.loads("POLYGON EMPTY")
        }
        with ignore_shapely_warnings_for_object_array():
            polygons.loc[segnum] = poly_d
    return polygons.reindex(index=coastal_lines_gdf.index)


@pytest.fixture(scope="session", autouse=True)
def coastal_points():
    fpath = datadir / "Coastal_points.shp"
    coastal_points_gdf = geopandas.read_file(fpath).set_index("id")
    return coastal_points_gdf.geometry


@pytest.fixture(scope="module")
def coastal_swn(coastal_lines_gdf):
    return swn.SurfaceWaterNetwork.from_lines(coastal_lines_gdf.geometry)


@pytest.fixture(scope="module")
def coastal_swn_w_poly(coastal_lines_gdf, coastal_polygons_gdf):
    return swn.SurfaceWaterNetwork.from_lines(
        coastal_lines_gdf.geometry, coastal_polygons_gdf.geometry
    )


@pytest.fixture(scope="session", autouse=True)
def coastal_flow_ts():
    csv_fname = "streamq_20170115_20170128_topnet_03046727_m3day.csv"
    ts = pd.read_csv(datadir / csv_fname, index_col=0)
    ts.columns = ts.columns.astype(int)
    ts.index = pd.to_datetime(ts.index)
    return ts


@pytest.fixture(scope="module")
def coastal_flow_m(coastal_flow_ts):
    coastal_flow_m = pd.DataFrame(coastal_flow_ts.mean(0)).T
    # coastal_flow_m.index = pd.DatetimeIndex(["2000-01-01"])
    return coastal_flow_m


def pytest_report_header(config):
    """Header for pytest to show versions of packages."""
    required = []
    extra = {}
    for item in metadata.requires("surface-water-network"):
        pkg_name = re.findall(r"[a-z0-9_\-]+", item, re.IGNORECASE)[0]
        if res := re.findall("extra == ['\"](.+)['\"]", item):
            assert len(res) == 1, item
            pkg_extra = res[0]
            if pkg_extra not in extra:
                extra[pkg_extra] = []
            extra[pkg_extra].append(pkg_name)
        else:
            required.append(pkg_name)

    processed = set()
    lines = []
    items = []
    for name in required:
        processed.add(name)
        try:
            version = metadata.version(name)
            items.append(f"{name}-{version}")
        except metadata.PackageNotFoundError:
            items.append(f"{name} (not found)")
    lines.append("required packages: " + ", ".join(items))
    installed = []
    not_found = []
    for name in extra["extra"]:
        if name in processed:
            continue
        processed.add(name)
        try:
            version = metadata.version(name)
            installed.append(f"{name}-{version}")
        except metadata.PackageNotFoundError:
            not_found.append(name)
    if installed:
        lines.append("optional packages: " + ", ".join(installed))
    if not_found:
        lines.append("optional packages not found: " + ", ".join(not_found))
    return "\n".join(lines)
