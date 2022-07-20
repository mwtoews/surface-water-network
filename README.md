# Surface water network
[![DOI](https://zenodo.org/badge/187739645.svg)](https://zenodo.org/badge/latestdoi/187739645)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/420bcd8896c14f18b2077dd987c78849)](https://app.codacy.com/manual/mwtoews/surface-water-network?utm_source=github.com&utm_medium=referral&utm_content=mwtoews/surface-water-network&utm_campaign=Badge_Grade_Dashboard)
[![CI](https://github.com/mwtoews/surface-water-network/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mwtoews/surface-water-network/actions/workflows/ci.yml)

A Python package to create and analyze surface water networks.


## Python packages

Python 3.7+ is required.

### Required

 - `geopandas` - process spatial data similar to pandas
 - `packaging` - used to check package versions
 - `pandas >=1.2` - tabular data analysis
 - `pyproj >=2.2` - spatial projection support
 - `rtree` - spatial index support

### Optional

 - `flopy` - read/write MODFLOW models
 - `netCDF4` - used to read TopNet files

## Testing

Run `pytest -v` or `python3 -m pytest -v`

For faster multi-core `pytest -v -n 2` (with `pytest-xdist`)

## Examples

```python
import geopandas
import pandas as pd
import swn
```

Read from Shapefile:
```python
shp_srs = 'tests/data/DN2_Coastal_strahler1z_stream_vf.shp'
lines = geopandas.read_file(shp_srs)
lines.set_index('nzsegment', inplace=True, verify_integrity=True)  # optional
```

Or, read from PostGIS:
```python
from sqlalchemy import create_engine, engine

con_url = engine.url.URL(drivername='postgresql', database='scigen')
con = create_engine(con_url)
sql = 'SELECT * FROM wrc.rec2_riverlines_coastal'
lines = geopandas.read_postgis(sql, con)
lines.set_index('nzsegment', inplace=True, verify_integrity=True)  # optional
```

Initialise and create network:
```python
n = swn.SurfaceWaterNetwork.from_lines(lines.geometry)
print(n)
# <SurfaceWaterNetwork: with Z coordinates
#   304 segments: [3046409, 3046455, ..., 3050338, 3050418]
#   154 headwater: [3046409, 3046542, ..., 3050338, 3050418]
#   3 outlets: [3046700, 3046737, 3046736]
#   no diversions />
```

Plot the network, write a Shapefile, write and read a SurfaceWaterNetwork file:
```python
n.plot()

swn.file.gdf_to_shapefile(n.segments, 'segments.shp')

n.to_pickle('network.pkl')
n = swn.SurfaceWaterNetwork.from_pickle('network.pkl')
```

Remove segments that meet a condition (stream order), or that are
upstream/downstream from certain locations:
```python
n.remove(
    n.segments.stream_order == 1,
    segnums=n.gather_segnums(upstream=3047927))
```

Read flow data from a TopNet netCDF file, convert from m3/s to m3/day:
```python

nc_path = 'tests/data/streamq_20170115_20170128_topnet_03046727_strahler1.nc'
flow = swn.file.topnet2ts(nc_path, 'mod_flow', 86400)
# remove time and truncate to closest day
flow.index = flow.index.floor('d')

# 7-day mean
flow7d = flow.resample('7D').mean()

# full mean
flow_m = pd.DataFrame(flow.mean(0)).T
```

Process a MODFLOW/flopy model:
```python
import flopy

m = flopy.modflow.Modflow.load('h.nam', model_ws='tests/data', check=False)
nm = swn.SwnModflow.from_swn_flopy(n, m)
nm.default_segment_data()
nm.set_segment_data_inflow(flow_m)
nm.plot()
nm.to_pickle('sfr_network.pkl')
nm = swn.SwnModflow.from_pickle('sfr_network.pkl', n, m)
nm.set_sfr_obj()
m.sfr.write_file('file.sfr')
nm.grid_cells.to_file('grid_cells.shp')
nm.reaches.to_file('reaches.shp')
```

## Citation

Toews, M. W.; Hemmings, B. 2019. A surface water network method for generalising streams and rapid groundwater model development. In: New Zealand Hydrological Society Conference, Rotorua, 3-6 December, 2019. p. 166-169.
