# Surface water network
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/420bcd8896c14f18b2077dd987c78849)](https://app.codacy.com/manual/mwtoews/surface-water-network?utm_source=github.com&utm_medium=referral&utm_content=mwtoews/surface-water-network&utm_campaign=Badge_Grade_Dashboard)
[![Travis Status](https://api.travis-ci.org/mwtoews/surface-water-network.svg?branch=master)](https://travis-ci.org/mwtoews/surface-water-network)

Creates surface water network, which can be used to create MODFLOW's SFR.


## Python packages

Python 3.6+ is required.

### Required

 - `geopandas` - process spatial data similar to pandas
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
n = swn.SurfaceWaterNetwork(lines.geometry)
print(n)
# <SurfaceWaterNetwork: with Z coordinates
#   304 segments: [3046409, 3046455, ..., 3050338, 3050418]
#   154 headwater: [3046409, 3046542, ..., 3050338, 3050418]
#   3 outlets: [3046700, 3046737, 3046736]
#   no diversions />
```

Plot the network, write a Shapefile:
```python
n.plot()

swn.file.gdf_to_shapefile(n.segments, 'segments.shp')
```

Remove segments that meet a condition (stream order), or that are
upstream/downstream from certain locations:
```python
n.remove(n.segments.stream_order == 1, segnums=n.query(upstream=3047927))
```

Read flow data from a TopNet netCDF file:
```python

nc_fname = 'streamq_20170115_20170128_topnet_03046727_strahler1.nc'
flow = swn.file.topnet2ts(os.path.join(datadir, nc_fname), 'mod_flow')
# convert from m3/s to m3/day
flow *= 24 * 60 * 60
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
nm = swn.MfSfrNetwork(n, m, inflow=flow_m)
m.sfr.write_file('file.sfr')
nm.grid_cells.to_file('grid_cells.shp')
nm.reaches.to_file('reaches.shp')
```

## Citation

Toews, M. W.; Hemmings, B. 2019. A surface water network method for generalising streams and rapid groundwater model development. In: New Zealand Hydrological Society Conference, Rotorua, 3-6 December, 2019. p. 166-169.
