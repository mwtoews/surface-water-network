# Surface water network

Creates surface water network to be used for MODFLOW's SFR.


## Python packages

Python 3.5+ is required.

### Required

 - `geopandas` - process spatial data similar to pandas

### Optional

 - `flopy` - read/write MODFLOW models
 - `gdal` - import from different geospatial formats
 - `netCDF4` - used to read TopNet files

## Testing

Run `py.test -v`

## Examples

```python
import geopandas
import swn
```

Read from Shapefile:
```python
shp_srs = 'tests/data/DN2_Coastal_strahler1z_stream_vf.shp'
lines = geopandas.read_file(shp_srs)
lines.set_index('nzsegment', inplace=True)  # optional
```

Or, read from PostGIS:
```python
from sqlalchemy import create_engine, engine

con_url = engine.url.URL(drivername='postgresql', database='scigen')
con = create_engine(con_url)
sql = 'SELECT * FROM wrc.rec2_riverlines_coastal'
lines = geopandas.read_postgis(sql, con)
lines.set_index('nzsegment', inplace=True)  # optional
```

Initialise and create network:
```python
n = swn.SurfaceWaterNetwork(lines)
```

Plot something, write a Shapefile:
```python
n.segments.sort_values('stream_order').plot('stream_order')
n.segments.to_file('segments.shp')
```

Remove segments that meet a condition (stream order), or that are
upstream/downstream from certain locations:
```python
n.remove(n.segments.stream_order == 1, segnums=n.query(upstream=3047927))
```

Read flow data from a TopNet netCDF file:
```python
nc_fname = 'streamq_20170115_20170128_topnet_03046727_strahler1.nc'
flow = swn.topnet2ts(os.path.join(datadir, nc_fname), 'mod_flow')
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
n.process_flopy(m, inflow=flow_m)
m.sfr.write_file('file.sfr')
n.grid_cells.to_file('grid_cells.shp')
n.reaches.to_file('reaches.shp')
```
