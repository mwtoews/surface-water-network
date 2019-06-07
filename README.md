# Surface water network

Creates surface water network to be used for MODFLOW's SFR.


## Python packages

### Required

 - `geopandas`

### Optional

 - `flopy` - read/write MODFLOW models
 - `gdal` - import from different geospatial formats
 - `rtree` - fast spatial indexing

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

Process a MODFLOW/flopy model:
```python
import flopy

m = flopy.modflow.Modflow.load('h.nam', model_ws='tests/data', check=False)
n.process_flopy(m)
m.sfr.write_file('file.sfr')
n.grid_cells.to_file('grid_cells.shp')
n.reaches.to_file('reaches.shp')
```
