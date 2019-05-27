# Surface water network

Creates surface water network to be used for MODFLOW's SFR.


## Python packages

### Required

 - `geopandas`

### Optional

 - `rtree` - fast spatial indexing
 - `gdal` - import from different geospatial formats


## Testing

Run `pytest -v`

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
n.reaches.sort_values('stream_order').plot('stream_order', legend=True)
n.reaches.to_file('reaches.shp')
```
