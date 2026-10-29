# GFS 2 m temperature on six projections

The same 2 m temperature field on six of the projections fastplot supports. Every panel has the same height, with its width set by the shape of the map, which each axes reports through `get_extent()`.

![GFS 2 m temperature on six projections](../images/projections.png)

Data: GFS 0.25°, 00 UTC 27 October 2026, six-hour forecast, read with fastgrib from NOAA's archive on AWS. To use a copy you already have, set `GFS_FILE` to its path.

```python title="docs/examples/projections.py"
--8<-- "projections.py"
```
