# Surface analysis

Sea-level pressure, smoothed a little and labelled every 4 hPa, over 10 m wind speed, with barbs every three degrees. Highs and lows are the extremes of a 41-point window, marked with their central pressure.

![Surface analysis](../images/surface_analysis.png)

Data: GFS 0.25°, 00 UTC 27 October 2026, six-hour forecast, read with fastgrib from NOAA's archive on AWS. To use a copy you already have, set `GFS_FILE` to its path.

```python title="docs/examples/surface_analysis.py"
--8<-- "surface_analysis.py"
```
