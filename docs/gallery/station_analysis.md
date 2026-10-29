# METAR temperature analysis

Temperature from every METAR report in the region at 06 UTC 27 October 2026, contoured on a Delaunay triangulation of the station positions with `tricontourf`. Triangles with a side longer than 2.5° are masked, so the analysis stops where the network does.

![METAR temperature analysis](../images/station_analysis.png)

Data: aviationweather.gov, saved in `docs/examples/data/metars_2026102706.csv`.

```python title="docs/examples/station_analysis.py"
--8<-- "station_analysis.py"
```
