# Rotation tracks and tornado paths

MRMS's 24-hour rotation track, the largest 0-2 km azimuthal shear each point saw between 12 UTC on 10 December and 12 UTC on 11 December 2021, merged from every WSR-88D in range. The blue lines are the 71 tornado paths in the Storm Prediction Center's database for those days, coloured by EF rating. The long swaths of strong shear line up with the surveyed paths, including the 168-mile path from north-west Tennessee across western Kentucky.

![Rotation tracks and tornado paths](../images/tornado_tracks.png)

Data: MRMS on AWS (`noaa-mrms-pds`), and SPC's tornado database saved in `docs/examples/data/spc_tornadoes_20211210.csv`.

```python title="docs/examples/tornado_tracks.py"
--8<-- "tornado_tracks.py"
```
