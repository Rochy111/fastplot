# Hurricane vortex structure

Azimuthal means around Hurricane Milton 12 hours into the HAFS-A forecast from 00 UTC 9 October 2024, in storm-relative coordinates: the primary circulation (tangential wind) with surfaces of absolute angular momentum, the secondary circulation (inflow, outflow and the eyewall updraught), and equivalent potential temperature from Bolton (1980) against the same angular momentum surfaces. In Emanuel's theory of the mature hurricane the eyewall is neutral to slantwise convection, so theta-e surfaces lie along M surfaces there; between 20 and 60 km from the centre they do.

![Hurricane vortex structure](../images/hafs_milton.png)

Data: HAFS-A storm nest from NOAA's HAFS archive on AWS (`noaa-nws-hafs-pds`), about 180 records fetched by byte range.

```python title="docs/examples/hafs_milton.py"
--8<-- "hafs_milton.py"
```
