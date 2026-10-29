# Dynamic tropopause map

Ertel potential vorticity is computed on 23 pressure levels from the GFS temperature and wind, using the isobaric form with spherical derivatives and the curvature term in the relative vorticity. Each column is searched from the top down for the first level below 2 PVU, and potential temperature and wind are interpolated to that crossing. Low values mark troughs, where stratospheric air extends down. Black contours show cyclonic vorticity in the 925-850 hPa layer.

![Dynamic tropopause map](../images/dynamic_tropopause.png)

Data: GFS 0.5°, 00 UTC 27 October 2026, six-hour forecast, read with fastgrib from NOAA's archive on AWS by byte range. The loading and finite-difference code is shared with the other diagnostics in `docs/examples/gfs3d.py`.

```python title="docs/examples/dynamic_tropopause.py"
--8<-- "dynamic_tropopause.py"
```

??? note "docs/examples/gfs3d.py, shared by the diagnostics"
    ```python
    --8<-- "gfs3d.py"
    ```
