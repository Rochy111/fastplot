# Quasi-geostrophic forcing and model ascent

Q-vectors and the forcing term of the QG omega equation, -2 div Q, computed from the GFS 700 hPa height and temperature with the geostrophic wind, next to the model's own vertical motion at the same level. Q-vectors converge where the large-scale balanced flow forces ascent. Where the two panels agree, as ahead of the trough over the western Atlantic, the model's ascent is that balanced response; where they differ, convection, terrain or the tropics are at work.

![Quasi-geostrophic forcing and model ascent](../images/qg_forcing.png)

Data: GFS 0.5°, 00 UTC 27 October 2026, six-hour forecast, read with fastgrib from NOAA's archive on AWS by byte range. The loading and finite-difference code is shared with the other diagnostics in `docs/examples/gfs3d.py`.

```python title="docs/examples/qg_forcing.py"
--8<-- "qg_forcing.py"
```

??? note "docs/examples/gfs3d.py, shared by the diagnostics"
    ```python
    --8<-- "gfs3d.py"
    ```
