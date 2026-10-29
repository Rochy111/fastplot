# Warm conveyor belt trajectories

966 air parcels started at 950 hPa over the eastern United States and western Atlantic, carried forward 48 hours by the three-dimensional GFS wind (u, v and omega on 19 levels, every 3 hours, Heun's method in one-hour steps). The 47 that rise more than 600 hPa, the warm conveyor belt criterion of Wernli and Davies (1997), climb from the boundary layer to the jet in about 12 hours and fan out towards Greenland and western Europe.

![Warm conveyor belt trajectories](../images/warm_conveyor.png)

Data: GFS 1° forecasts from 00 UTC 27 October 2026, 17 times from 0 to 48 hours, from NOAA's archive on AWS.

```python title="docs/examples/warm_conveyor.py"
--8<-- "warm_conveyor.py"
```
