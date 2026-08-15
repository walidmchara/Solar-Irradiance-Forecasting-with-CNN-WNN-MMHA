# Solar data

Place your combined CSV at `data/raw/solar.csv`.

Expected core columns:
`datetime_utc, city, GHI, DNI, DHI, ClearSky_GHI, Temperature_2m,
Relative_Humidity_2m, Surface_Pressure, Wind_Speed_10m,
Wind_Direction_10m, Precipitation, clear_sky_index, is_day`.

Time/cyclic features are regenerated automatically from `datetime_utc`.

Recommended protocol: train on several cities and evaluate on a completely unseen city.
Raw datasets are excluded from GitHub.
