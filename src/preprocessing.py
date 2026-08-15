from __future__ import annotations
import numpy as np
import pandas as pd


def normalize_solar_columns(frame):
    alias_map = {
        "global_irradiance": "GI",
        "ghi": "GI",
        "GHI": "GI",
        "gi": "GI",
        "ClearSky_GHI": "Clearsky_GHI",
        "clear_sky_ghi": "Clearsky_GHI",
        "ClearSky GHI": "Clearsky_GHI",
        "ClearSky_DNI": "Clearsky_DNI",
        "clear_sky_dni": "Clearsky_DNI",
        "ClearSky DNI": "Clearsky_DNI",
        "ClearSky_DHI": "Clearsky_DHI",
        "clear_sky_dhi": "Clearsky_DHI",
        "ClearSky DHI": "Clearsky_DHI",
        "Solar Height": "SH",
        "solar_height": "SH",
        "sh": "SH",
        "Solar_Zenith_Angle": "Solar_Zenith_Angle",
        "solar_zenith_angle": "Solar_Zenith_Angle",
        "zenith_angle": "Solar_Zenith_Angle",
        "Ambient Temperature": "T",
        "ambient_temperature": "T",
        "temperature_2m": "T",
        "Wind Speed": "WS",
        "wind_speed": "WS",
        "wind_speed_10m": "WS",
        "Wind Direction": "WD",
        "wind_direction": "WD",
        "wind_direction_10m": "WD",
        "Relative Humidity": "RH",
        "relative_humidity": "RH",
        "relative_humidity_2m": "RH",
        "Precipitation": "P",
        "precipitation": "P",
        "Cloud Cover": "CC",
        "cloud_cover": "CC",
        "Atmospheric Pressure": "PA",
        "atmospheric_pressure": "PA",
        "surface_pressure": "PA",
        "pressure": "PA",
        "is_day": "is_day",
    }

    renamed = frame.copy()
    for source, target in alias_map.items():
        if source in renamed.columns and target not in renamed.columns:
            renamed = renamed.rename(columns={source: target})
    return renamed


def add_time_features(frame, datetime_column):
    frame = frame.copy()
    dt = pd.to_datetime(frame[datetime_column], utc=True, errors="coerce")
    frame = frame.loc[dt.notna()].copy()
    dt = pd.to_datetime(frame[datetime_column], utc=True)
    frame["year"] = dt.dt.year
    frame["month"] = dt.dt.month
    frame["day"] = dt.dt.day
    frame["hour"] = dt.dt.hour
    frame["day_of_year"] = dt.dt.dayofyear
    frame["day_of_week"] = dt.dt.dayofweek
    frame["hour_sin"] = np.sin(2*np.pi*frame["hour"]/24.0)
    frame["hour_cos"] = np.cos(2*np.pi*frame["hour"]/24.0)
    frame["day_of_year_sin"] = np.sin(2*np.pi*frame["day_of_year"]/365.25)
    frame["day_of_year_cos"] = np.cos(2*np.pi*frame["day_of_year"]/365.25)
    frame["month_sin"] = np.sin(2*np.pi*(frame["month"]-1)/12.0)
    frame["month_cos"] = np.cos(2*np.pi*(frame["month"]-1)/12.0)
    return frame


def load_solar_csv(csv_path, datetime_column):
    frame = pd.read_csv(csv_path)
    frame = normalize_solar_columns(frame)
    frame = add_time_features(frame, datetime_column)
    return frame.sort_values(datetime_column).reset_index(drop=True)
