from __future__ import annotations
import numpy as np
import pandas as pd

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
    frame = add_time_features(frame, datetime_column)
    return frame.sort_values(datetime_column).reset_index(drop=True)
