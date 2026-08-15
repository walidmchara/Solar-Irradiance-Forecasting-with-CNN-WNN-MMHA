from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

def make_city_sequences(frame, feature_columns, target_column, city_column, sequence_length, scaler, datetime_column="datetime_utc"):
    xs, ys, meta = [], [], []
    for city, group in frame.groupby(city_column, sort=False):
        group = group.sort_values(datetime_column).reset_index(drop=True)
        x = scaler.transform(group[feature_columns].to_numpy(dtype=np.float32))
        y = group[target_column].to_numpy(dtype=np.float32)
        for end in range(sequence_length-1, len(group)):
            start = end-sequence_length+1
            xs.append(x[start:end+1])
            ys.append(y[end])
            meta.append({"city": city, "datetime_utc": str(group.loc[end, datetime_column])})
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), meta

class SolarSequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
