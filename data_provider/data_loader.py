"""Gravity-only data loader for the PBI-Net submission repository.

We use ``Dataset_Gravity`` for the hourly station series in our experiments.
It applies a chronological split and training-only standardization.
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class Dataset_Gravity(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path=None, target='value',
                 scale=True, timeenc=1, freq='h', seasonal_patterns=None):
        self.seq_len, self.label_len, self.pred_len = size or (96, 48, 24)
        self.flag = flag
        self.scale = scale
        self.freq = freq

        row_count = len(pd.read_csv(
            os.path.join(root_path, data_path or 'gravity_all_merged.csv')
        ))
        border1s = [0, int(row_count * 0.7), int(row_count * 0.9)]
        border2s = [int(row_count * 0.7), int(row_count * 0.9), row_count]

        border1 = border1s[['train', 'val', 'test'].index(flag)]
        border2 = border2s[['train', 'val', 'test'].index(flag)]

        csv_path = os.path.join(
            root_path,
            data_path or 'gravity_all_merged.csv',
        )
        df_raw = pd.read_csv(csv_path, index_col='date', parse_dates=True)

        if target not in df_raw.columns:
            raise ValueError(f"Target column {target!r} was not found in the CSV")

        full_data = df_raw[[target]].values.astype(np.float32)
        self.dates = df_raw.index[border1:border2]
        self.data = full_data[border1:border2]

        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(full_data[border1s[0]:border2s[0]])
            self.data = self.scaler.transform(self.data)

        print(
            f"[{flag}] gravity dataset loaded: "
            f"{len(self.data)} rows, range {border1}:{border2}"
        )

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        seq_x = self.data[idx:idx + self.seq_len]
        seq_y = self.data[
            idx + self.seq_len - self.label_len:
            idx + self.seq_len + self.pred_len
        ]

        dates_x = self.dates[idx:idx + self.seq_len]
        dates_y = self.dates[
            idx + self.seq_len - self.label_len:
            idx + self.seq_len + self.pred_len
        ]

        seq_x_mark = time_features(dates_x, freq=self.freq)
        seq_y_mark = time_features(dates_y, freq=self.freq)

        return (
            seq_x,
            seq_y,
            seq_x_mark.transpose(1, 0),
            seq_y_mark.transpose(1, 0),
        )

    def inverse_transform(self, data):
        if hasattr(self.scaler, 'mean_'):
            return self.scaler.inverse_transform(data)
        print("Warning: scaler not fitted; returning input unchanged")
        return data
