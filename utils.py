"""
utils.py
Utility functions: dataset class, inversion helper, metrics.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

class TimeSeriesDataset(Dataset):
    def __init__(self, arr, input_length=168, horizon=24):
        self.arr = arr.astype('float32')
        self.input_length = input_length
        self.horizon = horizon
    def __len__(self):
        return max(0, len(self.arr) - self.input_length - self.horizon + 1)
    def __getitem__(self, idx):
        x = self.arr[idx: idx + self.input_length]
        y = self.arr[idx + self.input_length: idx + self.input_length + self.horizon, 0]
        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(-1)

def invert_target(scaled_arr, scaler, feature_count=7):
    # scaled_arr: shape (n, horizon, 1)
    n, h, _ = scaled_arr.shape
    flat = scaled_arr.reshape(-1, 1)
    zeros = np.zeros((flat.shape[0], feature_count))
    zeros[:, 0] = flat[:, 0]
    inv = scaler.inverse_transform(zeros)[:, 0]
    return inv.reshape(n, h)

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = math.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    return {"MAE": mae, "RMSE": rmse}
