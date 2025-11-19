"""
data_pipeline.py
Downloads and preprocesses the UCI Household Power Consumption dataset,
resamples to hourly, creates features and splits into train/val/test.
Saves processed arrays as numpy .npz files for downstream use.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
RAW_ZIP = DATA_DIR / "household_power_consumption.zip"
RAW_TXT = DATA_DIR / "household_power_consumption.txt"

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def download_and_extract():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    if not RAW_TXT.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, RAW_ZIP)
        with zipfile.ZipFile(RAW_ZIP, 'r') as z:
            z.extractall(DATA_DIR)
        print("Download complete.")
    else:
        print("Dataset already present.")

def load_and_preprocess(resample_rule="H"):
    print("Loading dataset...")
    df = pd.read_csv(RAW_TXT, sep=';', parse_dates={'datetime': ['Date','Time']}, infer_datetime_format=True, low_memory=False, na_values=['?'])
    df = df[['datetime','Global_active_power']].set_index('datetime')
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df.sort_index().dropna()
    # Resample to hourly (configurable)
    df_res = df.resample(resample_rule).mean().interpolate()
    df_res['hour'] = df_res.index.hour
    df_res['dayofweek'] = df_res.index.dayofweek
    df_res['month'] = df_res.index.month
    df_res['is_weekend'] = (df_res['dayofweek'] >= 5).astype(int)
    df_res['roll_24_mean'] = df_res['Global_active_power'].rolling(24, min_periods=1).mean()
    df_res['roll_168_mean'] = df_res['Global_active_power'].rolling(168, min_periods=1).mean()
    df_res = df_res.fillna(method='ffill').fillna(method='bfill')
    return df_res

def split_and_scale(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = ['Global_active_power','hour','dayofweek','month','is_weekend','roll_24_mean','roll_168_mean']
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    scaler = MinMaxScaler()
    scaler.fit(train[feature_cols])
    train_s = pd.DataFrame(scaler.transform(train[feature_cols]), index=train.index, columns=feature_cols)
    val_s = pd.DataFrame(scaler.transform(val[feature_cols]), index=val.index, columns=feature_cols)
    test_s = pd.DataFrame(scaler.transform(test[feature_cols]), index=test.index, columns=feature_cols)
    # save scaler
    joblib.dump(scaler, OUT_DIR / "scaler.save")
    # save arrays
    np.savez(OUT_DIR / "processed.npz", train=train_s.values, val=val_s.values, test=test_s.values, train_index=train.index.astype(str).values, val_index=val.index.astype(str).values, test_index=test.index.astype(str).values)
    print("Saved processed data to outputs/processed.npz and scaler.save")
    return train_s, val_s, test_s

if __name__ == "__main__":
    download_and_extract()
    df = load_and_preprocess("H")
    split_and_scale(df)
