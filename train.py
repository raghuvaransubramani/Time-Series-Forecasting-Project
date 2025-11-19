"""
train.py
Top-level training script to run the full pipeline. Example usage:
    python train.py --mode full
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import joblib, json
from code.utils import TimeSeriesDataset, invert_target, compute_metrics
from code.model_lstm_attention import LSTMWithAttention
from code.model_transformer import TransformerTS
from code.baseline_models import sarimax_forecast, prophet_forecast, metrics as baseline_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def load_processed():
    p = OUT_DIR / "processed.npz"
    if not p.exists():
        # run data pipeline
        from code.data_pipeline import download_and_extract, load_and_preprocess, split_and_scale
        download_and_extract()
        df = load_and_preprocess("H")
        train_s, val_s, test_s = split_and_scale(df)
    else:
        arr = np.load(p, allow_pickle=True)
        train_s = pd.DataFrame(arr['train'])
        val_s = pd.DataFrame(arr['val'])
        test_s = pd.DataFrame(arr['test'])
    scaler = joblib.load(OUT_DIR / "scaler.save")
    return train_s.values, val_s.values, test_s.values, scaler

def train_lstm(train_arr, val_arr, test_arr, scaler, epochs=15, batch_size=64, input_len=168, horizon=24):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = TimeSeriesDataset(train_arr, input_length=input_len, horizon=horizon)
    val_ds = TimeSeriesDataset(val_arr, input_length=input_len, horizon=horizon)
    test_ds = TimeSeriesDataset(test_arr, input_length=input_len, horizon=horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = LSTMWithAttention(train_arr.shape[1], enc_hidden=128, dec_hidden=64, attn_dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, epochs+1):
        model.train(); total=0.0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            opt.zero_grad()
            preds, _ = model(X, future_steps=y.size(1))
            loss = crit(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * X.size(0)
        val_loss, preds_v, targets_v, attn_v = eval_model(model, val_loader, crit)
        print(f"Epoch {epoch} TrainLoss {total/len(train_loader.dataset):.6f} ValLoss {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss; torch.save(model.state_dict(), OUT_DIR / "best_lstm_attn.pth"); print("Saved best model")

    # final eval
    test_loss, preds_t, targets_t, attn_t = eval_model(model, test_loader, crit)
    preds_inv = invert_target(preds_t, scaler); targets_inv = invert_target(targets_t, scaler)
    mets = compute_metrics(targets_inv, preds_inv)
    # save preds & attention for first batch
    np.savez(OUT_DIR / "lstm_preds.npz", preds=preds_inv, targets=targets_inv, attn=attn_t)
    with open(OUT_DIR / "lstm_metrics.json", "w") as f:
        json.dump(mets, f)
    print("LSTM metrics:", mets)
    return mets

def eval_model(model, loader, criterion):
    model.eval(); total=0.0; ps,ts,ats=[],[],[]
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), y.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            preds, attn = model(X, future_steps=y.size(1))
            loss = criterion(preds, y)
            total += loss.item() * X.size(0)
            ps.append(preds.cpu().numpy()); ts.append(y.cpu().numpy()); ats.append(attn.cpu().numpy())
    return total/len(loader.dataset), np.concatenate(ps,0), np.concatenate(ts,0), np.concatenate(ats,0)

def run_baselines(train_df, test_df):
    # train_df: original (unscaled) train pandas series; test_df: original test pandas series
    sarima_pred = sarimax_forecast(train_df['Global_active_power'], test_df.index)
    sarima_metrics = baseline_metrics(test_df['Global_active_power'].values, sarima_pred.values)
    # Prophet forecast
    prophet_train = train_df.reset_index().rename(columns={'datetime':'ds','Global_active_power':'y'})[['ds','y']]
    periods = len(test_df) + len(train_df)*0  # we directly predict for test range; for speed keep simple
    fcst = prophet_forecast(prophet_train, periods=len(test_df)+len(test_df))  # simple long forecast
    pred_prophet = fcst.set_index('ds').loc[test_df.index]['yhat'].values
    prophet_metrics = baseline_metrics(test_df['Global_active_power'].values, pred_prophet)
    # save
    return {'SARIMAX': sarima_metrics, 'Prophet': prophet_metrics}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full','lstm','baselines','pipeline'], default='full')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    if args.mode in ['full', 'pipeline']:
        from code.data_pipeline import download_and_extract, load_and_preprocess, split_and_scale
        download_and_extract(); df = load_and_preprocess('H'); train_s, val_s, test_s = split_and_scale(df)
    else:
        # ensure processed.npz exists
        pass
    # load processed arrays and scaler
    data_file = OUT_DIR / "processed.npz"
    if data_file.exists():
        arr = np.load(data_file, allow_pickle=True)
        train_arr = arr['train']; val_arr = arr['val']; test_arr = arr['test']
    else:
        # fallback to reading from outputs produced by pipeline
        train_arr = train_s.values; val_arr = val_s.values; test_arr = test_s.values
    scaler = joblib.load(OUT_DIR / "scaler.save")
    if args.mode in ['full','lstm']:
        train_lstm(train_arr, val_arr, test_arr, scaler, epochs=args.epochs)
    if args.mode in ['full','baselines']:
        # load raw original series for baselines
        raw = pd.read_csv(Path(__file__).resolve().parents[1]/'data'/'household_power_consumption.txt', sep=';', parse_dates={'datetime':['Date','Time']}, infer_datetime_format=True, low_memory=False, na_values=['?'])
        raw = raw[['datetime','Global_active_power']].set_index('datetime')
        raw['Global_active_power'] = pd.to_numeric(raw['Global_active_power'], errors='coerce')
        raw = raw.sort_index().dropna().resample('H').mean().interpolate()
        n=len(raw); train_end=int(n*0.7); val_end=int(n*0.85)
        train_df = raw.iloc[:train_end]; val_df = raw.iloc[train_end:val_end]; test_df = raw.iloc[val_end:]
        bas = run_baselines(train_df, test_df)
        print('Baselines:', bas)
