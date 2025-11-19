# Project Report — Advanced Time Series Forecasting with Attention

## 1. Problem statement
Forecast electricity consumption (hourly) using advanced deep learning models (LSTM+Attention, Transformer) and compare against classical baselines (SARIMAX, Prophet).

## 2. Dataset
UCI Household Power Consumption. Resampled to hourly frequency; > 100k hours of data over years — satisfies >=5000 observations requirement.

## 3. Preprocessing
- Parse datetime, convert to numeric target.
- Resample to hourly means, interpolate short gaps.
- Create time features: hour, dayofweek, month, is_weekend.
- Rolling features: 24h and 168h means.
- Train/val/test split: 70/15/15 (time-aware).
- Scale features with MinMaxScaler (fit on train only).

## 4. Models implemented
- Custom **LSTM + Additive Attention** (encoder LSTM + attention + small decoder LSTMCell for autoregressive multistep forecasting).
- Simple **Transformer encoder** with positional encodings and pooling head that predicts the entire horizon.
- Baselines: SARIMAX (seasonal) and Prophet (trend + seasonality).

## 5. Hyperparameter tuning
Optuna stub is included in the notebook and can be run to tune encoder size, decoder size, attention dim, learning rate, etc.

## 6. Evaluation protocol
- Metrics: MAE, RMSE (reported per horizon averaged over test set).
- Models compared on the same test set (time-aware split).
- Attention interpretability: heatmaps of attention weights (forecast steps × encoder timesteps) and simple ablation (zeroing top-k attended timesteps).

## 7. Results (example)
Results are saved in `outputs/final_metrics_table.csv`. The notebook and train script produce these outputs after training.

## 8. Reproducibility & production notes
- All code organized modularly in `/code` and can be run locally or in Colab.
- Save best model weights and scaler in `/outputs`.
- Package model with FastAPI/TorchServe for serving; monitor drift & performance in production.

