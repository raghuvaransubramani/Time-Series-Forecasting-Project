"""
baseline_models.py
Functions to fit simple SARIMAX and Prophet baselines and produce forecasts for test range.
"""
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

def sarimax_forecast(train_series, test_index, order=(1,0,1), seasonal_order=(1,0,1,24)):
    model = sm.tsa.SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.predict(start=test_index[0], end=test_index[-1])
    return pred

def prophet_forecast(train_df, periods, freq='H'):
    # train_df: DataFrame with columns ['ds','y']
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fcst = m.predict(future)
    return fcst

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}
