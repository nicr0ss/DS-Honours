import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from model.estimator import GARegressor
import time
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
import pickle
import torch
import cudf

with open("../grid_large_std.pkl", "rb") as f:
    grid = pickle.load(f)

def calculate_midpoint(value):
    if value is None or value == "None":
        return np.nan
    try:
        if '-' in value:
            parts = value.split('-')
            # Convert each part to float after stripping whitespace
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
        else:
            return float(value)
    except Exception:
        return np.nan

grid["Coarse_mid"] = grid.PercentCoarse.apply(calculate_midpoint)
numeric_cols = grid.select_dtypes(include=[np.number]).columns
filtered_numeric_cols = numeric_cols.difference(['latitude', 'longitude', 'Coarse_mid']).to_numpy()
tab_x = list(filtered_numeric_cols)
tab_l = ['latitude', 'longitude']
tab_y = ["Coarse_mid"]
df = grid[~grid.Coarse_mid.isna()]
X, y = df[tab_x + tab_l], df[tab_y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Specify the hyperparameters for the GA model.
# Check the docstring of`GeoAggregator` class for details.
params = {
    'x_cols': tab_x,
    'spa_cols': tab_l,
    'y_cols': tab_y,
    'attn_variant': 'MCPA',
    'model_variant': 'small',
    'd_model': 32,
    # 'n_attn_layer': 1,
    # 'idu_points': 4,
    # 'seq_len': 128,
    'attn_dropout': 0.2,
    'attn_bias_factor': None,
    'reg_lin_dims': [16, 1],
    'epochs': 1,
    'lr': 5e-3,
    'batch_size': 8,
    'verbose': True   # show model summary
}

# Initialize the GA model.
print(f"All data loaded, initialising model. Epochs: {params.get('epochs')}")

model = GARegressor(
    **params
)

# Train the GA model (need to pass co-variates, spatial coordinates and target variable).
model.fit(X=cudf.from_pandas(X_train[tab_x]), l=cudf.from_pandas(X_train[tab_l]), y=cudf.from_pandas(y_train))

y_pred = model.predict(X=X_test[tab_x], l=X_test[tab_l])

print(f'R-sq = {r2_score(y_true=y_test[tab_y], y_pred=y_pred)}')
print(f'MAE = {mean_absolute_error(y_true=y_test[tab_y], y_pred=y_pred)}')