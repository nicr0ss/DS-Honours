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

with open("../grid_large_std.pkl", "rb") as f:
    grid = pickle.load(f)

with open("AEM_15_model_v1_train.pkl", "rb") as f:
    model = pickle.load(f)

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

X, y = grid[tab_x + tab_l], grid[tab_y]

y_pred = model.predict(X=X[tab_x], l=X[tab_l])

preds = pd.concat([X[tab_l], y], axis=1)
preds["pred"] = y_pred
preds.to_csv("v1_train_grid.csv")