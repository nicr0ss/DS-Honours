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
from joblib import Parallel, delayed
from datetime import datetime  # For timestamp logging
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

grid = pd.read_parquet('../deeper/PercentCoarse_15_30.parquet')
print(len(grid))

grid = grid.copy()
grid["Coarse_mid"] = grid.PercentCoarse_15_30
grid = grid.drop(columns=["PercentCoarse_15_30"])
numeric_cols = grid.select_dtypes(include=[np.number]).columns
filtered_numeric_cols = numeric_cols.difference(['latitude', 'longitude', 'Coarse_mid']).to_numpy()
tab_x = list(filtered_numeric_cols)
tab_l = ['latitude', 'longitude']
tab_y = ["Coarse_mid"]
df = grid[~grid.Coarse_mid.isna()]
X, y = df[tab_x + tab_l], df[tab_y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("All data loaded.")

tensors = {"15_30" : pd.read_parquet('../deeper/PercentCoarse_15_30_NaDROP.parquet'),
          "30_45" : pd.read_parquet('../deeper/PercentCoarse_30_45_NaDROP.parquet'),
          "45_60" : pd.read_parquet('../deeper/PercentCoarse_45_60_NaDROP.parquet'),
          "60_75" : pd.read_parquet('../deeper/PercentCoarse_60_75_NaDROP.parquet'),
          "75_90" : pd.read_parquet('../deeper/PercentCoarse_75_90_NaDROP.parquet'),
          "90_105" : pd.read_parquet('../deeper/PercentCoarse_90_105_NaDROP.parquet'),
          "105_120" : pd.read_parquet('../deeper/PercentCoarse_105_120_NaDROP.parquet'),
          "120_135" : pd.read_parquet('../deeper/PercentCoarse_120_135_NaDROP.parquet'),
          "135_150" : pd.read_parquet('../deeper/PercentCoarse_135_150_NaDROP.parquet')
          }

params_v2 = {"15_30" : {'x_cols': tab_x, 'spa_cols': tab_l, 'y_cols': tab_y, 'attn_variant': 'MCPA', 
                     'attn_bias_factor': None, 'lr': 5e-3, 'batch_size': 8, 'd_model': 32, 'n_attn_layer': 3,
                     'idu_points': 7, 'seq_len': 400, 'attn_dropout': 0.08416754053841363, 'reg_lin_dims': [1],
                     'epochs': 26},
         "30_45" : {'x_cols': tab_x, 'spa_cols': tab_l, 'y_cols': tab_y, 'attn_variant': 'MCPA', 
                     'attn_bias_factor': None, 'lr': 5e-3, 'batch_size': 8, 'd_model': 32, 'n_attn_layer': 2,
                    'idu_points': 4, 'seq_len': 81, 'attn_dropout': 0.11051365611964778, 'reg_lin_dims': [16, 1],
                    'epochs': 15},
         "45_60" : {'x_cols': tab_x, 'spa_cols': tab_l, 'y_cols': tab_y, 'attn_variant': 'MCPA', 
                     'attn_bias_factor': None, 'lr': 5e-3, 'batch_size': 8, 'd_model': 64, 'n_attn_layer': 2, 
                    'idu_points': 6, 'seq_len': 81, 'attn_dropout': 0.271551633680783, 'reg_lin_dims': [16, 1],
                    'epochs': 22},
         "60_75" : {'x_cols': tab_x, 'spa_cols': tab_l, 'y_cols': tab_y, 'attn_variant': 'MCPA', 
                     'attn_bias_factor': None, 'lr': 5e-3, 'batch_size': 8, 'd_model': 32, 'n_attn_layer': 1,
                    'idu_points': 3, 'seq_len': 81, 'attn_dropout': 0.26988506055372774, 'reg_lin_dims': [1],
                    'epochs': 29},
         "75_90" : {'x_cols': tab_x, 'spa_cols': tab_l, 'y_cols': tab_y, 'attn_variant': 'MCPA', 
                     'attn_bias_factor': None, 'lr': 5e-3, 'batch_size': 8, 'd_model': 32, 'n_attn_layer': 2,
                    'idu_points': 2, 'seq_len': 64, 'attn_dropout': 0.2662901952622706, 'reg_lin_dims': [16, 1],
                    'epochs': 14},
         "90_105" : {'x_cols': tab_x, 'spa_cols': tab_l, 'y_cols': tab_y, 'attn_variant': 'MCPA', 
                     'attn_bias_factor': None, 'lr': 5e-3, 'batch_size': 8, 'd_model': 32, 'n_attn_layer': 1,
                     'idu_points': 8, 'seq_len': 64, 'attn_dropout': 0.19717295358847917, 'reg_lin_dims': [16, 1],
                     'epochs': 14},
         "105_120" : {'x_cols': tab_x, 'spa_cols': tab_l, 'y_cols': tab_y, 'attn_variant': 'MCPA', 
                     'attn_bias_factor': None, 'lr': 5e-3, 'batch_size': 8, 'd_model': 32, 'n_attn_layer': 2,
                      'idu_points': 8, 'seq_len': 64, 'attn_dropout': 0.14848092149309738, 'reg_lin_dims': [16, 1],
                      'epochs': 30},
         "120_135" : {'x_cols': tab_x, 'spa_cols': tab_l, 'y_cols': tab_y, 'attn_variant': 'MCPA', 
                     'attn_bias_factor': None, 'lr': 5e-3, 'batch_size': 8, 'd_model': 64, 'n_attn_layer': 2,
                      'idu_points': 3, 'seq_len': 400, 'attn_dropout': 0.4881817208983043, 'reg_lin_dims': [16, 1],
                      'epochs': 30},
         "135_150" : {'x_cols': tab_x, 'spa_cols': tab_l, 'y_cols': tab_y, 'attn_variant': 'MCPA', 
                     'attn_bias_factor': None, 'lr': 5e-3, 'batch_size': 8, 'd_model': 64, 'n_attn_layer': 2,
                      'idu_points': 4, 'seq_len': 64, 'attn_dropout': 0.11996337310454076, 'reg_lin_dims': [4, 1],
                      'epochs': 17}
            }

def train_eval(slice):
    path = str(slice) + "/"
    grid = tensors.get(slice)
    params = params_v2.get(slice)
    grid["Coarse_mid"] = grid["PercentCoarse_" + slice]
    grid = grid.drop(columns=["PercentCoarse_" + slice])
    numeric_cols = grid.select_dtypes(include=[np.number]).columns
    filtered_numeric_cols = numeric_cols.difference(['latitude', 'longitude', 'Coarse_mid']).to_numpy()
    tab_x = list(filtered_numeric_cols)
    tab_l = ['latitude', 'longitude']
    tab_y = ["Coarse_mid"]
    df = grid[~grid.Coarse_mid.isna()]
    X, y = df[tab_x + tab_l], df[tab_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GARegressor(**params)
    with open("v2_full.txt", "a") as f:
        f.write(f"train {slice} \n")
        f.close()
    model.fit(X=X_train[tab_x], l=X_train[tab_l], y=y_train)
    y_pred = model.predict(X=X_test[tab_x], l=X_test[tab_l])
    tests = pd.concat([X_test[tab_l], y_test], axis=1)
    tests["pred"] = y_pred
    tests.to_csv(path + "v2_test.csv")

def full_pred(slice):
    path = str(slice) + "/"
    grid = tensors.get(slice)
    params = params_v2.get(slice)
    grid["Coarse_mid"] = grid["PercentCoarse_" + slice]
    grid = grid.drop(columns=["PercentCoarse_" + slice])
    numeric_cols = grid.select_dtypes(include=[np.number]).columns
    filtered_numeric_cols = numeric_cols.difference(['latitude', 'longitude', 'Coarse_mid']).to_numpy()
    tab_x = list(filtered_numeric_cols)
    tab_l = ['latitude', 'longitude']
    tab_y = ["Coarse_mid"]
    df = grid[~grid.Coarse_mid.isna()]
    X, y = df[tab_x + tab_l], df[tab_y]
    model = GARegressor(**params)
    with open("v2_full.txt", "a") as f:
        f.write(f"test {slice} \n")
        f.close()
    model.fit(X=X[tab_x], l=X[tab_l], y=y)
    df = grid
    X, y = df[tab_x + tab_l], df[tab_y]
    y_pred = model.predict(X=X[tab_x], l=X[tab_l])
    tests = pd.concat([X[tab_l]], axis=1)
    tests["pred"] = y_pred
    tests.to_csv(path + "v2_full_grid.csv")

keys = list(tensors.keys())
tasks = []
for key in keys:
#    tasks.append(delayed(train_eval)(key))
    tasks.append(delayed(full_pred)(key))

Parallel(n_jobs=-1)(tasks)