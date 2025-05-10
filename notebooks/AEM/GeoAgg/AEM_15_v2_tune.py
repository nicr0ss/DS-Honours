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

# Determine the number of cores
N_CORES = os.cpu_count()
N_CORES_CV = 40
N_CORES_TRIALS = 8
print(f"Number of available cores: {N_CORES}")
print(f"Using {N_CORES_CV} cores for cross-validation")
print(f"Using {N_CORES_TRIALS} parallel trials")

# Load data
grid = pd.read_parquet('../grid_large_std_v2.parquet')

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

grid = grid.copy()
grid["Coarse_mid"] = grid.PercentCoarse.apply(calculate_midpoint)
numeric_cols = grid.select_dtypes(include=[np.number]).columns
filtered_numeric_cols = numeric_cols.difference(['latitude', 'longitude', 'Coarse_mid']).to_numpy()
tab_x = list(filtered_numeric_cols)
tab_l = ['latitude', 'longitude']
tab_y = ["Coarse_mid"]
df = grid[~grid.Coarse_mid.isna()]
X, y = df[tab_x + tab_l], df[tab_y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("All data loaded.")

def objective(trial, n_split=5):
    """
    Example Optuna objective function that performs k-fold cross validation,
    parallelizing across folds with joblib.Parallel.
    """
    # Suggest hyperparameters
    params = {
        'x_cols':        tab_x,
        'spa_cols':      tab_l,
        'y_cols':        tab_y,
        'attn_variant':  'MCPA',
        'd_model':       trial.suggest_categorical('d_model', [32, 64, 80]),
        'n_attn_layer':  trial.suggest_int('n_attn_layer', 1, 3),
        'idu_points':    trial.suggest_int('idu_points', 2, 8),
        'seq_len':       trial.suggest_categorical('seq_len', [64, 81, 100, 144, 256, 400]),
        'attn_dropout':  trial.suggest_float('attn_dropout', 0.01, 0.5),
        'attn_bias_factor': None,
        'reg_lin_dims':  trial.suggest_categorical('reg_lin_dims', [[1], [4, 1], [16, 1]]),
        'epochs':        trial.suggest_int('epochs', 3, 30),
        'lr':            5e-3,
        'batch_size':    8,
    }
    
    # K-Fold splitter
    kf = KFold(n_splits=n_split, shuffle=True)
    with open("results_v2.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Trial {trial.number} begins at {timestamp}\n")
        f.close()

    def train_and_evaluate_fold(trn_idx, val_idx):
        """
        Train a GARegressor on the fold's training set and return its MAE on the validation set.
        """
        with open("nodes_v2.txt", "a") as f:
            f.write(f"{trial.number} starting CV {trn_idx, val_idx} \n")
            f.close()
        # Split data
        trn_X, trn_y = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        val_X, val_y = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Create and train the model
        model = GARegressor(**params)
        model.fit(
            X=trn_X[tab_x],
            l=trn_X[tab_l],
            y=trn_y
        )

        # Predict and calculate loss
        y_pred = model.predict(
            X=val_X[tab_x],
            l=val_X[tab_l]
        )
        fold_loss = mean_absolute_error(y_true=val_y, y_pred=y_pred)
        with open("r2_v2.txt", "a") as f:
            f.write(f"Trial {trial.params}, CV R^2 {r2_score(val_y, y_pred)}, MAE: {fold_loss} \n")
            f.close()
        return fold_loss

    # Use exactly 5 cores for cross-validation folds
    fold_losses = Parallel(n_jobs=N_CORES_CV)(
        delayed(train_and_evaluate_fold)(trn_idx, val_idx)
        for trn_idx, val_idx in kf.split(X_train, y_train)
    )

    return np.mean(fold_losses)

def logging_callback(study, trial):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write trial details to the text file
    with open("results_v2.txt", "a") as f:
        f.write(f"Trial {trial.number} completed at {timestamp}\n")
        f.write(f"Trial loss (MAE): {trial.value}\n")
        f.write(f"Trial parameters: {trial.params}\n")
        f.write("-" * 40 + "\n")
        f.close()
    
    # Prepare trial data for the CSV file
    trial_data = {
        "Trial": trial.number,
        "Timestamp": timestamp,
        "MAE": trial.value
    }
    
    # Merge trial parameters into the trial data dictionary
    trial_data.update(trial.params)
    
    csv_file = "trial_results.csv"
    if os.path.exists(csv_file):
        temp = pd.read_csv(csv_file)
        # Append the new trial data
        temp = temp.append(trial_data, ignore_index=True)
    else:
        temp = pd.DataFrame([trial_data])
    
    # Save the updated DataFrame to CSV
    temp.to_csv(csv_file, index=False)

sampler = TPESampler()
start_time = time.time()
study = optuna.create_study(
    direction='minimize',
    study_name='ga-hp!',
    sampler=sampler
)

# Run multiple trials in parallel, while each trial uses 5 cores for CV
print(f"Starting optimization with {N_CORES_TRIALS} parallel trials, using {N_CORES_CV} cores for CV")
study.optimize(
    objective, 
    n_trials=150, 
    n_jobs=N_CORES_TRIALS,  # Run this many trials in parallel
    gc_after_trial=True
)

end_time = time.time()

# Optionally, log the overall best results at the end of the experiment
best_params = study.best_params
best_value = study.best_value
best_trial = study.best_trial

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("results_v2.txt", "a") as f:
    f.write(f"Overall experiment completed at {timestamp}\n")
    f.write(f"Total elapsed time = {end_time - start_time:.4f}s\n")
    f.write(f"Best hyperparameters: {best_params}\n")
    f.write(f"Best overall loss (MAE): {best_value}\n")
    f.write(f"Best trial details: {best_trial}\n")
    f.write("=" * 40 + "\n")
    f.close()