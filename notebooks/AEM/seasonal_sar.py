import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer
import ipyparallel as ipp
from scipy.stats import linregress
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
import xml.etree.ElementTree as ET
import json
from pyproj import Transformer
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import pyproj
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm_joblib import tqdm_joblib

s3_sar_5021 = "https://ds-h-ca-bigdata.s3.us-east-2.amazonaws.com/CA_DWR_VERT.csv"
sar_5021 = pd.read_csv(s3_sar_5021)

date_cols = sorted(
    [c for c in sar_5021.columns if c.startswith('D')],
    key=lambda c: pd.to_datetime(c[1:], format='%Y%m%d')
)

abs_vals = sar_5021[date_cols].cumsum(axis=1)
abs_vals.columns = pd.to_datetime([c[1:] for c in date_cols], format='%Y%m%d')

abs_df = pd.concat([sar_5021[['LAT','LON']].reset_index(drop=True),
                    abs_vals.reset_index(drop=True)], axis=1)

def seasonality(row, period=61):
    date_cols = [c for c in row.index if isinstance(c, pd.Timestamp)]
    ts = pd.Series(
        data=row.loc[date_cols].values,
        index=pd.DatetimeIndex(date_cols)
    ).sort_index().dropna()
    if len(ts) < 2 * period:
        return np.nan
    result = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")
    seasonal = result.seasonal.dropna()
    amplitude = seasonal.max() - seasonal.min() 
    return amplitude

n_rows = len(abs_df)
chunk_size = 100
amplitudes = []

with tqdm(total=n_rows, desc="Computing seasonal amplitude") as pbar:
    for start in range(0, n_rows, chunk_size):
        chunk = abs_df.iloc[start : start + chunk_size]
        chunk_amps = Parallel(n_jobs=-1)(
            delayed(seasonality)(row, 61) 
            for _, row in chunk.iterrows()
        )
        amplitudes.extend(chunk_amps)
        pbar.update(len(chunk))

abs_df["seasonal_amplitude"] = amplitudes
abs_df[["LAT", "LON", "seasonal_amplitude"]].to_csv("seasonal_SAR.csv")