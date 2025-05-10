import pandas as pd
import numpy as np
import os
import glob
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
from pyproj import Transformer
import xml.etree.ElementTree as ET
import json
import ipyparallel as ipp
import boto3
import pickle
import io
import geopandas as gpd
from shapely.geometry import Point
from joblib import Parallel, delayed

s3_bucket = 'ds-h-ca-bigdata'
s3_prefix = 'ECOSTRESS/'

session = boto3.Session(profile_name='default')
s3 = session.client('s3', region_name='us-east-2')

response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix, Delimiter='/')
year_folders = [prefix['Prefix'] for prefix in response.get('CommonPrefixes', [])]
print("Year folders found:", year_folders)


yearly_dfs = {}

for folder in year_folders:
    tif_response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=folder, Delimiter='/')
    tif_files = [obj['Key'] for obj in tif_response.get('Contents', []) if obj['Key'].endswith('.tif')]
    df_list = []
    for tif_file in tqdm(tif_files, desc=f"Processing files in {folder}", leave=False):
        s3_path = f'/vsis3/{s3_bucket}/{tif_file}'
        with rasterio.open(s3_path) as src:
            data = src.read(1)
            transform = src.transform
        
        rows, cols = data.shape
        row_inds, col_inds = np.indices((rows, cols))
        xs, ys = rasterio.transform.xy(transform, row_inds, col_inds)
        xs = np.array(xs)
        ys = np.array(ys)
        
        df = pd.DataFrame({
            'x': xs.flatten(),
            'y': ys.flatten(),
            'value': data.flatten()
        })
        df_list.append(df)
    
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        parts = folder.split('/')
        if len(parts) >= 2:
            year = parts[1]
        else:
            year = folder
        yearly_dfs[year] = combined_df

central_5021 = pd.read_csv("../data/almonds.csv")

readings_df = {}

for year, df in tqdm(yearly_dfs.items()):
    readings_df[year] = df[(df.value != -9999999827968.0) & (~df.value.isna())]

def create_point(x, y):
    return Point((x, y))

points_dict = {}

for key, df in readings_df.items():
    points = Parallel(n_jobs=-1)(
        delayed(create_point)(x, y)
        for x, y in tqdm(zip(df['x'], df['y']), total=len(df), desc=f"Processing {key}")
    )
    points_dict[key] = points

ecostress_gdf = {}

for year, df in tqdm(readings_df.items()):
    ecostress_gdf[year] = gpd.GeoDataFrame(df, geometry=points_dict.get(year), crs="EPSG:4326")

central_gdf = gpd.GeoDataFrame(
    central_5021,
    geometry=gpd.points_from_xy(central_5021.longitude, central_5021.latitude),
    crs="EPSG:4326"
)
central_gdf = central_gdf.to_crs("EPSG:3857")

central_gdf['buffer'] = central_gdf.geometry.buffer(1000)
buffers_gdf = central_gdf.copy()
buffers_gdf = buffers_gdf.set_geometry('buffer')

for year, points_gdf in tqdm(ecostress_gdf.items()):
    buffers_gdf = buffers_gdf.to_crs(points_gdf.crs)
    joined = gpd.sjoin(points_gdf, buffers_gdf, how="inner", predicate="within")
    mean_values = joined.groupby("index_right")["value"].mean()
    col_name = f"evapo_{year}"
    buffers_gdf.loc[mean_values.index, col_name] = mean_values

buffers_gdf.to_pickle("almonds_ECOSTRESS.pkl")