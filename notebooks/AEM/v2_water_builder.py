import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
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
import pickle

with open('grid_large_std_v2.pkl', 'rb') as file:
    grid_avg = pickle.load(file)

NHD_CA = gpd.read_file("NHD_CA.gpkg")
NHD_CA = NHD_CA.to_crs(grid_avg.crs)

SGMA = "../data/GWBasins.shp"
gdf = gpd.read_file(SGMA)
gdf.set_crs(epsg=3857, inplace=True)
gdf.set_index('OBJECTID', inplace=True)
gdf['Basin_Prefix'] = gdf['Basin_Numb'].str.split('-').str[0].astype(int)
gdf = gdf[gdf['Basin_Numb'] == '5-021']

gdf = gdf.to_crs(NHD_CA.crs)

NHD_CA = NHD_CA[NHD_CA.geometry.intersects(gdf.unary_union)]

union_dict = {}
code_description_dict = {}

for (fcode, description), group in tqdm(NHD_CA.groupby(["fcode", "fcode_description"])):
    union_geom = group.geometry.unary_union
    union_dict[fcode] = union_geom
    code_description_dict[fcode] = description

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

def func1(fcode):
    shape = union_dict.get(fcode)
    mask = []
    for index, row in tqdm(grid_avg.iterrows()):
        if row.geometry.intersects(shape):
            mask.append(1)
        else:
            mask.append(0)
    return mask

fcodes = list(union_dict.keys())

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(func1)(num) 
    for num in fcodes
)

with open('water_masks.pkl', 'wb') as file:
    pickle.dump(results, file)