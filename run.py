import numpy as np
import pandas as pd
import xarray as xr
import numpy as np
import xarray as xr
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import logging   
from data_processing import data_processing_pipeline

logger = logging.getLogger(__name__)  

EPOCHS = 500

VAR_CHANNELS = ["delta_qv_physics", "delta_t_physics", "plev_channel"]  # add more variables as needed

LEVEL_DIM = "newlev"
TIME_DIM = "time"
SAMPLE_DIM = "sample"
ITER = 5

ds_gfs = xr.open_dataset("GFS_merged_latid197.nc")
ds_rap = xr.open_dataset("RAP_merged_latid197.nc")

# For one column for now
target_lon = 51.5
# Pick the nearest grid column at that longitude and drop the lon dimension
gfs_col = ds_gfs.sel(lon=target_lon, method="nearest", drop=True)
rap_col = ds_rap.sel(lon=target_lon, method="nearest", drop=True)


model_datasets = {
    "GFS": gfs_col,
    "RAP": rap_col
}

ds_train, ds_validation, ds_test = data_processing_pipeline(model_datasets = model_datasets, VAR_CHANNELS=VAR_CHANNELS, 
                                                            LEVEL_DIM=LEVEL_DIM, TIME_DIM=TIME_DIM, SAMPLE_DIM=SAMPLE_DIM)

