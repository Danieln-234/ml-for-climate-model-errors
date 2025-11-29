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
from run_cnn_single_column import run_cnn_ig_pipeline

def run_long_pipeline(ds_gfs, ds_rap, VAR_CHANNELS, EPOCHS, device,
                      run_count, return_all=True, compute_correlations=False):
    model_stats_columns = []
    ig_stats_columns = []
    lon_vals = [] #TODO slightly unsure but I believe it is god to expliciltyindex the lon loactions just in case confusion later

    # TODO this is hardcoded, please put higher level in config maybe 
    lons = ds_gfs.lon.sel(lon=slice(51.1, 94)).values

    for lon in lons:
        logger.info(f"Processing longitude: {lon}")
        lon_val = float(lon)
        gfs_col = ds_gfs.sel(lon=lon, method="nearest", drop=True) # nearest due to float issues, maybe not needed with above lon float thing idkkkk
        rap_col = ds_rap.sel(lon=lon, method="nearest", drop=True)
        model_datasets = {
            "GFS": gfs_col,
            "RAP": rap_col
        }

        ds_train, ds_validation, ds_test = data_processing_pipeline(model_datasets = model_datasets, var_channels=VAR_CHANNELS)

        model_stats_summary, mean_ig, std_ig, ig_stability, corr_mean, corr_std = run_cnn_ig_pipeline(
            ds_train, ds_validation, ds_test, VAR_CHANNELS,
            EPOCHS, device, run_count,
            return_all=return_all,
            compute_correlations=compute_correlations,
            logger=logger,
        ) 
        lon_vals.append(lon_val)
        model_stats_columns.append(model_stats_summary)
        ig_stats_columns.append(mean_ig)

    ig_stats_columns = np.stack(ig_stats_columns, axis=0)
    lon_vals = np.array(lon_vals)
    #TODO
    keys = model_stats_columns[0].keys()
    model_stats_columns = {
        k: np.array([d[k] for d in model_stats_columns])
        for k in keys
    }
    level_coord_values = ds_test.level_coord 
    return model_stats_columns, ig_stats_columns, lon_vals, level_coord_values

def compute_peak_plev_per_lon(ig_stats_columns, lon_vals, plev_vals, VAR_CHANNELS,
                              var_name):
    """TODO
    """
    c_idx = VAR_CHANNELS.index(var_name)

    # [N_lon, L] for that channel
    mean_ig_for_channel = ig_stats_columns[:, c_idx, :]

    # index of max IG along vertical
    max_lev_idx = mean_ig_for_channel.argmax(axis=1)  

    # map to actual pressure levels
    peak_plev = plev_vals[max_lev_idx]                

    return peak_plev

import matplotlib.pyplot as plt

def plot_peak_plev_vs_lon(lon_vals, peak_plev, var_name):
    plt.figure(figsize=(8, 4))
    #plt.scatter(lon_vals, peak_plev, s=25)
    plt.plot(lon_vals, peak_plev, alpha=0.5)
    

    plt.xlabel("Longitude")
    plt.ylabel("Pressure level of peak |IG|")
    plt.title(f"Level of maximum mean |IG| vs longitude\n({var_name})")

    # plev is in hPa with larger values near surface:
    plt.gca().invert_yaxis()

    plt.grid(True)
    plt.tight_layout()


def plot_acc_loss_vs_lon(lon_vals, model_stats_columns):
    plt.figure(figsize=(8, 4))

    # Accuracy
    plt.plot(lon_vals, model_stats_columns["mean_acc"], alpha=0.7, label="Mean acc")
    plt.fill_between(lon_vals, model_stats_columns["mean_acc"] - model_stats_columns["std_acc"], model_stats_columns["mean_acc"] + model_stats_columns["std_acc"], alpha=0.3, label="±1σ of the 20 runs")

    # Mean loss
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(lon_vals, model_stats_columns["mean_loss"], alpha=0.7, color="red", linestyle="--", label="Mean loss")
    plt.fill_between(lon_vals, model_stats_columns["mean_loss"] - model_stats_columns["std_loss"], model_stats_columns["mean_loss"] + model_stats_columns["std_loss"], alpha=0.3, color="red", label="±1σ of the 20 runs")
    

    plt.xlabel("Longitude")
    plt.ylabel("Accuracy")
    plt.title("Column CNN accuracy vs longitude")
    plt.grid(True)
    plt.tight_layout()




if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 500
    VAR_CHANNELS = ["delta_qv_physics", "delta_t_physics", "plev_channel"]
    run_count = 1
    compute_correlations = False
    return_all = True
    run_count = 20

    ds_gfs = xr.open_dataset("GFS_merged_latid197.nc")
    ds_rap = xr.open_dataset("RAP_merged_latid197.nc")

    model_stats_columns, ig_stats_columns, lon_vals, plev_vals = run_long_pipeline(
        ds_gfs=ds_gfs,
        ds_rap=ds_rap,
        run_count = run_count,
        VAR_CHANNELS=VAR_CHANNELS,
        EPOCHS=EPOCHS,
        device=device,
        return_all=return_all,
        compute_correlations=compute_correlations,
    )

    # Plot per channel
    for var_name in VAR_CHANNELS:
        peak_plev = compute_peak_plev_per_lon(
            ig_stats_columns, lon_vals, plev_vals, VAR_CHANNELS, var_name=var_name
        )
        plot_peak_plev_vs_lon(lon_vals, peak_plev, var_name)

    #TODO: hmmmm a bit messy with plev level being attributed to IG stats instead of gloabl thing. Practically is fine, but for neatness hmmmm
    plot_acc_loss_vs_lon(lon_vals, model_stats_columns)
    plt.show()
    





