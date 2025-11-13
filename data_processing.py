#TODO add far more comments and docstrings
import numpy as np
import pandas as pd
import xarray as xr
import numpy as np
import xarray as xr
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import logging   

logging.basicConfig(
    level=logging.INFO,                           
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                   
        logging.FileHandler("CNN.log")    
    ]
)

# Data preparation 
def set_time_coordinates(ds, start_time="2016-08-11T00:00", freq="3H"):
    """
    Assigns a real UTC time coordinate to the dataset
    Args:
        ds: Input dataset that has a dimension named 'init_time'.
        start_time: The start timestamp in ISO format (default '2016-08-11T00:00').
        freq: Time step spacing, e.g. '3H' for 3-hourly data.

    Returns:
        ds: Dataset with a new 'time' coordinate and 'init_time' replaced.

    """
    n = ds.sizes["init_time"]
    t0 = np.datetime64(start_time)
    time = pd.date_range(t0, periods=n, freq=freq).to_numpy()

    ds = ds.assign_coords(time=("init_time", time))
    ds = ds.swap_dims({"init_time": "time"})
    return ds


def apply_misc_changes(ds: xr.Dataset) -> xr.Dataset:
    """
    MISC data edits
      1. Removes final 4 timesteps (indices 241–244).
      2. Sets all data at 2016-08-31 18:00 UTC to NaN.

    Args:
        ds: Dataset that has a 'time' coordinate (output of set_time_coordinates).

    Returns
        ds: Modified dataset.
    """
    # Remove final timesteps
    ds = ds.isel(time=slice(0, -4))

    # Set 2016-08-31 18:00 UTC to NaN
    target_time = np.datetime64("2016-08-31T18:00")
    if target_time in ds.time.values:
        ds = ds.copy() 
        ds.loc[dict(time=target_time)] = np.nan

    return ds



def compute_hourly_level_stats(train_ds: xr.Dataset, time_dim="time", level_dim="newlev"):
    """
    TODO add datatype in funciton header
    Compute mean/std for each UTC hour × pressure level from TRAIN set.
    Only variables with both time and level dims are handled.
    """
    stats = {}
    for var in train_ds.data_vars:
        da = train_ds[var]
        if (np.issubdtype(da.dtype, np.number) and time_dim in da.dims and level_dim in da.dims):

            # TODO maybe make more flexible, i.e. add in multiple channels
            reduce_dims = [i for i in da.dims if i not in (time_dim, level_dim)]

            g = da.groupby(f"{time_dim}.hour")      
            mean = g.mean(dim=[time_dim] + reduce_dims, skipna=True)
            std  = g.std (dim=[time_dim] + reduce_dims, skipna=True)

            stats[var] = {"mean": mean, "std": std, "level_dim": level_dim}
    return stats


def apply_hourly_level_normalisation(ds: xr.Dataset, stats: dict, time_dim="time", level_dim="newlev"):
    """
    Apply joint (UTC hour × pressure level) normalisation using TRAIN stats.
    """
    ds_norm = ds.copy()
    hour_idx = xr.DataArray(ds[time_dim].dt.hour.values, dims=time_dim, name="hour") 

    for var, s in stats.items():
        # Safety thing for liekly future use of this function
        if var not in ds_norm:
            continue
        da = ds_norm[var]
        if time_dim in da.dims and level_dim in da.dims:
            mean = s["mean"].sel(hour=hour_idx)
            std  = s["std"].sel(hour=hour_idx)
            safe_std = xr.where((std == 0) | (~np.isfinite(std)), 1.0, std)
            ds_norm[var] = (da - mean) / safe_std
    return ds_norm
#TODO add the moramlisation per plevel!!!!!!

# TODO: maybe remove
def add_positional_channel(ds, p_norm, name="plev_channel"):
    shape = (ds.sizes["sample"], ds.sizes["time"], ds.sizes["newlev"])
    p3 = np.broadcast_to(p_norm, shape).astype("float32")
    da = xr.DataArray(p3, dims=("sample","time","newlev"))
    return ds.assign(**{name: da})


# TODO diagnostoc tool, could delete or move somehwere
def compute_nan_fraction(
    ds: xr.Dataset,
    var_names,
    sample_dim="sample",
    time_dim="time",
    level_dim="newlev",
):
    """
    Compute NaN diagnostics for the given variables.
    """
    # Gather per-var arrays shaped [S, T, L]
    arrays = []
    for v in var_names:
        if v in ds and (sample_dim in ds[v].dims and time_dim in ds[v].dims and level_dim in ds[v].dims):
            arrays.append(ds[v].transpose(sample_dim, time_dim, level_dim).values)
        else:
            raise ValueError(f"{v} missing or has wrong dims in dataset.")

    # Stack to [S, T, C, L]
    data = np.stack(arrays, axis=2)

    # Fraction of NaNs over C×L for each (S,T)
    nan_fraction = np.isnan(data).mean(axis=(2, 3))  # [S, T] in [0,1]

    # Any-NaN boolean over channels×levels for each (S,T)
    any_nan_bool = np.isnan(data).any(axis=(2, 3))   # [S, T], True if any missing

    # Wrap as DataArray for convenience
    nan_frac_da = xr.DataArray(
        nan_fraction,
        dims=(sample_dim, time_dim),
        coords={sample_dim: ds[sample_dim], time_dim: ds[time_dim]},
        name="nan_fraction"
    )

    any_nan_da = xr.DataArray(
        any_nan_bool,
        dims=(sample_dim, time_dim),
        coords={sample_dim: ds[sample_dim], time_dim: ds[time_dim]},
        name="has_any_nan"
    )

    # Global stats
    S = data.shape[0]
    T = data.shape[1]
    total_pairs = int(S * T)
    pairs_with_any_nan = int(any_nan_bool.sum())
    pairs_with_any_nan_pct = 100.0 * pairs_with_any_nan / max(1, total_pairs)

    flat = nan_fraction.reshape(-1)
    mean_pct = float(flat.mean() * 100.0)
    std_pct  = float(flat.std()  * 100.0)
    max_pct  = float(flat.max()  * 100.0)

    # Per-sample (# of times with any NaN) and per-time (# of samples with any NaN)
    per_sample_any_nan_count = any_nan_da.sum(dim=time_dim).astype("int64")  # [sample]
    per_time_any_nan_count   = any_nan_da.sum(dim=sample_dim).astype("int64")# [time]

    summary = {
        "mean_%": mean_pct,
        "std_%":  std_pct,
        "max_%":  max_pct,
        "total_pairs": total_pairs,
        "pairs_with_any_nan": pairs_with_any_nan,
        "pairs_with_any_nan_%": pairs_with_any_nan_pct,
        "per_sample_any_nan_count": per_sample_any_nan_count, 
        "per_time_any_nan_count":   per_time_any_nan_count,  
    }

    
    return nan_frac_da, summary, any_nan_da







#  build (X, y) from an xarray.Dataset
def build_xy_from_xr(ds: xr.Dataset, var_names, sample_dim = "sample", level_dim="newlev", time_dim="time"):
    """
    Essentialy comverts an xarray.Dataset to (X, y) numpy arrays for model training for pytorch.
    Returns:
      X: np.ndarray of shape [N_examples, C, L]
      y: np.ndarray of shape [N_examples]  (class id from model_id)
    One example per (model, time).
    """
    
    # Safety checks
    # TODO this needed??
    for v in var_names:
        if v not in ds:
            raise ValueError(f"Variable '{v}' not found in dataset.")

    # Extract labels from the dataset coord 'model_id' (per-sample)
    # We will broadcast to (sample, time)
    model_ids = ds["model_id"].values  # shape [n_models]
    n_models = model_ids.shape[0]
    n_times  = ds.sizes[time_dim]
    n_levels = ds.sizes[level_dim]
    n_channels = len(var_names)
    

    # Stack per (model, time)
    # Build data array per variable -> shape [sample, time, newlev]
    arrays = []
    for v in var_names:
        a = ds[v].transpose(sample_dim, time_dim, level_dim).values  # [S, T, L]
        arrays.append(a)
    # Stack channels -> [S, T, C, L]
    data = np.stack(arrays, axis=2)

    # Flatten (model, time) -> N examples
    X = data.reshape(n_models * n_times, n_channels, n_levels)

    # Label per (model, time): repeat model_id across time
    y = np.repeat(model_ids, n_times)


    # TODO make smth more sophsitcated maybe by adding another channel
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X.astype(np.float32), y.astype(np.int64)


# TODO not sure where in thepipeline this should be located
class ColumnDataset(Dataset):
    def __init__(self, X, y, var_names, level_coord):
        self.X = torch.from_numpy(X)  
        self.y = torch.from_numpy(y)  
        self.var_names = var_names
        self.level_coord = np.asarray(level_coord)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    @classmethod
    def from_xr(cls, ds, var_names):
        X, y = build_xy_from_xr(ds, var_names)
        level_coord = ds["plev"].values
        return cls(X, y, var_names,level_coord)

def data_processing_pipeline(model_datasets,
                              VAR_CHANNELS, LEVEL_DIM, TIME_DIM, SAMPLE_DIM):
    """
    TODO ADD DESCRIPTION-combine normalise
    """

    labelled_datasets = []

    for idx, (name, ds) in enumerate(model_datasets.items()):
        # assign both numeric and string labels
        ds_labeled = ds.assign_coords(model_id=idx, model_name=name)
        labelled_datasets.append(ds_labeled)


    # concatenate along a new dimension 
    ds_combined = xr.concat(labelled_datasets, dim="sample")


    ds_combined = set_time_coordinates(ds_combined)
    ds_combined = apply_misc_changes(ds_combined)


    # Split validation/train
    ds_train = ds_combined.isel(time=slice(0, 176))
    ds_validation  = ds_combined.isel(time=slice(177, 216))
    ds_test  = ds_combined.isel(time=slice(217, 240)) 

    #Add normalised pressure coord, a height hint for the CNN, note that I think maybe the pooling later will remove some of this info, so this is essentially adding t back in
    p = ds_train["plev"].values.astype("float32")
    #p_norm = (p - p.mean()) / (p.std() + 1e-8)

    ds_train = add_positional_channel(ds_train, p, "plev_channel")
    ds_validation  = add_positional_channel(ds_validation,  p, "plev_channel")
    ds_test  = add_positional_channel(ds_test,  p, "plev_channel")

    #Normalise
    norm_stats = compute_hourly_level_stats(ds_train)
    ds_train_norm = apply_hourly_level_normalisation(ds_train, norm_stats)
    ds_validation_norm = apply_hourly_level_normalisation(ds_validation, norm_stats)
    ds_test_norm = apply_hourly_level_normalisation(ds_test, norm_stats)



    # NAN stats - SORT!!!!!!!, probably delte for memory reasons later TODO
    nan_frac_train, train_summary, train_any = compute_nan_fraction(ds_train_norm, VAR_CHANNELS)
    nan_frac_validation,  validation_summary,  validation_any  = compute_nan_fraction(ds_validation_norm,  VAR_CHANNELS)
    nan_frac_test,  test_summary,  test_any  = compute_nan_fraction(ds_test_norm,  VAR_CHANNELS)

    # TODO add nan_frac to ds??

    # Log both % and counts
    logger.info(
        "Train NaN fraction: mean %.2f%% ± %.2f%% (max %.2f%%) | pairs_with_any_nan=%d/%d (%.2f%%)",
        train_summary["mean_%"], train_summary["std_%"], train_summary["max_%"],
        train_summary["pairs_with_any_nan"], train_summary["total_pairs"], train_summary["pairs_with_any_nan_%"]
    )
    logger.info(
        "validation  NaN fraction: mean %.2f%% ± %.2f%% (max %.2f%%) | pairs_with_any_nan=%d/%d (%.2f%%)",
        validation_summary["mean_%"],  validation_summary["std_%"],  validation_summary["max_%"],
        validation_summary["pairs_with_any_nan"], validation_summary["total_pairs"], validation_summary["pairs_with_any_nan_%"]
    )
    logger.info(
        "Test  NaN fraction: mean %.2f%% ± %.2f%% (max %.2f%%) | pairs_with_any_nan=%d/%d (%.2f%%)",
        test_summary["mean_%"],  test_summary["std_%"],  test_summary["max_%"],
        test_summary["pairs_with_any_nan"], test_summary["total_pairs"], test_summary["pairs_with_any_nan_%"]
    )

    train_ds = ColumnDataset.from_xr(ds_train, VAR_CHANNELS)
    validation_ds  = ColumnDataset.from_xr(ds_validation, VAR_CHANNELS)
    test_ds  = ColumnDataset.from_xr(ds_test, VAR_CHANNELS)


    

    return train_ds, validation_ds, test_ds



if __name__ == "__main__":
    logger = logging.getLogger(__name__)  

    EPOCHS = 500

    VAR_CHANNELS = ["delta_qv_physics", "delta_t_physics", "plev_channel"]  # add more variables as needed

    LEVEL_DIM = "newlev"
    TIME_DIM = "time"
    SAMPLE_DIM = "sample"

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

    logger.info("Train dataset size: %d samples", len(ds_train))
    logger.info("Validation dataset size: %d samples", len(ds_validation))  
    logger.info("Test dataset size: %d samples", len(ds_test))





