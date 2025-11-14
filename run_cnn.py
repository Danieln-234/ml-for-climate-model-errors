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

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
from captum.attr import IntegratedGradients


logging.basicConfig(
    level=logging.INFO,                           
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                    
    ]
)


class SimpleLevelCNN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            

            nn.Conv1d(16, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            

            nn.AdaptiveAvgPool1d(1),  

        )
        self.head = nn.Linear(32, n_classes)

    def forward(self, x):
        z = self.net(x)           # [B, 32, 1]
        z = z.squeeze(-1)         # [B, 32]
        logits = self.head(z)     # [B, n_classes]
        return logits

class EarlyStopping:
    """Early stopping utility to stop training when validation loss does not improve."""
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.count = 0
        self.best_state = None
        self.best_epoch = None

    def step(self, val_loss, model, epoch):
        improved = (self.best - val_loss) > self.min_delta
        if improved:
            self.best = val_loss
            self.count = 0
            self.best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
            self.best_epoch = epoch

        else:
            self.count += 1
        return self.count >= self.patience

    def restore(self, model):
        """Restore the best model weights saved during training."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)



def evaluate(loader, model, criterion, device):
    """"helpter function to evaluate model on a given dataset loader.For the CNN"""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * yb.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    avg_loss = loss_sum / total
    accuracy = correct / total
    return avg_loss, accuracy




def run_single_cnn(ds_train, ds_validation, ds_test, var_channels, epochs, device, logger=logging.getLogger(__name__)):
    """"Run a single training of the CNN model and return the trained model along with test loss and accuracy.
    """
    # Build datasets
    # Careful if this adds to the time bc can dvide by the number of times we repeat the CNN per column
    train_loader = DataLoader(ds_train, batch_size=128, shuffle=True)
    validation_loader  = DataLoader(ds_validation,  batch_size=256, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=256, shuffle=False)


    # Input: [B, C, L] where C=no. channels (2 here), L=no. levels (46)
    num_classes = len(np.unique(ds_train.y))

    #num_classes = len(torch.unique(ds_train.y))

    model = SimpleLevelCNN(in_channels=len(var_channels), n_classes=num_classes)

    # Train
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # CHECK other options
        factor=0.5,      
        patience=20,
        min_lr=1e-6
    )
    early_stop = EarlyStopping(patience=40, min_delta=0.0)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * yb.size(0)

        train_loss = running / len(ds_train)
        val_loss, val_acc = evaluate(validation_loader, model, criterion, device)

        # Step the LR scheduler based on validation loss
        scheduler.step(val_loss)

        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:02d}, train_loss={train_loss:.4f}, validation_loss={val_loss:.4f}, validation_acc={val_acc:.3f}, {current_lr:.3f}"
        )

        #Early stopping check
        should_stop = early_stop.step(val_loss, model, epoch)
        if should_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    early_stop.restore(model)

    # Evaluation
    model.eval()
    # Full test-set metrics

    test_loss, test_acc = evaluate(test_loader, model, criterion, device)
    logger.info("FINAL TEST, loss=%.4f, acc=%.3f", test_loss, test_acc)

    return model, test_loss, test_acc

def compute_mean_variable_profile(ds_test, device):
    """Compute mean and std of input variables over the test dataset. This is only for the IG (TODO hmmm maybe plots instead in the single IG, less complicated) which is done over the total test ds"""
    X_list = []

    for i in range(len(ds_test)):
        X, y = ds_test[i]              # X: [C, L], already on CPU tensor I assume
        X_list.append(X.numpy())       

    X_all = np.stack(X_list, axis=0)   # [N, C, L]
    mean_vars = X_all.mean(axis=0)     # [C, L]
    std_vars  = X_all.std(axis=0)      # [C, L]

    return mean_vars, std_vars


def run_single_integrated_gradients(model, ds_test, device, logger, return_all=False):
    """Run Integrated Gradients on the test dataset and return mean and std of attributions over test samples."""
    model.eval()
    integrated_gradients = IntegratedGradients(model)
    attribution_array = []

    # Iterate over the test samples, then take average of attributiuons and std, along with average variables
    for i in range(len(ds_test)):
        X, y = ds_test[i]
        with torch.no_grad():
            logits = model(X.unsqueeze(0).to(device))
            target_idx = logits.argmax(dim=1).item()

        input_img = X.unsqueeze(0).to(device)

 
        baseline = torch.zeros_like(input_img) 

        attributions_ig = integrated_gradients.attribute(
            input_img,
            baselines=baseline,
            target=target_idx,
            n_steps=200     # TODO: pick based on computaiton complexoty
        )
        reduced_attr = attributions_ig.squeeze(0)
        reduced_attr = reduced_attr.abs() 
        attribution_array.append(reduced_attr.cpu().detach().numpy())

        logger.info("input_img shape: %s", tuple(input_img.shape))
        logger.info("attributions_ig shape: %s", tuple(attributions_ig.shape))
        
    all_attrs = np.stack(attribution_array, axis=0)  
    mean_over_samples = all_attrs.mean(axis=0) 
    std_over_samples  = all_attrs.std(axis=0) 

    if return_all:
        return mean_over_samples, std_over_samples, all_attrs
    else:
        return mean_over_samples, std_over_samples



def run_cnn_ig_pipeline(ds_train, ds_validation, ds_test, var_channels, epochs, device, run_count, compute_correlations=False, logger=logging.getLogger(__name__)):
    """TODO: Be very careful about where the absolutes are, they seem very arbtritray and ad hoc where placed, maybe centralise"""
    #TODO Do loop, also maybe split CNN training into separate script????
    model_run_stats = []
    ig_stats = []
    corr_runs = []    # correlation matrix for abs(IG) vs abs(variables)

    # NB this only gets the first column sample correlation matrix- otherwise we get too many and it becomes like memory inetnsive
    rep_all_attrs = None
    rep_all_inputs = None 



    for run_idx in range(run_count):
        logger.info("Starting CNN run %d of %d", run_idx + 1, run_count)
        model, test_loss, test_acc = run_single_cnn(
            ds_train,
            ds_validation,
            ds_test,
            var_channels,
            epochs, device, logger
        )
        model_run_stats.append({
            "run_idx": run_idx,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        # Option to get correlation-  nb this is optional as memory intensive for when we run magnitudes more samples
        #TODO COME BACK to his, I feel like this may need expanding
        if compute_correlations and rep_all_attrs is None:
            # For ONE run, keep full per-sample IGs + inputs
            mean_ig_run, std_ig_run, all_attrs = run_single_integrated_gradients(
                model, ds_test, device, logger, return_all=True
            )
            # get variable values
            X_list = []
            for i in range(len(ds_test)):
                X, y = ds_test[i]
                X_list.append(X.detach().cpu().numpy())
            rep_all_inputs = np.stack(X_list, axis=0)  # [N, C, L]
            rep_all_attrs = all_attrs                 # [N, C, L]
        else:
            mean_ig_run, std_ig_run = run_single_integrated_gradients(
                model, ds_test, device, logger, return_all=False
            )

        ig_stats.append(mean_ig_run)  # each [C, L]



    #Accuracy/loss aggregation
    accs  = np.array([entry["test_acc"]  for entry in model_run_stats])  
    losses = np.array([entry["test_loss"] for entry in model_run_stats]) 

    mean_acc = accs.mean()
    std_acc  = accs.std()

    mean_loss = losses.mean()
    std_loss  = losses.std()

    model_stats_summary = {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_loss": mean_loss,
        "std_loss": std_loss
    }

    # IG stats aggregation
    ig_stats = np.stack(ig_stats, axis=0)   # shape (R, C, L) (TODO:KEEP TRACK)

    mean_ig = ig_stats.mean(axis=0)       
    std_ig  = ig_stats.std(axis=0) 

    # Compute correlations if needed
    if compute_correlations and rep_all_attrs is not None:
        #NB:
        # rep_all_inputs: [N, C, L]
        # rep_all_attrs:  [N, C, L]
        # Pearson correlation per (c, l)
        N, C, L = rep_all_inputs.shape
        corrs = np.full((C, L), np.nan)
        for c in range(C):
            for l in range(L):
                #TODO: nb made abs here
                x = np.abs(rep_all_inputs[:, c, l])
                a = np.abs(rep_all_attrs[:, c, l])
                if x.std() > 0 and a.std() > 0:
                    corrs[c, l] = np.corrcoef(x, a)[0, 1]
        corr_results = corrs
        stability = 1.0 - std_ig / (mean_ig + 10**(-8))
    else:
        corr_results = None
    

    return model_stats_summary, mean_ig, std_ig, corr_results



def plot_ig_summary(
    ds_test,
    var_channels,
    mean_ig,
    level_values,
    device,
    corr_results=None,
    figsize=(7, 5),
):

    # Compute mean variable profiles
    mean_vars, std_vars = compute_mean_variable_profile(ds_test, device)

    C, L = mean_ig.shape

    for c in range(C):
        fig, ax1 = plt.subplots(figsize=figsize)

        # IG vs pressure
        ax1.plot(level_values, mean_ig[c, :], "b-", label="Mean IG")
        ax1.set_xlabel("Pressure level")
        ax1.set_ylabel("Integrated Gradients", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.invert_xaxis()  # optional (makes low pressure right side)

        # Variable profile on the right y-axis
        ax2 = ax1.twinx()
        ax2.plot(level_values, mean_vars[c, :], "r--", label="Mean Variable")
        ax2.set_ylabel("Variable magnitude", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # Optional correlation line
        if corr_results is not None:
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 60))
            ax3.plot(level_values, corr_results[c, :], "g-.", label="Correlation")
            ax3.set_ylabel("Correlation (X vs IG)", color="green")
            ax3.set_ylim(-1, 1)
            ax3.tick_params(axis="y", labelcolor="green")

        plt.title(f"IG + Variable + Correlation: {var_channels[c]}")
        plt.tight_layout()
        plt.show()

    # Correlation heatmap (optional)
    if corr_results is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(corr_results, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation (X vs IG)")
        plt.yticks(range(C), var_channels)
        plt.xticks(range(L), [f"{p:.0f}" for p in level_values], rotation=45)
        plt.xlabel("Pressure level")
        plt.title("Correlation Heatmap (Variable vs IG)")
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 500
    ITER = 5

    EPOCHS = 500

    VAR_CHANNELS = ["delta_qv_physics", "delta_t_physics", "plev_channel"]  # add more variables as needed

    LEVEL_DIM = "newlev"
    TIME_DIM = "time"
    SAMPLE_DIM = "sample"
    run_count = 10
    compute_correlations = True

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

    ds_train, ds_validation, ds_test = data_processing_pipeline(model_datasets = model_datasets, var_channels=VAR_CHANNELS)

    model_stats_summary, mean_ig, std_ig, corr_results =run_cnn_ig_pipeline(ds_train, ds_validation, ds_test, VAR_CHANNELS, EPOCHS, device, run_count, compute_correlations=compute_correlations, logger=logger)
    # MOve into piepline above TODO defineitely 
    plev = ds_test.level_coord
    plot_ig_summary(
        ds_test,
        var_channels=VAR_CHANNELS,
        level_values=plev,
        mean_ig=mean_ig,
        corr_results=corr_results,
        device = device
    )