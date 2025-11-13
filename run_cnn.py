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
        logging.FileHandler("CNN.log")    
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
    # Build datasets
    # Careful if this adds to the time bc can dvide by the number of times we repeat the CNN per column
    train_loader = DataLoader(ds_train, batch_size=128, shuffle=True)
    validation_loader  = DataLoader(ds_validation,  batch_size=256, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=256, shuffle=False)


    # Input: [B, C, L] where C=#channels (2 here), L=#levels (46)
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

def run_single_integrated_gradients(model, ds_test, device, logger):
    model.eval()
    X, y = ds_test[0]
    logits = model(X.unsqueeze(0).to(device))

    input_img = X.unsqueeze(0).to(device)

    #  Integrated Gradients 
    integrated_gradients = IntegratedGradients(model)
    baseline = torch.zeros_like(input_img)   # standard zero baseline; change if you prefer

    pred_label_idx = int(torch.argmax(logits, dim=1).item())

    attributions_ig = integrated_gradients.attribute(
        input_img,
        baselines=baseline,
        target=pred_label_idx,
        n_steps=200     # TODO: pick based on computaiton complexoty
    )

    logger.info("input_img shape: %s", tuple(input_img.shape))
    logger.info("attributions_ig shape: %s", tuple(attributions_ig.shape))


    # TODO: come back in case this should be done later and preferthe other format
    reduced_attr = attributions_ig.squeeze(0)
    return reduced_attr



def run_cnn(ds_train, ds_validation, ds_test, var_channels, epochs, device, logger=logging.getLogger(__name__)):

    #TODO Do loop
    model, test_loss, test_acc = run_single_cnn(
        ds_train,
        ds_validation,
        ds_test,
        var_channels,
        epochs, device, logger
    )
    reduced_attr = run_single_integrated_gradients(model, ds_test, device, logger)
    return





    # TODO have below as a separate thing so can iteratively call later on and flexible???

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

    run_cnn(ds_train, ds_validation, ds_test, VAR_CHANNELS, EPOCHS, device, logger)