import random  
import time
from re import split
from pathlib import Path
from genericpath import isfile

import torch
import numpy as np

from Utils.device import device
from Utils.data import get_data_loader
from Models.statt import STATT, WSTATT

def train_epoch(epoch, model, unknown_class, learning_rate, dataset, batch_size, timestamps, bands=[]):
    '''
    Trains a specificed model for a single epoch

    Args:
        epoch - the current training epoch the model is on
        model - either STATT or WSTATT
        unknown_class - specifies which crop label to ignore
        learning_rate - specifies the step size the model takes to correct itself during optimization
        dataset - pre-compiled training dataset of 34 satellite grids
        batch_size - the number of batches processed at a time from the data loader
        timestamps - specifies the equally-spaced points of the year that we are looking at the satellite images from
        bands - specifies the weather bands taken into account when WSTATT is selected
    Returns:
        epoch_loss - the average loss during training for this epoch
    '''
    print(f"########## Training EPOCH {epoch} ##########")
    start_time = time.time()

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    epoch_loss = 0

    sample_grids = random.sample(dataset, len(dataset))

    for grid_num, grid in enumerate(sample_grids):
        grid_time = time.time()

        print("\x1b[2K" + f"Getting data loader for grid {grid}...", end="\r", flush=True)
        data_loader = get_data_loader(grid, batch_size, bands, timestamps)

        grid_loss = 0

        for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):
            print("\x1b[2K" + f"Training on {grid}'s batch {batch + 1}", end="\r", flush=True)
            optim.zero_grad()

            image_tensor = image_patch.to(device, non_blocking=True)
            weather_tensor = weather_patch.to(device, non_blocking=True)
            label_patch = label_patch.type(torch.long).to(device, non_blocking=True)

            if isinstance(model, STATT):
                out = model(image_tensor)
            else:
                out = model(image_tensor, weather_tensor)

            batch_loss = criterion(out, label_patch)

            batch_loss.backward()

            optim.step()

            grid_loss += batch_loss.item()

        grid_loss = grid_loss / (batch + 1) 
        print("\x1b[2K" + f'Grid Num: {grid_num+1:02}, Grid: {grid}, Loss: {grid_loss:.4f}, Time: {(time.time() - grid_time):.2f}')

        epoch_loss += grid_loss

    epoch_loss = epoch_loss / (grid_num + 1)
    print(f'\tTest Loss: {epoch_loss:.4f}, Time: {(time.time() - start_time):.2f}')

    return epoch_loss