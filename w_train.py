from genericpath import isfile
import os
import sys

# Drop the cluster's default library injection tracking completely
os.environ.pop("LD_LIBRARY_PATH", None)

# Force the environment link loader to stick strictly to the conda environment
conda_lib = "/users/0/hinsv006/miniconda3/envs/carson/lib"
os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:/lib64"
sys.path.insert(0, conda_lib)

import random  
import time
from re import split
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from data import get_data_loader, w_get_data_loader

# Import WSTATT model
from Models.statt import WSTATT

torch.backends.cudnn.enabled = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Active Device Status:", "cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    in_channels = 10
    out_channels = 33

    unknown_class = 100
    learning_rate = 0.0001
    num_epochs = 10

    input_patch_size = 32
    output_patch_size = 32
    batch_size = 16

    NUM_SAMPLES = 32

    weather_bands = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
    bands = [0, 1, 2, 3, 4, 5, 6]
    in_channels_weather = len(bands)

    timestamps = 6

    # Load the provided datasets.
    train_dataset = np.load(r"../WSTATT_DATA/DISTRIBUTION/T11SKA/train_set_T11SKA_DISTRI1.npy").tolist()
    val_dataset = np.load(r"../WSTATT_DATA/DISTRIBUTION/T11SKA/validation_set_T11SKA_DISTRI1.npy").tolist()
    test_dataset = np.load(r"../WSTATT_DATA/DISTRIBUTION/T11SKA/test_set_T11SKA_DISTRI1.npy").tolist()

    print("########## BUILDING MODEL ##########")
    wstatt = WSTATT(
        in_channels=in_channels,
        in_channels_w=in_channels_weather,
        out_channels=out_channels
    )
    if os.path.isfile("Wstatt.pt"):
        wstatt.load_state_dict(torch.load("Wstatt.pt"),strict = False)
        print("WSTATT Model Loaded")
    else:
        print(f"WSTATT Model Complete")

    print("########## TRAINING MODEL ##########")
    start_time = time.time()

    wstatt = wstatt.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)

    optim = torch.optim.Adam(wstatt.parameters(), lr=learning_rate)

    train_loss = []

    wstatt.train()

    epoch_loss = 0

    sample_grids = random.sample(train_dataset, NUM_SAMPLES)

    for grid_num, grid in enumerate(sample_grids):
        grid_time = time.time()

        print("\x1b[2K" + f"Getting data loader for grid {grid}...", end="\r", flush=True)
        data_loader = w_get_data_loader(grid, batch_size, bands, timestamps)

        grid_loss = 0

        for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):
            print("\x1b[2K" + f"Training on {grid}'s batch {batch + 1}", end="\r", flush=True)
            optim.zero_grad()

            image_tensor = image_patch.to(device, non_blocking=True)
            weather_tensor = weather_patch.to(device, non_blocking=True)
            label_patch = label_patch.type(torch.long).to(device, non_blocking=True)

            out = wstatt(image_tensor, weather_tensor)

            batch_loss = criterion(out, label_patch)

            batch_loss.backward()

            optim.step()

            grid_loss += batch_loss.item()

        grid_loss = grid_loss / (batch + 1)
        print("\x1b[2K" + f'Grid Num: {grid_num + 1}, Grid: {grid}, Loss: {grid_loss:.4f}, Time: {(time.time() - grid_time):.2f}')

        epoch_loss += grid_loss

    epoch_loss = epoch_loss / (grid_num + 1)
    print(f'\tBand(s): {bands}, Test Loss: {epoch_loss:.4f}, Epoch Time: {(time.time() - start_time):.2f},')

    train_loss.append(epoch_loss)

    save = input("Save this model (y/n)?: ")
    if str(save).lower() == "y":
        torch.save(wstatt.state_dict(), "Wstatt.pt")