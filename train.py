from genericpath import isfile
import os
import sys

# Drop the cluster's default library injection tracking completely
os.environ.pop("LD_LIBRARY_PATH", None)

# Force the environment link loader to stick strictly to the conda environment
conda_lib = "/users/0/hinsv006/miniconda3/envs/jerry/lib"
os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:/lib64"
sys.path.insert(0, conda_lib)

import random  
import time
from re import split
from pathlib import Path
import torch
import numpy as np
from data import get_data_loader

# --- MODEL IMPORT ---
from Models.statt import STATT, WSTATT
from Models.mctnet import CT_NET

torch.backends.cudnn.enabled = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Active Device Status:", "cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    in_channels = 10
    in_channels_weather = 1
    out_channels = 33

    bands = 10
    classes = 33

    unknown_class = 100
    learning_rate = 0.0001
    num_epochs = 10

    input_patch_size = 32
    output_patch_size = 32
    batch_size = 16

    NUM_SAMPLES = 32

    timestamps = 12

    # Load the provided datasets.
    train_dataset = np.load(r"../WSTATT_DATA/DISTRIBUTION/T11SKA/train_set_T11SKA_DISTRI1.npy").tolist()
    val_dataset = np.load(r"../WSTATT_DATA/DISTRIBUTION/T11SKA/validation_set_T11SKA_DISTRI1.npy").tolist()
    test_dataset = np.load(r"../WSTATT_DATA/DISTRIBUTION/T11SKA/test_set_T11SKA_DISTRI1.npy").tolist()

    print("########## BUILDING MODELS ##########")
    # --- STATT Baseline Models ---
    model = STATT(
        in_channels=in_channels,
        out_channels=out_channels
    )
    if os.path.isfile("Statt.pt"):
        model.load_state_dict(torch.load("Statt.pt"),strict = False)
        print("STATT Model Loaded")
    else:
        print(f"STATT Model Complete")
    #wstatt = CNN_WSTATT(
    #    in_channels=in_channels,
    #    in_channels_w=in_channels_weather,
    #    out_channels=out_channels
    #)
    #if os.path.isfile("Wstatt.pt"):
    #    wstatt.load_state_dict(torch.load("Wstatt.pt"),strict = False)
    #    print("WSTATT Model Loaded")
    #else:
    #    print(f"WSTATT Model Complete")

    # --- CT Net Models ----
    #model = CT_NET(
    #    bands=bands,
    #    classes=classes,
    #)
    #if os.path.isfile("CTNet.pt"):
    #    model.load_state_dict(torch.load("CTNet.pt"),strict = False)
    #    print("CT Net Model Loaded")
    #else:
    #    print(f"CT Net Model Complete")

    print("########## TRAINING MODELS ##########")
    start_time = time.time()

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []

    model.train()

    epoch_loss = 0

    sample_grids = random.sample(train_dataset, NUM_SAMPLES)

    for grid_num, grid in enumerate(sample_grids):
        grid_time = time.time()

        print("\x1b[2K" + f"Getting data loader for grid {grid}...", end="\r", flush=True)
        data_loader = get_data_loader(grid, batch_size, timestamps)

        grid_loss = 0

        for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):
            print("\x1b[2K" + f"Training on {grid}'s batch {batch + 1}", end="\r", flush=True)
            optim.zero_grad()

            image_tensor = image_patch.to(device, non_blocking=True)
            weather_tensor = weather_patch.to(device, non_blocking=True)
            label_patch = label_patch.type(torch.long).to(device, non_blocking=True)

            out = model(image_tensor)

            batch_loss = criterion(out, label_patch)

            batch_loss.backward()

            optim.step()

            grid_loss += batch_loss.item()

        grid_loss = grid_loss / (batch + 1) 
        print("\x1b[2K" + f'Grid Num: {grid_num + 1}, Grid: {grid}, Loss: {grid_loss:.4f}, Time: {(time.time() - grid_time):.2f}')

        epoch_loss += grid_loss

    epoch_loss = epoch_loss / (grid_num + 1)
    print(f'\tTest Loss: {epoch_loss:.4f}, Time: {(time.time() - start_time):.2f}')

    train_loss.append(epoch_loss)

    torch.save(model.state_dict(), "Statt.pt")
    #torch.save(model.state_dict(), "Wstatt.pt")
    #torch.save(model.state_dict(), "CTNet.pt")

    '''
    print("########## TEST MODELS ##########")
    model = CT_NET(
        bands=bands,
        classes=classes,
    )

    model = model.to(device)

    print("LOAD MODEL")
    model.load_state_dict(torch.load("CTNet.pt"),strict = False)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)

    print("#######################################################################")

    threshold = 50000
    labels_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    class_names = ['Corn','Cotton','Rice','Sunflower','Barley','Winter_Wheat','Safflower','Dry Beans','Onions','Tomatoes',
                   'Cherries','Grapes','Citrus','Almonds','Walnut','Pistachio','Garlic','Olives','Pomegranates','Alfalfa',
                   'Hay','Barren_land','Fallow_and_Idle','Deciduous_Forests','Evergreen_forest','Mixed_Forests',
                   'Clover_and_wildflower','Shrubland','Grass','Woody_wetlands','Herbaceous_Wetlands','Water','Urban']

    # Initialize metrics storage
    test_loss = []    # Track loss per test run
    label_list = []   # Collect all ground truth labels
    pred_list = []    # Collect all model predictions

    # Set model to evaluation mode (disables dropout/BatchNorm)
    model.eval()

    # Test dataset - normally multiple grids
    sample_grids = random.sample(test_dataset, NUM_SAMPLES)

    epoch_loss = 0  # Accumulate loss across grids
    # Process each grid in test dataset
    for grid_num, grid in enumerate(sample_grids):
        print("\x1b[2K" + f"Getting data loader for grid {grid}...", end="\r", flush=True)
        data_loader = get_data_loader(grid, batch_size)

        grid_loss = 0  # Accumulate loss for this grid
        # Process all batches in grid
        for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):
            print("\x1b[2K" + f"Testing on {grid}'s batch {batch + 1}", end="\r", flush=True)

            # Forward pass WITHOUT gradient calculation (saves memory)
            image_tensor = image_patch.to(device, non_blocking=True)
            weather_tensor = weather_patch.to(device, non_blocking=True)

            with torch.no_grad():
                patch_out = model(image_tensor)

            # Convert model outputs to probabilities using softmax
            # dim=1 applies softmax across classes (channel dimension)
            patch_prob_out = torch.nn.functional.softmax(patch_out, dim=1)

            # Detach from computation graph and move to CPU
            patch_prob_out_numpy = patch_prob_out.cpu().detach().numpy()

            # Get predicted class (index with highest probability)
            # Shape: [batch, height, width]
            pred_patch = np.argmax(patch_prob_out_numpy, axis=1)

            # Prepare labels for loss calculation
            label_patch_device = label_patch.type(torch.long).to(device)

            # Calculate loss
            batch_loss = criterion(patch_out, label_patch_device)

            grid_loss += batch_loss.item()  # Accumulate batch loss

            # Flatten predictions and labels to 1D arrays
            pred_patch_flat = np.reshape(pred_patch, (-1))        # [batch*height*width]
            label_patch_flat = np.reshape(label_patch, (-1))      # [batch*height*width]

            # Filter out unknown_class pixels (ignore_index)
            valid_mask = label_patch_flat != unknown_class
            pred_grid_flat = pred_patch_flat[valid_mask]
            label_grid_flat = label_patch_flat[valid_mask]

            # Collect valid predictions and labels for overall metrics
            for l in range(pred_grid_flat.shape[0]):
                label_list.append(label_grid_flat[l])
                pred_list.append(pred_grid_flat[l])

        # Calculate average loss for current grid
        grid_loss = grid_loss / (batch + 1)
        print("\x1b[2K" + f'Grid Num: {grid_num:02}, Grid: {grid}, Loss: {grid_loss:.4f}')
        epoch_loss += grid_loss

    # Convert collected results to numpy arrays
    label_array = np.array(label_list)  # All ground truth labels
    pred_array = np.array(pred_list)    # All model predictions

    # Calculate overall test loss
    epoch_loss = epoch_loss / (grid_num + 1)

    print(f'\tTest Loss:{epoch_loss:.4f}')

    test_loss.append(epoch_loss)  # Store for later analysis

    # print('Overall unknown:', np.sum(pred_array == 100), '  labels:', np.sum(label_array == 100))
    # print(classification_report(label_array, pred_array, target_names=class_names, digits=4,labels = labels_list))


    # Compute support (i.e., the number of occurrences per class in label_array)
    unique_labels, support = np.unique(label_array, return_counts=True)

    # Filter labels with support above 50,000
    valid_labels = unique_labels[support > threshold]

    # Create a filtered class name list
    filtered_class_names = [class_names[i] for i in range(len(class_names)) if labels_list[i] in valid_labels]

    # Compute classification report only for selected labels
    print("## Classification Report ##")
    print(classification_report(label_array, pred_array, target_names=filtered_class_names, digits=4, labels=valid_labels))
    '''