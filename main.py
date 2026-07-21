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
from re import split
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from Models.cnn import CNN_STATT, CNN_WSTATT
from data import get_data_loader

torch.backends.cudnn.enabled = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Active Device Status:", "cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    in_channels = 10
    in_channels_weather = 7
    out_channels = 33

    unknown_class = 100
    learning_rate = 0.0001
    num_epochs = 10

    input_patch_size = 32
    output_patch_size = 32
    batch_size = 16

    NUM_SAMPLES = 32

    # List of all possible grid names in the google drive folder, based on their naming conventions
    dataset = [
        f"T11SKA_{year}_{first_digit}_{second_digit}"
        for year in (2018, 2019, 2020)
        for first_digit in range(10)
        for second_digit in range(10)
    ]

    shuffled = random.sample(dataset, len(dataset))
    split_idx = int(len(shuffled) * 0.6)
    
    train_dataset = shuffled[:split_idx]

    split_dataset = shuffled[split_idx:]
    split_idx = int(len(split_dataset) * 0.5)

    val_dataset = split_dataset[:split_idx]
    test_dataset = split_dataset[split_idx:]

    print("########## BUILDING MODELS ##########")
    statt = STATT(
        in_channels=in_channels,
        out_channels=out_channels
    )
    if os.path.isfile("Statt.pt"):
        statt.load_state_dict(torch.load("Statt.pt"),strict = False)
        print("STATT Model Loaded")
    else:
        print(f"STATT Model Complete")

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

    print("########## TRAINING MODELS ##########")
    statt = statt.to(device)
    wstatt = wstatt.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)

    statt_optim = torch.optim.Adam(statt.parameters(), lr=learning_rate)
    wstatt_optim = torch.optim.Adam(wstatt.parameters(), lr=learning_rate)

    statt_train_loss = []
    wstatt_train_loss = []

    statt.train()
    wstatt.train()

    statt_epoch_loss = 0
    wstatt_epoch_loss = 0

    sample_grids = random.sample(train_dataset, NUM_SAMPLES)

    for grid_num, grid in enumerate(sample_grids):

        print("\x1b[2K" + f"Getting data loader for grid {grid}...", end="\r", flush=True)
        data_loader = get_data_loader(grid, batch_size)

        statt_grid_loss = 0
        wstatt_grid_loss = 0

        for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):
            print("\x1b[2K" + f"Training on {grid}'s batch {batch + 1}", end="\r", flush=True)
            statt_optim.zero_grad()
            wstatt_optim.zero_grad()

            image_tensor = image_patch.to(device, non_blocking=True)
            weather_tensor = weather_patch.to(device, non_blocking=True)
            label_patch = label_patch.type(torch.long).to(device, non_blocking=True)

            statt_out = statt(image_tensor)
            wstatt_out = wstatt(image_tensor, weather_tensor)

            statt_batch_loss = criterion(statt_out, label_patch)
            wstatt_batch_loss = criterion(wstatt_out, label_patch)

            statt_batch_loss.backward()
            wstatt_batch_loss.backward()

            statt_optim.step()
            wstatt_optim.step()

            statt_grid_loss += statt_batch_loss.item()
            wstatt_grid_loss += wstatt_batch_loss.item()

        statt_grid_loss = statt_grid_loss / (batch + 1) 
        wstatt_grid_loss = wstatt_grid_loss / (batch + 1)
        print("\x1b[2K" + f'Grid Num: {grid_num + 1} Grid: {grid} STATT Loss: {statt_grid_loss:.4f} WSTATT: {wstatt_grid_loss:.4f}')

        statt_epoch_loss += statt_grid_loss
        wstatt_epoch_loss += wstatt_grid_loss

    statt_epoch_loss = statt_epoch_loss / (grid_num + 1)
    wstatt_epoch_loss = wstatt_epoch_loss / (grid_num + 1)
    print(f'\tSTATT Test Loss: {statt_epoch_loss:.4f} WSTATT Test Loss: {wstatt_epoch_loss:.4f}')

    statt_train_loss.append(statt_epoch_loss)
    wstatt_train_loss.append(wstatt_epoch_loss)

    torch.save(statt.state_dict(), "Statt.pt")
    torch.save(wstatt.state_dict(), "Wstatt.pt")

'''
print("########## TEST MODELS ##########")
statt = STATT(
    in_channels=in_channels,
    out_channels=out_channels
)

wstatt = WSTATT(
    in_channels=in_channels,
    in_channels_w=in_channels_weather,
    out_channels=out_channels
)

statt = statt.to(device)
wstatt = wstatt.to(device)

print("LOAD MODEL")
statt.load_state_dict(torch.load("Statt.pt"),strict = False)
wstatt.load_state_dict(torch.load("Wstatt.pt"),strict = False)

criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)

print("#######################################################################")

threshold = 50000
labels_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
class_names = ['Corn','Cotton','Rice','Sunflower','Barley','Winter_Wheat','Safflower','Dry Beans','Onions','Tomatoes',
               'Cherries','Grapes','Citrus','Almonds','Walnut','Pistachio','Garlic','Olives','Pomegranates','Alfalfa',
               'Hay','Barren_land','Fallow_and_Idle','Deciduous_Forests','Evergreen_forest','Mixed_Forests',
               'Clover_and_wildflower','Shrubland','Grass','Woody_wetlands','Herbaceous_Wetlands','Water','Urban']

# Initialize metrics storage
statt_test_loss = []    # Track loss per test run
wstatt_test_loss = []
label_list = []   # Collect all ground truth labels
statt_pred_list = []    # Collect all model predictions
wstatt_pred_list = []

# Set model to evaluation mode (disables dropout/BatchNorm)
statt.eval()
wstatt.eval()

# Test dataset - normally multiple grids
sample_grids = random.sample(test_dataset, NUM_SAMPLES)

statt_epoch_loss = 0  # Accumulate loss across grids
wstatt_epoch_loss = 0
# Process each grid in test dataset
for grid_num, grid in enumerate(sample_grids):
    print("\x1b[2K" + f"Getting data loader for grid {grid}...", end="\r", flush=True)
    data_loader = get_data_loader(grid, batch_size)

    statt_grid_loss = 0  # Accumulate loss for this grid
    wstatt_grid_loss = 0
    # Process all batches in grid
    for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):
        print("\x1b[2K" + f"Testing on {grid}'s batch {batch + 1}", end="\r", flush=True)
        # Forward pass WITHOUT gradient calculation (saves memory)
        image_tensor = image_patch.to(device, non_blocking=True)
        weather_tensor = weather_patch.to(device, non_blocking=True)

        with torch.no_grad():
            statt_patch_out = statt(image_tensor)
            wstatt_patch_out = wstatt(image_tensor, weather_tensor)

        # Convert model outputs to probabilities using softmax
        # dim=1 applies softmax across classes (channel dimension)
        statt_patch_prob_out = torch.nn.functional.softmax(statt_patch_out, dim=1)
        wstatt_patch_prob_out = torch.nn.functional.softmax(wstatt_patch_out, dim=1)

        # Detach from computation graph and move to CPU
        statt_patch_prob_out_numpy = statt_patch_prob_out.cpu().detach().numpy()
        wstatt_patch_prob_out_numpy = wstatt_patch_prob_out.cpu().detach().numpy()

        # Get predicted class (index with highest probability)
        # Shape: [batch, height, width]
        statt_pred_patch = np.argmax(statt_patch_prob_out_numpy, axis=1)
        wstatt_pred_patch = np.argmax(wstatt_patch_prob_out_numpy, axis=1)

        # Prepare labels for loss calculation
        label_patch_device = label_patch.type(torch.long).to(device)

        # Calculate loss
        statt_batch_loss = criterion(statt_patch_out, label_patch_device)
        wstatt_batch_loss = criterion(wstatt_patch_out, label_patch_device)

        statt_grid_loss += statt_batch_loss.item()  # Accumulate batch loss
        wstatt_grid_loss += wstatt_batch_loss.item()

        # Flatten predictions and labels to 1D arrays
        statt_pred_patch_flat = np.reshape(statt_pred_patch, (-1))        # [batch*height*width]
        wstatt_pred_patch_flat = np.reshape(wstatt_pred_patch, (-1))      # [batch*height*width]
        label_patch_flat = np.reshape(label_patch, (-1))      # [batch*height*width]

        # Filter out unknown_class pixels (ignore_index)
        valid_mask = label_patch_flat != unknown_class
        statt_pred_grid_flat = statt_pred_patch_flat[valid_mask]
        wstatt_pred_grid_flat = wstatt_pred_patch_flat[valid_mask]
        label_grid_flat = label_patch_flat[valid_mask]

        # Collect valid predictions and labels for overall metrics
        for l in range(statt_pred_grid_flat.shape[0]):
            label_list.append(label_grid_flat[l])
            statt_pred_list.append(statt_pred_grid_flat[l])
            wstatt_pred_list.append(wstatt_pred_grid_flat[l])

    # Calculate average loss for current grid
    statt_grid_loss = statt_grid_loss / (batch + 1)
    wstatt_grid_loss = wstatt_grid_loss / (batch + 1)
    print("\x1b[2K" + f'Grid Num: {grid_num:02} Grid: {grid} STATT Loss: {statt_grid_loss:.4f} WSTATT Loss: {wstatt_grid_loss:.4f}')
    statt_epoch_loss += statt_grid_loss
    wstatt_epoch_loss += wstatt_grid_loss

# Convert collected results to numpy arrays
label_array = np.array(label_list)  # All ground truth labels
statt_pred_array = np.array(statt_pred_list)    # All model predictions
wstatt_pred_array = np.array(wstatt_pred_list)

# Calculate overall test loss
statt_epoch_loss = statt_epoch_loss / (grid_num + 1)
wstatt_epoch_loss = wstatt_epoch_loss / (grid_num + 1)

print(f'\tSTATT Test Loss:{statt_epoch_loss:.4f} WSTATT Test Loss:{wstatt_epoch_loss:.4f}')

statt_test_loss.append(statt_epoch_loss)  # Store for later analysis
wstatt_test_loss.append(wstatt_epoch_loss)

# print('Overall unknown:', np.sum(pred_array == 100), '  labels:', np.sum(label_array == 100))
# print(classification_report(label_array, pred_array, target_names=class_names, digits=4,labels = labels_list))


# Compute support (i.e., the number of occurrences per class in label_array)
unique_labels, support = np.unique(label_array, return_counts=True)

# Filter labels with support above 50,000
valid_labels = unique_labels[support > threshold]

# Create a filtered class name list
filtered_class_names = [class_names[i] for i in range(len(class_names)) if labels_list[i] in valid_labels]

# Compute classification report only for selected labels
print("## STATT Classification Report ##")
print(classification_report(label_array, statt_pred_array, target_names=filtered_class_names, digits=4, labels=valid_labels))

print("## WSTATT Classification Report ##")
print(classification_report(label_array, wstatt_pred_array, target_names=filtered_class_names, digits=4, labels=valid_labels))
'''