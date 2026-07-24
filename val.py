import random
import time

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

from Utils.device import device
from Utils.data import get_data_loader
from Models.statt import STATT, WSTATT

# Initialize metrics storage
test_loss = []    # Track loss per test run

def validate_epoch(epoch, model, unknown_class, learning_rate, val_dataset, batch_size, timestamps, threshold, class_names, labels_list, bands=[]):
    '''
    Validates a specified model for a single epoch 

    Args:
        epoch - the current training epoch the model is on
        model - either STATT or WSTATT
        unknown_class - specifies which crop label to ignore
        learning_rate - specifies the step size the model takes to correct itself during optimization
        dataset - pre-compiled training dataset of 34 satellite grids
        batch_size - the number of batches processed at a time from the data loader
        timestamps - specifies the equally-spaced points of the year that we are looking at the satellite images from
        threshold - used to determine supported labels
        class_names - list of crop label names
        labels_list - list of numbers corresponding with class_names
        bands - specifies the weather bands taken into account when WSTATT is selected
    Returns:
        epoch_loss - the average loss during validation for this epoch
    '''
    print(f"########## Vaidating EPOCH {epoch+1} ##########")
    label_list = []   # Collect all ground truth labels
    pred_list = []    # Collect all model predictions

    start_time = time.time()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)

    # Set model to evaluation mode (disables dropout/BatchNorm)
    model.eval()

    # Test dataset - normally multiple grids
    sample_grids = random.sample(val_dataset, len(val_dataset))

    epoch_loss = 0  # Accumulate loss across grids
    # Process each grid in test dataset
    for grid_num, grid in enumerate(sample_grids):
        grid_time = time.time()

        print("\x1b[2K" + f"Getting data loader for grid {grid}...", end="\r", flush=True)
        data_loader = get_data_loader(grid, batch_size, bands, timestamps)

        grid_loss = 0  # Accumulate loss for this grid
        # Process all batches in grid
        for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):
            print("\x1b[2K" + f"Testing on {grid}'s batch {batch + 1}", end="\r", flush=True)

            image_tensor = image_patch.to(device)
            weather_tensor = weather_patch.to(device)
            label_tensor = label_patch.type(torch.long).to(device)

            # Forward pass WITHOUT gradient calculation (saves memory)
            with torch.no_grad():
                if isinstance(model, STATT):
                    out = model(image_tensor)
                else:
                    out = model(image_tensor, weather_tensor)

            # Convert model outputs to probabilities using softmax
            # dim=1 applies softmax across classes (channel dimension)
            patch_prob_out = torch.nn.functional.softmax(out, dim=1)

            # Detach from computation graph and move to CPU
            patch_prob_out_numpy = patch_prob_out.cpu().detach().numpy()

            # Get predicted class (index with highest probability)
            # Shape: [batch, height, width]
            pred_patch = np.argmax(patch_prob_out_numpy, axis=1)

            # Calculate loss
            batch_loss = criterion(out, label_tensor)

            grid_loss += batch_loss.item()  # Accumulate batch loss

            # Flatten predictions and labels to 1D arrays
            pred_patch_flat = np.reshape(pred_patch, (-1))      # [batch*height*width]
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
        print("\x1b[2K" + f'Grid Num: {grid_num+1:02}, Grid: {grid}, Loss: {grid_loss:.4f}, Time: {(time.time() - grid_time):.2f}')
        epoch_loss += grid_loss

    # Convert collected results to numpy arrays
    label_array = np.array(label_list)  # All ground truth labels
    pred_array = np.array(pred_list)    # All model predictions

    # Calculate overall test loss
    epoch_loss = epoch_loss / (grid_num + 1)

    print(f'\tValidation Loss:{epoch_loss:.4f}')

    # Compute support (i.e., the number of occurrences per class in label_array)
    unique_labels, support = np.unique(label_array, return_counts=True)

    # Filter labels with support above 50,000
    valid_labels = unique_labels[support > threshold]

    # Create a filtered class name list
    filtered_class_names = [class_names[i] for i in range(len(class_names)) if labels_list[i] in valid_labels]

    # Compute accuracy score for selected labels
    print(f"## Accuracy Score for EPOCH {epoch+1} ##")
    print(f"\t{accuracy_score(label_array, pred_array):.4f}")

    # Compute classification report only for selected labels
    print(f"## Classification Report for EPOCH {epoch+1} ##")
    print(classification_report(label_array, pred_array, target_names=filtered_class_names, digits=4, labels=valid_labels))

    return epoch_loss