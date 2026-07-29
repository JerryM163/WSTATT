
import time
import random

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from Utils.device import device
from Utils.data import get_data_loader
from Models.statt import STATT, WSTATT

class_color_list = ['#ffd300','#ff2626','#00a8e2','#ffff00','#e2007c','#a57000','#d6d600','#a50000','#ffcc66','#f2a377','#ff00ff','#704489','#ffff7c',
                '#00a582','#e8d6af','#00ff8c','#ff6666','#334933','#af9970','#ffa5e2','#a5f28c','#ccbfa3','#bfbf77','#93cc93','#93cc93','#93cc93',
                '#e8bfff','#c6d69e','#e8ffbf','#7cafaf','#7cafaf','#4970a3','#9a9a9a',"#9773cd"] ## add unknown to end and subtract one from vamx in imshow()
colormap = mpl.colors.ListedColormap(class_color_list)

def test_model_preds(model, test_dataset, batch_size, timestamps, bands=[]):
    print("########## Testing Model ##########")

    start_time = time.time()

    model = model.to(device)

    # Set model to evaluation mode (disables dropout/BatchNorm)
    model.eval()

    # Test dataset - normally multiple grids
    sample_grids = random.sample(test_dataset, len(test_dataset))

    # Generate cols and rows based on size of test_dataset
    cols = len(test_dataset)//2 # Normally 7
    rows = cols*2               # Normally 2

    plt.figure(figsize=(24, 2*rows)) # VISUALIZING SATELLITE DATA

    for grid_num, grid in enumerate(sample_grids):
        grid_time = time.time()

        print("\x1b[2K" + f"Getting data loader for grid {grid}...", end="\r", flush=True)
        data_loader = get_data_loader(grid, batch_size, bands, timestamps)

        # Initialize list to store predictions for the current grid
        preds = []

        # Process all batches in grid
        for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):
            print("\x1b[2K" + f"Testing on {grid}'s batch {batch + 1}", end="\r", flush=True)

            image_tensor = image_patch.to(device)
            weather_tensor = weather_patch.to(device)
            label_tensor = label_patch.type(torch.long).to(device)

            # Forward pass WITHOUT gradient calculation (saves memory)
            with torch.no_grad():
                if isinstance(model, WSTATT):
                    out = model(image_tensor, weather_tensor)
                else:
                    out = model(image_tensor)
            

            preds.append(out.to(device)) 

        # Gather predictions together to get full image of predicted label
        grid_out = torch.concat(preds, dim=0)

        # Find which class probability is best for each pixel in the prediction
        grid_out = grid_out.argmax(dim=0).cpu()

        # Plot predicted label
        plt.subplot(rows, cols, grid_num + 1)
        plt.axis('off')
        plt.imshow(grid_out, cmap=colormap, interpolation='none', vmin=0, vmax=len(class_color_list)-1)

    # Display final predictions
    plt.tight_layout(pad = 0.1)
    plt.title(f"PREDICTED Labels")
    plt.show()

    print("########## Fetching Ground-Truth Labels ##########")

    plt.figure(figsize=(24, 2*rows)) # VISUALIZING SATELLITE DATA
    
    for grid_num, grid in enumerate(sample_grids):
        plt.subplot(rows, cols, grid_num + 1)
        plt.imshow(grid, cmap=colormap, interpolation='none', vmin=0, vmax=len(class_color_list)-1)

    # Display ground-truth labels
    plt.tight_layout(pad = 0.1)
    plt.title(f"GROUND TRUTH Labels")
    plt.show()