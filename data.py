import os
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import matplotlib.colors
from pathlib import Path

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


# To set your own directory path, find the WSTATT_DATA folder in your Google Drive and copy it 'as path'
# The long string I used as the parameter to Path() is simply my own unique path to WSTATT_DATA
data_dir = Path(r"../WSTATT_DATA")

combined_label_data_dir = Path(r"../WSTATT_DATA/LABEL_DATA/NUMPY/COMBINED_LABELS")
eroded_label_data_dir = Path(r"../WSTATT_DATA/LABEL_DATA/NUMPY/ERODED_LABELS")
sat_data_dir = Path(r"../WSTATT_DATA/SATELLITE/NUMPY")
weather_data_dir = Path(r"../WSTATT_DATA/WEATHER/DAYMET")

input_patch_size = 32
output_patch_size = 32

# List of all possible grid names in the google drive folder, based on their naming conventions
dataset = [
    f"T11SKA_{year}_{first_digit}_{second_digit}"
    for year in (2018, 2019, 2020)
    for first_digit in range(10)
    for second_digit in range(10)
]

def create_weather_satellite_patches(grid):
    '''
    Args:
        grid - A single WSTATT data sample (eg. T11SKA_2018_0_0) as a string
    Returns:
        (image_patches, weather_patches, label_patches) - Numpy arrays of image, weather, and label patches
    '''
    print(f'\rCreating patches for grid: {grid}', end="")

    image = np.load(os.path.join(sat_data_dir, grid + "_image.npy"))
    label = np.load(os.path.join(eroded_label_data_dir, grid + "_label.npy"))
    weather = np.load(os.path.join(weather_data_dir, grid + "_daymet_10980_global_normalised_year_day_average_grid_array.npy"))

    # Replace NaN values in the weather data to prevent future issues
    weather[np.isnan(weather)] = 0

    height, width = label.shape

    # Calculate padding to ensure that the image patches are centered around the label patches
    padding = (input_patch_size - output_patch_size) // 2

    image_patches = []
    label_patches = []
    weather_patches = []

    # Slide a window over the label array to extract patches, and extract corresponding image and weather patches
    for i in range(height // output_patch_size):    # Vertical steps
        for j in range(width // output_patch_size): # Horizontal steps
            i_label_start = i * output_patch_size
            i_label_end = (i + 1) * output_patch_size
            j_label_start = j * output_patch_size
            j_label_end = (j + 1) * output_patch_size

            i_image_start = i_label_start - padding
            i_image_end = i_label_end + padding
            j_image_start = j_label_start - padding
            j_image_end = j_label_end + padding

            # Ensure that the calculated indices for the image patches are within the bounds of the image dimensions
            if (0 <= i_image_start < height and
                0 <= i_image_end <= height and
                0 <= j_image_start < width and
                0 <= j_image_end <= width):

                # Finally extract the image and label patches using the calculated indices 
                image_patch = image[:, :, i_image_start:i_image_end, j_image_start:j_image_end]
                label_patch = label[i_label_start:i_label_end, j_label_start:j_label_end]

                # Append the extracted patches to their respective lists
                image_patches.append(image_patch)
                label_patches.append(label_patch)
                weather_patches.append(weather)

    # Convert the lists of patches into numpy arrays for efficiency and ensure they have the correct data types for further processing
    image_patches = np.array(image_patches).astype(np.float32)     # Shape: (num_samples, timesteps, channels, height, width)
    weather_patches = np.array(weather_patches).astype(np.float32) # Shape: (num_samples, timesteps, weather_features)
    label_patches = np.array(label_patches).astype(np.int8)        # Shape: (num_samples, height, width)

    return image_patches, weather_patches, label_patches

class SEGMENTATION(Dataset):
    """Custom PyTorch Dataset for satellite image patches with weather data.

    Combines:
    - Satellite image patches (spatio-temporal data)
    - Weather data (temporal features)
    - Segmentation labels (ground truth)
    """
    def __init__(self, image_patches, weather_patches, label_patches):
        """Initialize dataset with preprocessed patches.

        Args:
            image_patches: Array of satellite image patches
                shape: (num_samples, timesteps, channels, height, width)
            weather_patches: Array of weather data sequences
                shape: (num_samples, timesteps, weather_features)
            label_patches: Array of segmentation labels
                shape: (num_samples, height, width)
        """
        self.image_patches = image_patches
        self.weather_patches = weather_patches
        self.label_patches = label_patches

    def __len__(self):
        ''' Returns the total number of samples in the dataset '''
        return len(self.label_patches)

    def __getitem__(self, idx):
        ''' Returns a specific indexed sample from the dataset including its image, weather, and label patch'''
        return(
            self.image_patches[idx],
            self.weather_patches[idx],
            self.label_patches[idx]
        )

def get_random_sample():
    '''
    Returns:
        A randomized sample of the grid names in the dataset (eg. T11SKA_2018_0_0)
    '''
    return random.sample(dataset, len(dataset))

def get_data_loader(grid, batch_size):
    '''
    Args:
        grid - A single WSTATT data sample (eg. T11SKA_2018_0_0) as a string
        batch_size - Number of samples per batch; needed to determine how many batches are needed for each data_loader
    Returns:
        data_loader - A data_loader object containing shuffled batches of image, weather, and label patches
    '''

    image_patches, weather_patches, label_patches = create_weather_satellite_patches(grid)

    data = SEGMENTATION(image_patches, weather_patches, label_patches)

    data_loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    return data_loader