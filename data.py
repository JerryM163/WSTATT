import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.colors
from pathlib import Path

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from Models.cnn import CNN_STATT, CNN_WSTATT

# To set your own directory path, find the WSTATT_DATA folder in your Google Drive and copy it 'as path'
# The long string I used as the parameter to Path() is simply my own unique path to WSTATT_DATA
data_dir = Path(r"../WSTATT_DATA")

combined_label_data_dir = Path(r"../WSTATT_DATA/LABEL_DATA/NUMPY/COMBINED_LABELS")
eroded_label_data_dir = Path(r"../WSTATT_DATA/LABEL_DATA/NUMPY/ERODED_LABELS")
sat_data_dir = Path(r"../WSTATT_DATA/SATELLITE/NUMPY")
weather_data_dir = Path(r"../WSTATT_DATA/WEATHER/DAYMET")

#combined_label_data_dir = Path(r"G:\.shortcut-targets-by-id\1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX\WSTATT_DATA\LABEL_DATA\NUMPY\COMBINED_LABELS")
#eroded_label_data_dir = Path(r"G:\.shortcut-targets-by-id\1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX\WSTATT_DATA\LABEL_DATA\NUMPY\ERODED_LABELS")
#sat_data_dir = Path(r"G:\.shortcut-targets-by-id\1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX\WSTATT_DATA\SATELLITE\NUMPY")
#weather_data_dir = Path(r"G:\.shortcut-targets-by-id\1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX\WSTATT_DATA\WEATHER\DAYMET")

input_patch_size = 32
output_patch_size = 32

GRID_CACHE = {}

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

def create_patches(grid):
    """Create image patches, weather data, and label patches for a given grid location.

    Args:
        grid: Identifier for the geographical area to process

    Returns:
        Tuple of (image_patches, weather_patches, label_patches) as numpy arrays
    """

    # Load satellite image data (4D array: [timesteps, channels, height, width])
    image = np.load(os.path.join(sat_data_dir, grid + "_image.npy"))

    # Load label data (2D array: [height, width] - ground truth for each pixel)
    label = np.load(os.path.join(combined_label_data_dir, grid + "_combined_label.npy"))

    # Load weather data (3D array: [timesteps, weather_features])
    weather = np.load(os.path.join(weather_data_dir, grid + '_daymet_10980_global_normalised_year_day_average_grid_array.npy'))

    # Handle missing values in weather data by replacing NaNs with 0
    weather[np.isnan(weather)] = 0

    # Get dimensions of the label data (same as full image size)
    height, width = label.shape

    # Calculate padding difference needed between input and output patches
    # This ensures the input patch is larger than the output label patch
    diff = (input_patch_size - output_patch_size) // 2

    # Initialize lists to store patches
    image_patches = []   # Will store satellite image patches
    weather_patches = []  # Will store weather data sequences
    label_patches = []    # Will store label patches

    # Slide a window over the image to create patches
    # Step size is output_patch_size (non-overlapping output patches)
    for i in range(height // output_patch_size):        # Vertical steps
        for j in range(width // output_patch_size):     # Horizontal steps
            # Calculate label patch boundaries (central region)
            i_label_start = i * output_patch_size
            i_label_end = (i + 1) * output_patch_size
            j_label_start = j * output_patch_size
            j_label_end = (j + 1) * output_patch_size

            # Calculate larger image patch boundaries (with padding)
            i_image_start = i_label_start - diff
            i_image_end = i_label_end + diff
            j_image_start = j_label_start - diff
            j_image_end = j_label_end + diff

            # Check if the image patch is within the original image boundaries
            if (0 <= i_image_start < height and
                0 <= i_image_end <= height and
                0 <= j_image_start < width and
                0 <= j_image_end <= width):

                # Extract satellite image patch (all timesteps and channels)
                # Shape: [timesteps, channels, patch_height, patch_width]
                image_patch = image[:, :, i_image_start:i_image_end, j_image_start:j_image_end]

                # Extract corresponding label patch (ground truth)
                # Shape: [patch_height, patch_width]
                label_patch = label[i_label_start:i_label_end, j_label_start:j_label_end]

                # Store extracted patches
                image_patches.append(image_patch)
                label_patches.append(label_patch)

                # Associate the entire weather sequence with this spatial patch
                # Same weather data used for all patches from this grid
                weather_patches.append(weather)

    # Convert lists to numpy arrays for efficient processing
    image_patches = np.array(image_patches).astype(np.float32)     # Shape: [num_patches, timesteps, channels, H, W]
    weather_patches = np.array(weather_patches).astype(np.float32) # Shape: [num_patches, timesteps, weather_features]
    label_patches = np.array(label_patches).astype(np.int8)        # Shape: [num_patches, H, W]

    return image_patches, weather_patches, label_patches

def get_data_loader(grid, batch_size):
    '''
    Args:
        grid - A single WSTATT data sample (eg. T11SKA_2018_0_0) as a string
        batch_size - Number of samples per batch; needed to determine how many batches are needed for each data_loader
    Returns:
        data_loader - A data_loader object containing shuffled batches of image, weather, and label patches
    '''
    if grid in GRID_CACHE:
        image_patches, weather_patches, label_patches = GRID_CACHE[grid]
    else:
        image_patches, weather_patches, label_patches = create_patches(grid)
        GRID_CACHE[grid] = (image_patches, weather_patches, label_patches)

    data = SEGMENTATION(image_patches, weather_patches, label_patches)

    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )