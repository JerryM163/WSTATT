import time

import torch
import numpy as np

from data import DataLoader, get_data_loader

# TODO: Add positional encoding for the 24 timestamps
# TODO: Potentially replace transformers with just transformer encoders
# TODO: Potentially add dropout layers in the transformers

# TODO: Build out the WSTATT version; figure out how the weather data is incorporated into the network

class MCT_STATT(torch.nn.Module):
    '''
    Utilizes the joint capabilities of CNNs and Transformers to derive a full spatial
    context for crop mapping.

    Architecture:
        Positional Encoder: ???????
        CTFusion Modules: Same input independently processed by CNN and Transformer sub modules,
            their result is then concatenated together and put through a max pooling operation
        MLP: Multi-layer perceptron for final classification

    '''
    def __init__(self, in_channels, out_channels):
        '''Initializes the model'''
        super(MCT_STATT, self).__init__()

        # --- CNN Sub Modules ---
        # CNN 1
        self.conv1_1 = torch.nn.Conv2d(10, 64, 3, padding=1)
        self.norm1_1 = torch.nn.BatchNorm2d(64)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.norm1_2 = torch.nn.BatchNorm2d(64)

        # CNN 2
        self.conv2_1 = torch.nn.Conv1d(128, 128, 3, padding=1)
        self.norm2_1 = torch.nn.BatchNorm1d(128)
        self.conv2_2 = torch.nn.Conv1d(128, 128, 3, padding=1)
        self.norm2_2 = torch.nn.BatchNorm1d(128)

        # CNN 3
        self.conv3_1 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.norm3_1 = torch.nn.BatchNorm1d(256)
        self.conv3_2 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.norm3_2 = torch.nn.BatchNorm1d(256)

        # --- Transformer Sub Modules ---
        # Initial Embedding layer
        self.linear1 = torch.nn.Linear(10240, 64)
        # Transformer 1
        self.transformer1 = torch.nn.Transformer( 
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            num_encoder_layers=2,
            batch_first=True
        )                                         
        # Transformer 2
        self.transformer2 = torch.nn.Transformer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            num_encoder_layers=2,
            batch_first=True
        )
        # Transformer 3
        self.transformer3 = torch.nn.Transformer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            num_encoder_layers=2,
            batch_first=True
        )

        # --- Multi-layer Perceptron Classifier
        self.mlp1 = torch.nn.Linear(512, 256)
        self.mlp2 = torch.nn.Linear(256, out_channels)

        # --- Shared Operations
        self.maxpool = torch.nn.MaxPool1d(2)
        self.relu = torch.nn.ReLU()

    def concat(self, cnn, trans):
        ''' Combine CNN and Transformer features '''
        return torch.cat([cnn, trans], dim=2)

    def forward(self, x):
        ''' Processes sentinel-2 data'''
        # Initial shape of 'x' is: (16, 24, 10, 32, 32)
        batches, timestamps, channels, height, width = x.shape

        # Reshape 'x' for use in the 1st CTFusion module: (384, 10, 32, 32)
        cnn_x = x.reshape(batches*timestamps, channels, height, width)

        # The transformer should take in a sequence like (batch, timestamps, features)
        # Reshape 'x' by combining channels and features: (16, 24, 10240)
        trans_out = x.reshape(batches, timestamps, channels*height*width)

        # Compress the features: (16, 24, 64)
        trans_out = self.linear1(trans_out)

        # --- CNN Sub-Module 1 ---
        cnn_x = self.conv1_1(cnn_x) # Conv2D
        cnn_x = self.norm1_1(cnn_x) # BatchNorm2D
        cnn_x = self.conv1_2(cnn_x) # Conv2D
        cnn_x = self.norm1_2(cnn_x) # BatchNorm2D
        cnn_out = self.relu(cnn_x)  # ReLU
        # Outputs Shape: (384, 64, 32, 32)

        # --- Transformer Sub-Module 1 ---
        trans_out = self.transformer1(trans_out, trans_out)
        # Outputs Shape: (16, 24, 64)

        # CNN and Transformer outputs don't match; revert CNN's outputs to be: (16, 24, 64, 32, 32)
        cnn_out = cnn_out.reshape(batches, timestamps, 64, height, width)

        # Finally, reshape the CNN outputs to be: (16, 24, 64) just like the Transformer
        cnn_out = cnn_out.mean(dim=(3,4))

        # Combine the features from both the CNN and Transformer: (16, 24, 128)
        concatenated = self.concat(cnn_out, trans_out)

        # Rearrange the vector: (16, 128, 24)
        concatenated = concatenated.transpose(1, 2)

        # Temporal pooling from 24 to 12: (16, 128, 12)
        pooled = self.maxpool(concatenated)

        # Prepare shape of trans_out: (16, 12, 128)
        trans_out = pooled.transpose(1, 2)

        # --- CNN Sub-Module 2 ---
        cnn_x = self.conv2_1(pooled) # Conv1D
        cnn_x = self.norm2_1(cnn_x)  # BatchNorm1D
        cnn_x = self.conv2_2(cnn_x)  # Conv1D
        cnn_x = self.norm2_2(cnn_x)  # BatchNorm1D
        cnn_out = self.relu(cnn_x)   # ReLU
        # Outputs (16, 128, 12)

        # --- Tranformer Sub-Module 2 ---
        trans_out = self.transformer2(trans_out, trans_out)
        # Outputs (16, 12, 128)

        # Rearrange the positions to match the Transformer output: (16, 12, 128)
        cnn_out = cnn_out.transpose(1, 2)

        # Combine the features from the CNN and Transformer sub modules: (16, 12, 256)
        concatenated = self.concat(cnn_out, trans_out)

        # Rearrage the vector to once again prepare for pooling: (16, 256, 12)
        concatenated = concatenated.transpose(1, 2)

        # Temporal pooling again: (16, 256, 6)
        pooled = self.maxpool(concatenated)

        # Prepare shape of trans_out: (16, 6, 256)
        trans_out = pooled.transpose(1, 2)

        # --- CNN Sub Module 3 --- 
        cnn_x = self.conv3_1(pooled) # Conv1D
        cnn_x = self.norm3_1(cnn_x)  # BatchNorm1D
        cnn_x = self.conv3_2(cnn_x)  # Conv1D
        cnn_x = self.norm3_2(cnn_x)  # BatchNorm1D
        cnn_out = self.relu(cnn_x)   # ReLU
        # Outputs: (16, 256, 6)

        # --- Transformer Sub Module 3 ---
        trans_out = self.transformer3(trans_out, trans_out)
        # Outputs: (16, 6, 256)

        # Rearrange the positions to match the Transformer output: (16, 6, 256)
        cnn_out = cnn_out.transpose(1, 2)

        # Combine the features from the CNN and Transformer sub modules: (16, 6, 512)
        concatenated = self.concat(cnn_out, trans_out)

        # Rearrage the vector to once again prepare for pooling: (16, 512, 6)
        concatenated = concatenated.transpose(1, 2)

        # Temporal pooling again: (16, 512, 3)
        pooled = self.maxpool(concatenated)

        # Average over time dimension: (16, 512)
        pooled = pooled.mean(dim=2)

        # --- Multi-Layer Perceptron Classifier ---
        mlp1 = self.mlp1(pooled) # Outputs (16, 256)
        relu = self.relu(mlp1)  
        result = self.mlp2(relu) # Outputs (16, 33)

        return result

class MCT_WSTATT(torch.nn.Module):
    def __init__(self, in_channels, in_channels_w, out_channels):
        super(MCT_WSTATT, self).__init__()




if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Active Device Status:", "cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 32
    out_channels = 33
    unknown_class = 100
    learning_rate = 0.0001
    batch_size = 16

    model = MCT_STATT(
        in_channels=in_channels, 
        out_channels=out_channels
    )

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    grids = ["T11SKA_2019_7_2"]

    for grid_num, grid in enumerate(grids):
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
                out = model(image_tensor)

            # Prepare labels for loss calculation
            label_patch_device = label_patch.type(torch.long).to(device)

            # Calculate loss
            batch_loss = criterion(out, label_patch_device)

            grid_loss += batch_loss.item()  # Accumulate batch loss

        # Calculate average loss for current grid
        statt_grid_loss = statt_grid_loss / (batch + 1)
        wstatt_grid_loss = wstatt_grid_loss / (batch + 1)
        print("\x1b[2K" + f'Grid Num: {grid_num:02} Grid: {grid} STATT Loss: {statt_grid_loss:.4f} WSTATT Loss: {wstatt_grid_loss:.4f}')
        statt_epoch_loss += statt_grid_loss
        wstatt_epoch_loss += wstatt_grid_loss