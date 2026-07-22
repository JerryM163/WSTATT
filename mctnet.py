from os import times
import time

import torch
import numpy as np
from torch.nn.modules import BatchNorm2d, Conv2d, ReLU

from data import DataLoader, get_data_loader

# TODO: Add positional encoding for the 24 timestamps
# TODO: Potentially replace transformers with just transformer encoders
# TODO: Potentially add dropout layers in the transformers

# TODO: Build out the WSTATT version; figure out how the weather data is incorporated into the network

class CT_NET(torch.nn.Module):
    '''
    Utilizes the joint capabilities of CNNs and Transformers to derive a full spatial
    context for crop mapping.

    Architecture:
        Positional Encoder: ???????
        CTFusion Modules: Same input independently processed by CNN and Transformer sub modules,
            their result is then concatenated together and put through a max pooling operation
        MLP: Multi-layer perceptron for final classification

    '''
    def __init__(self, bands=10, classes=33):
        '''Initializes the model'''
        super(CT_NET, self).__init__()

        # --- CTFusion Module 1 ---

        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                bands,
                64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.embed_layer1 = torch.nn.Linear(bands, 64)
        self.trans1 = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
            ),
            num_layers=2,
        )

        # --- CTFusion Module 2 ---
        self.cnn2 = torch.nn.Sequential(
             torch.nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )
        self.embed_layer2 = torch.nn.Linear(128, 128)
        self.trans2 = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=512,
                batch_first=True,
            ),
            num_layers=2,
        )

        # --- CTFusion Module 3 ---
        self.cnn3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.embed_layer3 = torch.nn.Linear(256, 256)
        self.trans3 = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=2
        )

        # --- Pooling ---
        self.pool = torch.nn.MaxPool2d(2)

        # --- Fusion ---
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv2d(
                512, 
                256, 
                kernel_size=3, 
                padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                256, 
                256, 
                kernel_size=3, 
                padding=1
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        # --- Decoder ---
        self.decoder1 = torch.nn.Conv2d(
            256,
            128,
            kernel_size=3,
            padding=1,
        )
        self.decoder2 = torch.nn.Conv2d(
            128,
            64,
            kernel_size=3,
            padding=1,
        )

        # --- Classifer ---
        self.classifier = torch.nn.Conv2d(
            64,
            classes,
            kernel_size=1,
        )

    def concat(self, cnn, trans):
        ''' Combine CNN and Transformer features '''
        return torch.cat([cnn, trans], dim=2)
    
    def trans_forward(self, x, transformer, embedding):
        '''
            Input vector of shape:
                (batch, channels, height, width, timestamps)
            Outputs vector of shape:
                (batch, channels, height, width)
        '''

        batches, timestamps, channels, height, width = x.shape

        # Changes shape to: (batch, height, width, timestamps, channels)
        x = x.permute(0, 3, 4, 1, 2)
        
        # Can now operate per-pixel
        x = x.reshape(batches*height*width, timestamps, channels)

        x = embedding(x)

        x = transformer(x)

        # Collapses transformer output along temporal dimension
        x = x.mean(dim=1)

        x = x.reshape(batches, height, width, -1)

        # Changes shape to: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)

        return x

    def forward(self, x):
        ''' Processes sentinel-2 data'''
        batches, timestamps, channels, height, width = x.shape

        # --- Module 1 ---
        # Reshape input for 1st CNN: (384,10,32,32)
        cnn_x = x.reshape(batches*timestamps, channels, height, width)
        
        # Pass through 1st CNN sub module: (384,64,32,32)
        cnn_out = self.cnn1(cnn_x)

        # Reshape CNN output to original with updated channels: (16,24,64,32,32)
        cnn_out = cnn_out.reshape(batches, timestamps, 64, height, width)

        # Collapse CNN output along temporal dimension to match transformer output: (16,64,32,32)
        cnn_out = cnn_out.mean(dim=1)

        # Pass through 1st Transformer sub module: (16,64,32,32)
        trans_out = self.trans_forward(
            x,
            self.trans1,
            self.embed_layer1,
        )

        # Combines CNN and Transformer outputs: (16,128,32,32)
        x = torch.cat([cnn_out, trans_out], dim=1)

        # Pool features: (16,128,16,16)
        x = self.pool(x)

        # --- Module 2 ---
        # Pass through 2nd CNN sub module: (16,128,16,16)
        cnn_out = self.cnn2(x)

        # Permute input to send to Transformer: (16,16,16,128)
        trans_input = x.permute(0, 2, 3, 1)

        batches, height, width, channels = trans_input.shape

        # Rearrange input: (16,256,128)
        trans_input = trans_input.reshape(batches, height*width, channels)

        # Create an embedding to pass to the transformer: (16,256,128)
        trans_input = self.embed_layer2(trans_input)

        # Pass through the 2nd transformer: (16,256,128)
        trans_out = self.trans2(trans_input)

        # Reshapes transformer output: (16,16,16,128)
        trans_out = trans_out.reshape(batches, height, width, 128)

        # Permutes shape to be equal to the CNN's output: (16,128,16,16)
        trans_out = trans_out.permute(0,3,1,2)

        # Combines CNN and Transformer outputs: (16,256,16,16)
        x = torch.concat([cnn_out, trans_out], dim=1)

        # Pool features: (16,256,8,8)
        x = self.pool(x)

        # --- Module 3 ---
        # Pass to the 3rd CNN sub module: (16,256,8,8)
        cnn_out = self.cnn3(x)

        # Permute input to send to Transformer: (16,8,8,256)
        trans_input = x.permute(0, 2, 3, 1)

        batches, height, width, channels = trans_input.shape

        # Rearrange input: (16,64,256)
        trans_input = trans_input.reshape(batches, height*width, channels)

        # Create an embedding to pass to the transformer: (16,64,256)
        trans_input = self.embed_layer3(trans_input)

        # Pass through the 3rd transformer: (16,64,256)
        trans_out = self.trans3(trans_input)

        # Reshapes transformer output: (16,8,8,256)
        trans_out = trans_out.reshape(batches, height, width, 256)

        # Permutes shape to be equal to the CNN's output: (16,256,8,8)
        trans_out = trans_out.permute(0,3,1,2)

        # Combines CNN and Transformer outputs: (16,512,8,8)
        x = torch.concat([cnn_out, trans_out], dim=1)

        # --- Fusion Block ---
        # Prepare the output for the decoder and learn new spatial relationships: (16,256,8,8)
        x = self.fusion(x)

        # --- Decoder ---
        # Double height and width (16,256,16,16)
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        # Conv decoder for the channels: (16,128,16,16)
        x = self.decoder1(x)

        x = torch.nn.functional.relu(x)

        # Double height and width (16,128,32,32)
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        # Conv decoder for channels: (16,64,32,32)
        x = self.decoder2(x)

        x = torch.nn.functional.relu(x)

        # --- Classifier ---
        # Get pixel-wise predictions: (16,33,32,32)
        result = self.classifier(x)

        return result
       

class MCT_WSTATT(torch.nn.Module):
    def __init__(self, in_channels, in_channels_w, out_channels):
        super(MCT_WSTATT, self).__init__()




if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Active Device Status:", "cuda" if torch.cuda.is_available() else "cpu")

    bands = 10
    classes = 33
    unknown_class = 100
    learning_rate = 0.0001
    batch_size = 16

    model = CT_NET(bands, classes)

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
            label_tensor = label_patch.type(torch.long).to(device)

            with torch.no_grad():
                out = model(image_tensor)

            # Calculate loss
            batch_loss = criterion(out, label_tensor)

            grid_loss += batch_loss.item()  # Accumulate batch loss

        # Calculate average loss for current grid
        statt_grid_loss = statt_grid_loss / (batch + 1)
        wstatt_grid_loss = wstatt_grid_loss / (batch + 1)
        print("\x1b[2K" + f'Grid Num: {grid_num:02} Grid: {grid} STATT Loss: {statt_grid_loss:.4f} WSTATT Loss: {wstatt_grid_loss:.4f}')
        statt_epoch_loss += statt_grid_loss
        wstatt_epoch_loss += wstatt_grid_loss