import torch
from torch.nn.modules import TransformerEncoderLayer

class STNET(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STNET, self).__init__()

        # --- Encoder Layer ---
        # Block 1: Maintains spatial resolution; gains temporal understanding at widest layer
        self.conv1 = self.getConvBlock(in_channels, 64) # Output: 64 channels
        self.trans1 = self.getTransEncoder(64)          # Input received from 1st maxpool

        # Block 2: Doubles features, halves resolution; deeper temporal relationships
        self.conv2 = self.getConvBlock(64, 128)         # Output: 128 channels
        self.trans2 = self.getTransEncoder(128)         # Input received from 2nd maxpool

        # Block 3: Deepest features, halves resolution again; final temporal encoding
        self.conv3 = self.getConvBlock(128, 256)        # Output: 256 channels
        self.trans3 = self.getTransEncoder(256)         # Input received from 3rd maxpool

        # --- Shared Operations ---
        self.maxpool = torch.nn.MaxPool2d(2)

        # --- Decoder Path (Upsampling) ---
        # Upsampling Block 2 (from deepest features)
        self.unpool2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 2x upsampling
        self.upconv2_1 = torch.nn.Conv2d(256, 128, 3, padding=1)       # After skip connection
        self.upconv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)       # Output: 128 channels

        # Upsampling Block 1 (final resolution)
        self.unpool1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 2x upsampling
        self.upconv1_1 = torch.nn.Conv2d(128, 64, 3, padding=1)        # After skip connection
        self.upconv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)         # Output: 64 channels
    
    def getConvBlock(self, in_channels, out_channels):
        """Blueprint for similar Convolutional Layers 

        Returns:
            A CNN encoder layer that halves spatial resolution 
            to pick up on smaller and smaller spatial relationships
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(),
        )

    def getTransEncoder(self, d_model):
        """Blueprint for similar Transformer encoders

        Returns:
            A Transformer encoder that picks up on temporal relationships
        """
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
        )

        return torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=1,
        )

    def crop_and_concat(self, x1, x2):
        """Aligns and concatenates encoder features (x1) with decoder features (x2).

        Used for skip connections. Center-crops x1 to match x2's spatial dimensions.
        """
        # Calculate cropping offsets for center alignment
        offset_2 = (x1.shape[2] - x2.shape[2]) // 2  # Height offset
        offset_3 = (x1.shape[3] - x2.shape[3]) // 2  # Width offset

        # Crop the larger tensor (x1) to match x2's size
        x1_crop = x1[:, :, offset_2:offset_2+x2.shape[2], offset_3:offset_3+x2.shape[3]]

        # Concatenate along channel dimension
        return torch.cat([x1_crop, x2], dim=1)

    def forward(self, x):
        """Forward pass for sequence of satellite images.

        Args:
            x_s: Input sequence tensor [batch, seq_len, channels, height, width]

        Returns:
            Segmentation map [batch, out_channels, height, width]
        """
        # Get input dimensions
        batches, timestamps, channels, height, width = x.shape

        # Combine batch and sequence dimensions
        x = x.view(-1, channels, height, width) # Output: (batches*timestamps,10,32,32)
        
        # --- Encoder Path ---
        # Pass through encoder's 1st convolutional layer
        conv1 = self.conv1(x) # Output: (batches*timestamps,64,32,32)

        # Flatten spatial dimensions to conserve pixel-wise segmentation
        conv1_reshaped = conv1.view(batches*timestamps,64,height*width)
        conv1_reshaped = conv1_reshaped.permute(2,0,1) # Output: (1024,batches*timestamps,64)

        # Pass through encoder's 1st Transformer Encoder
        trans1 = self.trans1(conv1_reshaped) # Output: (1024,batches*timestamps,64)

        # Reshape transformer's output to show the features once again
        trans1 = trans1.permute(1,2,0)
        trans1 = trans1.view(batches*timestamps,64,height,width) # Output: (batches*timestamps,64,32,32)

        # Halve the spatial resolution (divide features by 2)
        maxpool1 = self.maxpool(trans1) # Output: (batches*timestamps,64,16,16)

        # Pass through encoder's 2nd convolutional year
        conv2 = self.conv2(maxpool1) # Output: (batches*timestamps,128,16,16)

        # Flatten spatial dimensions to conserve pixel-wise segmentation
        conv2_reshaped = conv2.view(batches*timestamps,128,256)
        conv2_reshaped = conv2_reshaped.permute(2,0,1) # Output: (256,batches*timestamps,128)

        # Pass through encoder's 1st Transformer Encoder
        trans2 = self.trans2(conv2_reshaped) # Output: (256,batches*timestamps,128)

        # Reshape transformer's output to show the features once again
        trans2 = trans2.permute(1,2,0)
        trans2 = trans2.view(batches*timestamps,128,16,16) # Output: (batches*timestamps,128,16,16)

        # Halve the spatial resolution (divide features by 2)
        maxpool2 = self.maxpool(trans2) # Output: (batches*timestamps,128,8,8)

        # Pass through encoder's 3rd convolutional year
        conv3 = self.conv3(maxpool2) # Output: (batches*timestamps,256,8,8)

        # Flatten spatial dimensions to conserve pixel-wise segmentation
        conv3_reshaped = conv3.view(batches*timestamps,64,256)
        conv3_reshaped = conv3_reshaped.permute(1,0,2) # Output: (64,batches*timestamps,256)

        # Pass through encoder's 1st Transformer Encoder
        trans3 = self.trans3(conv3_reshaped) # Output: (64,batches*timestamps,256)

        # Reshape transformer's output to show the features once again
        trans3 = trans3.permute(1,2,0)
        trans3 = trans3.view(batches,timestamps,256,8,8) # Output: (16,timestamps,256,8,8)

        # Average across the time dimension 
        encoder_out = trans3.mean(dim=1) # Output: (16,256,8,8)

        # Apply the same temporal reduction to the skip connections
        conv2 = conv2.view(batches,timestamps,128,16,16).mean(dim=1) # Output: (16,128,16,16)
        conv1 = conv1.view(batches,timestamps,64,height,width).mean(dim=1)      # Output: (16,64,32,32)

        # --- Decoder Path ---
        # Upsample features (from 1/4 to 1/2 resolution)
        unpool2 = self.unpool2(encoder_out) # Output: (16,128,16,16)

        # Combine with upsampled context (skip connection)
        concat2 = self.crop_and_concat(conv2, unpool2)  # Output: (16,256,16,16)

        # Process combined features
        upconv2 = self.relu(self.upconv2_1(concat2))
        upconv2 = self.relu(self.upconv2_2(upconv2))  # Output: (16,128,16,16)

        # Final upsampling (from 1/2 to full resolution)
        unpool1 = self.unpool1(upconv2)  # Output: (16,64,32,32)

        # Combine with upsampled features (skip connection)
        concat1 = self.crop_and_concat(conv1, unpool1)  # Output: (16,128,32,32)

        # Final convolution processing
        upconv1 = self.relu(self.upconv1_1(concat1))
        upconv1 = self.relu(self.upconv1_2(upconv1))  # Output: (16,64,32,32)

        # Output layer (class prediction per pixel)
        out = self.out(upconv1)  # Output: (16,33,32,32)

        return out

    def __str__(self):
        return "STNet Model"



        

