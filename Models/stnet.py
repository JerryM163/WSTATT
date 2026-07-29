from multiprocessing import context

import torch

class TemporalAttentionPooling(torch.nn.Module):
    def __init__(self, channels, context_dim, nheads):
        super(TemporalAttentionPooling, self).__init__()

        self.channels = channels       # Feature dimensions of input vector
        self.nheads = nheads     

        # Adapt previous context to feature dimensions
        self.context_proj = torch.nn.Linear(context_dim, channels)

        # Calculates score of channels per each head
        self.scores = torch.nn.Linear(channels, nheads)

        # Combines scores from each head
        self.head_fusion = torch.nn.Linear(channels*nheads,channels)

        # Takes in pooled attention score to get new context
        self.update_context = torch.nn.Sequential(
            torch.nn.Linear(
                channels,
                context_dim,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                context_dim,
                context_dim,
            )
        )

    def forward(self, x, context):
        # Compute scores
        scores = self.scores(x)
        scores = scores.permute(0,2,1)

        print("Scores:", scores.shape)

        # Apply previous context to scores when relevant
        if context is not None:
            var = self.context_proj(context).unsqueeze(1)
            print("Unsqueezed Context Proj:", var.shape)
            x += var

        # Compute attention from scores
        attention = torch.softmax(scores, dim=1)
        
        # Get an attention score for each head
        outputs = []
        for h in range(self.nheads):
            alpha = attention[:,h].unsqueeze(-1)

            pooled = torch.sum(
                alpha*x, 
                dim=1,
            )

            outputs.append(pooled)

        # Fuse each head's computed attention score
        pooled = torch.cat(
            outputs,
            dim=-1,
        )
        pooled = self.head_fusion(pooled)

        # Accumulates temporal memory throughout the encoder path
        out_context = self.update_context(pooled)
        if context is not None:
            out_context += context

        return pooled, attention, out_context



class STNET(torch.nn.Module):
    """
    Hierarchical, Spatio-Temporal, U-Net Model for sequence-based image segmentation

    Architecture:
        Input Sequence
                │
                ▼
        =================================================  
        Hierarchical Spatio-Temporal Encoder              
        ------------------------------------------------- 
        HST Block 1                                       ┌► HST Block 3
        • CNN Feature Extraction                          │  • CNN Feature Extraction
        • Pixel-wise Temporal Transformer                 │  • Pixel-wise Temporal Transformer
        • Temporal Attention Pooling                      │  • Context-guided Temporal Attention Pooling
        • Context Update                                  │  • Residual Context Update
        • Temporal Skip Feature                           │          │
                │                                         │          ▼
                │                                         │  =================================================
                ▼                                         │  U-Net Decoder
        Downsample                                        │  ------------------------------------------------- 
        ------------------------------------------------- │  • Upsampling
        HST Block 2                                       │  • Concatenate Temporal Skip Features
        • CNN Feature Extraction                          │  • CNN Refinement
        • Pixel-wise Temporal Transformer                 │          │
        • Context-guided Temporal Attention Pooling       │          ▼
        • Residual Context Update                         │  Output Segmentation Mask
        • Temporal Skip Feature                           │ 
                │                                         │
                │                                         │
                ▼                                         │
        Downsample ───────────────────────────────────────┘
    """
    def __init__(self, in_channels, out_channels):
        super(STNET, self).__init__()

        # --- Encoder Layer ---
        # Block 1: Maintains spatial resolution; gains temporal understanding at widest layer
        self.conv1 = self.getConvBlock(in_channels, 64) # Output: 64 channels
        self.trans1 = self.getTransEncoder(64,2,1)      # Input received from 1st maxpool

        # Block 2: Doubles features, halves resolution; deeper temporal relationships
        self.conv2 = self.getConvBlock(64, 128)         # Output: 128 channels
        self.trans2 = self.getTransEncoder(128,4,1)     # Input received from 2nd maxpool

        # Block 3: Deepest features, halves resolution again; final temporal encoding
        self.conv3 = self.getConvBlock(128, 256)        # Output: 256 channels
        self.trans3 = self.getTransEncoder(256,8,2)     # Input received from 3rd maxpool

        # --- Decoder Path (Upsampling) ---
        # Upsampling Block 2 (from deepest features)
        self.unpool2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 2x upsampling
        self.upconv2_1 = torch.nn.Conv2d(256, 128, 3, padding=1)       # After skip connection
        self.upconv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)       # Output: 128 channels

        # Upsampling Block 1 (final resolution)
        self.unpool1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 2x upsampling
        self.upconv1_1 = torch.nn.Conv2d(128, 64, 3, padding=1)        # After skip connection
        self.upconv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)         # Output: 64 channels

        # --- Temporal Pooling ---
        self.temp_pool1 = TemporalAttentionPooling(64, 64, nheads=2)
        self.temp_pool2 = TemporalAttentionPooling(128, 64, nheads=4)
        self.temp_pool3 = TemporalAttentionPooling(256, 64, nheads=8)

        # --- Shared Operations ---
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

        # --- Output layer (classifier) ---
        self.out = torch.nn.Conv2d(64, out_channels, kernel_size=1, padding=0)  # 1x1 conv
    
    def getConvBlock(self, in_channels, out_channels):
        """
        Blueprint for similar Convolutional Layers 

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

    def getTransEncoder(self, d_model, nhead, num_layers):
        """
        Blueprint for similar Transformer encoders

        Returns:
            A Transformer encoder that picks up on temporal relationships
        """
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            batch_first=True,
        )

        return torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def crop_and_concat(self, x1, x2):
        """
        Aligns and concatenates encoder features (x1) with decoder features (x2).

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
        """
        Forward pass for sequence of satellite images.

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

        conv1 = conv1.reshape(batches,timestamps,64,height,width)
        conv1 = conv1.permute(0,3,4,1,2)
        conv1 = conv1.reshape(batches*height*width,timestamps,64) # Output: (batches*32*32,timestamps,64)

        # Pass through encoder's 1st Transformer Encoder
        trans1 = self.trans1(conv1) # Output: (batches*32*32,timestamps,64)

        # Temporal pooling on the 1st transformer's output
        pooled1, alpha1, context1 = self.temp_pool1(trans1, None) # Output: (batches*32*32,64)

        # Reshape transformer's output for maxpooling
        trans1 = trans1.reshape(batches,height,width,timestamps,64)
        trans1 = trans1.permute(0,3,4,1,2)
        trans1 = trans1.reshape(batches*timestamps,64,height,width) # Output: (batches*timestamps,64,32,32)

        # Halve the spatial resolution (divide features by 2)
        maxpool1 = self.maxpool(trans1) # Output: (batches*timestamps,64,16,16)

        # Pass through encoder's 2nd convolutional year
        conv2 = self.conv2(maxpool1) # Output: (batches*timestamps,128,16,16)

        conv2 = conv2.reshape(batches,timestamps,128,16,16)
        conv2 = conv2.permute(0,3,4,1,2)
        conv2 = conv2.reshape(batches*16*16,timestamps,128) # Output: (batches*16*16,timestamps,128)

        # Pass through encoder's 1st Transformer Encoder
        trans2 = self.trans2(conv2) # Output: (batches*16*16,timestamps,128)

        # Temporal pooling on the 2nd transformer's output
        pooled2, alpha2, context2 = self.temp_pool2(trans2, context1) # Output: (batches*16*16,128)

        trans2 = trans2.reshape(batches,16,16,timestamps,128)
        trans2 = trans2.permute(0,3,4,1,2)
        trans2 = trans2.reshape(batches*timestamps,128,16,16) # Output: (batches*timestamps,128,16,16)

        # Halve the spatial resolution (divide features by 2)
        maxpool2 = self.maxpool(trans2) # Output: (batches*timestamps,128,8,8)

        # Pass through encoder's 3rd convolutional year
        conv3 = self.conv3(maxpool2) # Output: (batches*timestamps,256,8,8)

        conv3 = conv3.reshape(batches,timestamps,256,8,8)
        conv3 = conv3.permute(0,3,4,1,2)
        conv3 = conv3.reshape(batches*8*8,timestamps,256) # Output: (batches*8*8,timestamps,256)

        # Pass through encoder's 1st Transformer Encoder
        trans3 = self.trans3(conv3) # Output: (batches*8*8,timestamps,256)

        # Temporal pooling on the 3rd transformer's output
        pooled3, alpha3, context3 = self.temp_pool3(trans3, context2) # Output: (batches*8*8,256)

        # Prepare encoder's output and skip connections for the decoder
        pooled3 = pooled3.reshape(batches,8,8,256)
        pooled3 = pooled3.permute(0,3,1,2) # Output: (batches,256,8,8)

        pooled2 = pooled2.reshape(batches,16,16,128)
        pooled2 = pooled2.permute(0,3,1,2) # Output: (batches,128,16,16)

        pooled1 = pooled1.reshape(batches,32,32,64)
        pooled1 = pooled1.permute(0,3,1,2) # Output: (batches,64,32,32)

        # --- Decoder Path ---
        # Upsample features (from 1/4 to 1/2 resolution)
        unpool2 = self.unpool2(pooled3) # Output: (batches,128,16,16)

        # Combine with upsampled context (skip connection)
        concat2 = self.crop_and_concat(pooled2, unpool2)  # Output: (batches,256,16,16)

        # Process combined features
        upconv2 = self.relu(self.upconv2_1(concat2))
        upconv2 = self.relu(self.upconv2_2(upconv2))  # Output: (batches,128,16,16)

        # Final upsampling (from 1/2 to full resolution)
        unpool1 = self.unpool1(upconv2)  # Output: (batches,64,32,32)

        # Combine with upsampled features (skip connection)
        concat1 = self.crop_and_concat(pooled1, unpool1)  # Output: (batches,128,32,32)

        # Final convolution processing
        upconv1 = self.relu(self.upconv1_1(concat1))
        upconv1 = self.relu(self.upconv1_2(upconv1))  # Output: (batches,64,32,32)

        # --- Output layer (class prediction per pixel) ---
        out = self.out(upconv1)  # Output: (batches,33,32,32)

        # Return output from classifier layer
        return out

    def __str__(self):
        return "STNet Model"

        

