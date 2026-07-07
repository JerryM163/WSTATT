import torch

class WSTATT(torch.nn.Module):
    def __init__(self, in_channels, in_channels_w, out_channels):
        super(WSTATT, self).__init__()  # Initialize parent class (torch.nn.Module)

        # Feature dimensions for spatial and weather data processing
        self.s_fe = 64  # Spatial feature base dimension
        self.w_fe = 8   # Weather feature base dimension
        s_fe = self.s_fe
        w_fe = self.w_fe

        # Spatial Encoder Path (Downsampling)
        # Each block: Conv -> ReLU -> Conv -> ReLU -> MaxPool
        self.conv1_1 = torch.nn.Conv2d(in_channels, s_fe, 3, padding=1)  # First convolution
        self.conv1_2 = torch.nn.Conv2d(s_fe, s_fe, 3, padding=1)         # Second convolution
        self.conv2_1 = torch.nn.Conv2d(s_fe, s_fe*2, 3, padding=1)       # Increase features
        self.conv2_2 = torch.nn.Conv2d(s_fe*2, s_fe*2, 3, padding=1)     # Maintain features
        self.conv3_1 = torch.nn.Conv2d(s_fe*2, s_fe*4, 3, padding=1)     # Double features again
        self.conv3_2 = torch.nn.Conv2d(s_fe*4, s_fe*4, 3, padding=1)     # Maintain features

        # Decoder Path (Upsampling) with Skip Connections
        self.unpool2_cat = torch.nn.ConvTranspose2d(s_fe*8 + w_fe*8, s_fe*2, kernel_size=2, stride=2)  # Upsample + feature reduction
        self.upconv2_1 = torch.nn.Conv2d(s_fe*4, s_fe*2, 3, padding=1)   # Reduce features after skip connection
        self.upconv2_2 = torch.nn.Conv2d(s_fe*2, s_fe*2, 3, padding=1)   # Process features
        self.unpool1 = torch.nn.ConvTranspose2d(s_fe*2, s_fe, kernel_size=2, stride=2)  # Final upsampling
        self.upconv1_1 = torch.nn.Conv2d(s_fe*2, s_fe, 3, padding=1)     # Reduce features
        self.upconv1_2 = torch.nn.Conv2d(s_fe, s_fe, 3, padding=1)       # Final processing

        # Output layer (1x1 convolution for channel reduction)
        self.out = torch.nn.Conv2d(s_fe, out_channels, kernel_size=1, padding=0)

        # Shared Operations
        self.maxpool = torch.nn.MaxPool2d(2)    # Downsamples by 2x
        self.relu = torch.nn.ReLU()             # Activation function
        self.dropout = torch.nn.Dropout(p=0.1)  # Regularization

        # Temporal Processing (LSTMs)
        # For spatial features: bidirectional LSTM
        self.lstm = torch.nn.LSTM(s_fe*4, s_fe*4, batch_first=True, bidirectional=True)
        # For weather data: bidirectional LSTM
        self.lstm_w = torch.nn.LSTM(in_channels_w, w_fe*4, batch_first=True, bidirectional=True)

        # Attention Mechanism
        self.attention_encode = torch.nn.Linear(s_fe*8 + w_fe*8, 1)  # Computes attention scores

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)  # Xavier init for better convergence

    def crop_and_concat(self, x1, x2):
        """Aligns and concatenates encoder features (x1) with decoder features (x2)"""
        # Calculate cropping offsets for center alignment
        offset_2 = (x1.shape[2] - x2.shape[2]) // 2  # Height offset
        offset_3 = (x1.shape[3] - x2.shape[3]) // 2  # Width offset

        # Crop the larger tensor (x1) to match x2's spatial dimensions
        x1_crop = x1[:, :, offset_2:offset_2+x2.shape[2], offset_3:offset_3+x2.shape[3]]

        # Concatenate along channel dimension
        return torch.cat([x1_crop, x2], dim=1)

    def forward(self, x_s, x_w):
        """Forward pass with spatial data (x_s) and weather data (x_w)"""
        s_fe = self.s_fe
        w_fe = self.w_fe

        # Reshape inputs: Combine batch and sequence dimensions
        batch_size_s, seq_len_s, channels_s, input_patch_size, _ = x_s.shape
        batch_size_w, seq_len_w, channels_w, _, _ = x_w.shape
        x_s = x_s.view(-1, channels_s, input_patch_size, input_patch_size)  # [batch*seq, C, H, W]
        x_w = x_w.view(-1, seq_len_w, channels_w)  # [batch, seq_w, C_w] (after view)

        # --- Spatial Encoder Path ---
        # Block 1
        conv1 = self.relu(self.conv1_1(x_s))
        conv1 = self.relu(self.conv1_2(conv1))
        maxpool1 = self.maxpool(conv1)  # Downsample 2x

        # Block 2
        conv2 = self.relu(self.conv2_1(maxpool1))
        conv2 = self.relu(self.conv2_2(conv2))
        maxpool2 = self.maxpool(conv2)  # Downsample 4x from original

        # Block 3 (Deepest features)
        conv3 = self.relu(self.conv3_1(maxpool2))
        conv3 = self.relu(self.conv3_2(conv3))  # Now at 1/4 original resolution

        # --- Temporal Processing ---
        # Prepare spatial features for LSTM: [batch*H*W, seq_len, features]
        shape_enc = conv3.shape  # Save original shape for later reshaping
        conv3 = conv3.view(-1, seq_len_s, conv3.shape[1], conv3.shape[2]*conv3.shape[3])
        conv3 = conv3.permute(0, 3, 1, 2)  # [batch, H*W, seq_len, features]
        conv3 = conv3.reshape(conv3.shape[0]*conv3.shape[1], seq_len_s, s_fe*4)  # [batch*H*W, seq_len, features]

        # Process spatial sequence with LSTM
        lstm, _ = self.lstm(conv3)  # Output: [batch*H*W, seq_len, s_fe*8] (bidirectional)

        # Process weather data with LSTM
        lstm_w, _ = self.lstm_w(x_w)  # Output: [batch, seq_len_w, w_fe*8]

        # Align weather features with spatial sequence length
        lstm_subset = lstm_w[:, 15::15, :]  # Sample every 15th timestep
        lstm_subset = lstm_subset[:, :seq_len_s, :]  # Match spatial sequence length

        # Expand weather features to spatial dimensions
        lstm_subset = lstm_subset.reshape(lstm_subset.shape[0], lstm_subset.shape[1], lstm_subset.shape[2], 1, 1)
        lstm_subset = lstm_subset.repeat(1, 1, 1, int(input_patch_size/4), int(input_patch_size/4))  # Tile to H/4 x W/4
        lstm_subset = lstm_subset.view(-1, seq_len_s, lstm_subset.shape[2], lstm_subset.shape[3]*lstm_subset.shape[4])
        lstm_subset = lstm_subset.permute(0, 3, 1, 2)
        lstm_subset = lstm_subset.reshape(lstm_subset.shape[0]*lstm_subset.shape[1], seq_len_s, w_fe*8)  # [batch*H*W, seq_len, w_fe*8]

        # --- Attention Mechanism ---
        # Combine spatial and weather features
        lstm_concat = torch.cat([lstm, lstm_subset], dim=2)  # [batch*H*W, seq_len, s_fe*8 + w_fe*8]
        lstm_concat_relu = self.relu(lstm_concat.reshape(-1, w_fe*8 + s_fe*8))  # Flatten for linear layer

        # Compute attention weights
        attn_scores = self.attention_encode(torch.tanh(lstm_concat_relu))  # [batch*H*W*seq_len, 1]
        attn_scores = attn_scores.view(-1, shape_enc[2], shape_enc[3], seq_len_s)  # [batch, H, W, seq_len]
        attn_scores = attn_scores.permute(0, 3, 1, 2)  # [batch, seq_len, H, W]
        attention_weights = torch.nn.functional.softmax(
            torch.squeeze(torch.nn.functional.avg_pool2d(attn_scores, shape_enc[2])),  # Average over H/W
            dim=1  # Softmax over sequence dimension
        )  # [batch, seq_len]

        # Create context vector (weighted sum of features)
        attn_weights_4d = attention_weights.view(-1, 1, 1, seq_len_s).repeat(1, shape_enc[2], shape_enc[3], 1)
        weighted_features = (attn_weights_4d.view(-1, 1) * lstm_concat_relu)  # Element-wise multiplication
        context = torch.sum(weighted_features.view(-1, seq_len_s, s_fe*8 + w_fe*8), dim=1)  # Sum over sequence
        context = context.view(-1, shape_enc[2], shape_enc[3], s_fe*8 + w_fe*8).permute(0, 3, 1, 2)  # [batch, C, H, W]

        # Detach attention weights to stop gradient flow during decoder aggregation
        attention_weights_fixed = attention_weights.detach()

        # --- Decoder Path ---
        # Upsample context features
        unpool2_cat = self.unpool2_cat(context)  # Upsample 2x to 1/2 original resolution

        # Aggregate encoder features using attention weights
        conv2_view = conv2.view(-1, seq_len_s, conv2.shape[1], conv2.shape[2], conv2.shape[3])
        agg_conv2 = torch.sum(attention_weights_fixed.view(-1, seq_len_s, 1, 1, 1) * conv2_view, dim=1)

        # Combine with upsampled context
        concat2 = self.crop_and_concat(agg_conv2, unpool2_cat)  # Skip connection
        upconv2 = self.relu(self.upconv2_1(concat2))
        upconv2 = self.relu(self.upconv2_2(upconv2))

        # Final upsampling block
        unpool1 = self.unpool1(upconv2)  # Upsample to original resolution

        # Aggregate first encoder block features
        conv1_view = conv1.view(-1, seq_len_s, conv1.shape[1], conv1.shape[2], conv1.shape[3])
        agg_conv1 = torch.sum(attention_weights_fixed.view(-1, seq_len_s, 1, 1, 1) * conv1_view, dim=1)

        # Combine features
        concat1 = self.crop_and_concat(agg_conv1, unpool1)  # Skip connection
        upconv1 = self.relu(self.upconv1_1(concat1))
        upconv1 = self.relu(self.upconv1_2(upconv1))

        # Output layer
        out = self.out(upconv1)  # Final 1x1 convolution
        return out