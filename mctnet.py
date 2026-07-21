import torch

from data import DataLoader, get_data_loader

# TODO: Add parameters to module functions
# TODO: Build out the WSTATT version; figure out how the weather data is incorporated into the network

class MCT_STATT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCT_STATT, self).__init__()

        self.transformer1 = torch.nn.Transformer(
            d_model=32,
            nhead=4,
            batch_first=True
        )

        self.conv1_1 = torch.nn.Conv1d(32, 64, 3, padding=1)
        self.norm1_1 = torch.nn.BatchNorm1d(64)

        self.conv1_2 = torch.nn.Conv1d(64, 128, 3, padding=1)
        self.norm1_2 = torch.nn.BatchNorm1d(128)

        self.transformer2 = torch.nn.Transformer(
            d_model=32,
            nhead=4,
            batch_first=True
        )

        self.conv2_1 = torch.nn.Conv1d(80, 160, 3, padding=1)
        self.norm2_1 = torch.nn.BatchNorm1d(160)

        self.conv2_2 = torch.nn.Conv1d(160, 320, 3, padding=1)
        self.norm2_2 = torch.nn.BatchNorm1d(320)

        self.transformer3 = torch.nn.Transformer(
            d_model=80,
            nhead=10,
            batch_first=True
        )

        self.conv3_1 = torch.nn.Conv1d(32, 64, 3, padding=1)
        self.norm3_1 = torch.nn.BatchNorm1d(64)

        self.conv3_2 = torch.nn.Conv1d(64, 128, 3, padding=1)
        self.norm3_2 = torch.nn.BatchNorm1d(128)

        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.softmax = torch.nn.Softmax(out_channels)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

    def concat(self, x1, x2):
        return torch.cat([x1, x2], dim=2)

    def forward(self, x):
        x1, x2, x3, x4, x5 = x.shape
        x = x.view(x1*x2*x3, x4, x5)

        print("Initially:", x.shape)

        cnn_out = self.conv1_1(x)
        cnn_out = self.norm1_1(cnn_out)
        cnn_out = self.conv1_2(cnn_out)
        cnn_out = self.norm1_2(cnn_out)
        print("After 1st CNN:", cnn_out.shape)

        self.transformer1(x, x)
        print("After 1st Transformer:", x.shape)

        concat = self.concat(cnn_out, x)
        print("After 1st Concat:", concat.shape)
        pooled = self.maxpool(concat)
        print("After 1st Maxpool:", pooled.shape)

        cnn_out = self.conv2_1(pooled)
        cnn_out = self.norm2_1(cnn_out)
        cnn_out = self.conv2_2(cnn_out)
        cnn_out = self.norm2_2(cnn_out)
        print("After 2nd CNN:", cnn_out.shape)

        self.transformer2(pooled, x)
        print("After 2nd Transformer:", x.shape)

        concat = self.concat(cnn_out, x)
        print("After 2nd Concat:", concat.shape)
        pooled = self.maxpool(concat)
        print("After 2nd Maxpool:", pooled.shape)

        cnn_out = self.conv3_1(pooled)
        cnn_out = self.norm3_1(cnn_out)
        cnn_out = self.conv3_2(cnn_out)
        cnn_out = self.norm3_2(cnn_out)
        print("After 3rd CNN:", cnn_out.shape)

        self.transformer3(pooled, x)
        print("After 3rd Transformer:", x.shape)

        concat = self.concat(cnn_out, x)
        print("After 3rd Concat:", concat.shape)
        pooled = self.maxpool(concat)
        print("After 3rd Maxpool:", pooled.shape)

        linear = self.linear(pooled)
        print("After linear:", linear.shape)
        result = self.softmax(linear)

        return result

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
        data_loader = get_data_loader(grid, batch_size)

        optim.zero_grad()

        for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):

            out = model(image_patch.to(device))

            optim.step()