from shutil import ignore_patterns

from torch import optim

from device import device
from Models.baseline import STATT, WSTATT
from data import get_data_loaders

import torch

if __name__ == "__main__":
    in_channels = 10
    in_channels_weather = 7
    out_channels = 33

    unknown_class = 100
    learning_rate = 0.0001
    num_epochs = 10

    input_patch_size = 32
    output_patch_size = 32
    batch_size = 16

    print("#######################################################################")
    print("BUILDING MODELS")
    statt = STATT(
        in_channels=in_channels,
        out_channels=out_channels
    )
    wstatt = WSTATT(
        in_channels=in_channels,
        in_channels_w=in_channels_weather,
        out_channels=out_channels
    )

    models = [statt, wstatt]

    for model in models:
        print("#######################################################################")
        print(f"TRAINING {model}")
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_loss = []
        data_loaders = get_data_loaders(num_epochs, batch_size)

        for epoch in range(num_epochs):
            print(f'## EPOCH {epoch} ##')

            model.train()

            epoch_loss = 0

            for (idx, data_loader) in enumerate(data_loaders):
                loader_loss = 0

                for batch, [image_patch, label_patch] in enumerate(data_loader):
                    optimizer.zero_grad()

                    output = model(image_patch.to(device))

                    label_patch = label_patch.type(torch.long).to(device)

                    batch_loss = criterion(output, label_patch)

                    batch_loss.backward() 
                    optimizer.step()     

                    loader_loss += batch_loss.item()

                loader_loss = loader_loss / (batch + 1) 
                print(f'\tLoader Number: {idx}\Loader:{data_loader} \Loader Loss:{loader_loss}')
                epoch_loss += loader_loss

            epoch_loss = epoch_loss / (idx + 1)
            print(f'\nTrain Loss:{epoch_loss}')
            train_loss.append(epoch_loss) 