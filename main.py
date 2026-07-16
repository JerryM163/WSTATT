from random import random
from shutil import ignore_patterns

from torch import optim

from device import device
from Models.baseline import STATT, WSTATT
from data import get_random_sample, get_data_loader

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
    print("COMPLETED")

    for model in models:
        print("#######################################################################")
        print(f"TRAINING {model}")
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_loss = []

        for epoch in range(num_epochs):
            print(f'## EPOCH {epoch} ##')

            model.train()

            epoch_loss = 0

            dataset = get_random_sample()

            for grid_num, grid in enumerate(dataset):
                data_loader = get_data_loader(grid, batch_size)

                grid_loss = 0

                for batch, [image_patch, label_patch] in enumerate(data_loader):
                    optimizer.zero_grad()

                    output = model(image_patch.to(device))

                    label_patch = label_patch.type(torch.long).to(device)

                    batch_loss = criterion(output, label_patch)

                    batch_loss.backward() 
                    optimizer.step()     

                    grid_loss += batch_loss.item()

                grid_loss = grid_loss / (batch + 1) 
                print(f'\tGrid Number: {grid_num} Grid:{grid} Grid Loss:{grid_loss}')
                epoch_loss += grid_loss

            epoch_loss = epoch_loss / (grid_num + 1)
            print(f'\nTrain Loss:{epoch_loss}')
            train_loss.append(epoch_loss) 