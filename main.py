import random

from device import device
from Models.baseline import STATT, WSTATT
from data import get_data_loader

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

    # List of all possible grid names in the google drive folder, based on their naming conventions
    datasets = [
        f"T11SKA_{year}_{first_digit}_{second_digit}"
        for year in (2018, 2019, 2020)
        for first_digit in range(10)
        for second_digit in range(10)
    ]

    print("########## BUILDING MODELS ##########")
    statt = STATT(
        in_channels=in_channels,
        out_channels=out_channels
    )
    print(f"\tSTATT Model Complete")
    wstatt = WSTATT(
        in_channels=in_channels,
        in_channels_w=in_channels_weather,
        out_channels=out_channels
    )
    print(f"\tWSTATT Model Complete")

    print("########## TRAINING MODELS ##########")
    statt = statt.to(device)
    wstatt = wstatt.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=unknown_class)

    statt_optim = torch.optim.Adam(statt.parameters(), lr=learning_rate)
    wstatt_optim = torch.optim.Adam(wstatt.parameters(), lr=learning_rate)

    statt_train_loss = []
    wstatt_train_loss = []

    for epoch in range(num_epochs):
        print(f'\tTraining EPOCH {epoch} ##')

        statt.train()
        wstatt.train()

        statt_epoch_loss = 0
        wstatt_epoch_loss = 0

        dataset = random.sample(datasets, 1)

        for grid_num, grid in enumerate(dataset):
            print(f'\r\t\tTraining GRID {grid} {grid_num + 1}', end="")
            data_loader = get_data_loader(grid, batch_size)

            statt_grid_loss = 0
            wstatt_grid_loss = 0

            for batch, [image_patch, weather_patch, label_patch] in enumerate(data_loader):
                statt_optim.zero_grad()
                wstatt_optim.zero_grad()

                image_tensor = image_patch.to(device)
                weather_tensor = weather_patch.to(device)
                label_patch = label_patch.type(torch.long).to(device)

                statt_out = statt(image_tensor)
                wstatt_out = wstatt(image_tensor, weather_tensor)

                statt_batch_loss = criterion(statt_out, label_patch)
                wstatt_batch_loss = criterion(wstatt_out, label_patch)

                statt_batch_loss.backward()
                wstatt_batch_loss.backward()

                statt_optim.step()
                wstatt_optim.step()

                statt_grid_loss += statt_batch_loss.item()
                wstatt_grid_loss += wstatt_batch_loss.item()

            statt_grid_loss_avg = statt_grid_loss / (batch + 1) 
            wstatt_grid_loss_avg = wstatt_grid_loss / (batch + 1)
            print(f'\t\tLoss for GRID {grid}: STATT {statt_grid_loss_avg}, WSTATT {wstatt_grid_loss_avg}')

            statt_epoch_loss += statt_grid_loss
            wstatt_epoch_loss += wstatt_grid_loss

        statt_epoch_loss_avg = statt_epoch_loss / (grid_num + 1)
        wstatt_epoch_loss_avg = wstatt_epoch_loss / (grid_num + 1)
        print(f'\tLoss for EPOCH {epoch}: STATT {statt_epoch_loss_avg}, WSTATT {wstatt_epoch_loss_avg}')

        statt_train_loss.append(statt_epoch_loss_avg)
        wstatt_train_loss.append(wstatt_epoch_loss_avg)

    # print("########## VALIDATING MODELS ##########")