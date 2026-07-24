import os
import sys

# Drop the cluster's default library injection tracking completely
os.environ.pop("LD_LIBRARY_PATH", None)

# Force the environment link loader to stick strictly to the conda environment
conda_lib = "/users/0/hinsv006/miniconda3/envs/jerry/lib"
os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:/lib64"
sys.path.insert(0, conda_lib)

import torch
import numpy as np

from Models.statt import STATT, WSTATT

from Utils.early_stopper import EarlyStopper
from train import train_epoch
from val import validate_epoch

if __name__ == "__main__":
    # --- Model Variables ---
    model_choice = None # Chosen by the user 
    model = None        # Determined by 'model_choice'

    in_channels = 10  # Input variable for Sentinel-2's 10 satellite bands
    out_channels = 33 # Output variable for the 33 possible crop classes

    time_choices = [6,12,18,24] # Used to determine whether the user has selected valid timestamps
    timestamps = None           # Chosen by the user

    bands = []            # Chosen by the user
    in_channels_weather = 7 # Input variable for Daymet's 7 weather bands

    # --- Training/Validation Variables ---
    max_epochs = 40

    unknown_class = 100
    learning_rate = 0.0001

    train_dataset = np.load(r"../WSTATT_DATA/DISTRIBUTION/T11SKA/train_set_T11SKA_DISTRI1.npy").tolist()
    val_dataset = np.load(r"../WSTATT_DATA/DISTRIBUTION/T11SKA/validation_set_T11SKA_DISTRI1.npy").tolist()

    batch_size = 16

    threshold = 50000
    labels_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    class_names = ['Corn','Cotton','Rice','Sunflower','Barley','Winter_Wheat','Safflower','Dry Beans','Onions','Tomatoes',
                   'Cherries','Grapes','Citrus','Almonds','Walnut','Pistachio','Garlic','Olives','Pomegranates','Alfalfa',
                   'Hay','Barren_land','Fallow_and_Idle','Deciduous_Forests','Evergreen_forest','Mixed_Forests',
                   'Clover_and_wildflower','Shrubland','Grass','Woody_wetlands','Herbaceous_Wetlands','Water','Urban']
    
    train_loss = []
    val_loss = []

    # --- Early Stopping Variables ---
    patience = 3
    min_delta = 0.05
    
    print("########## BUILDING MODELS ##########")
    # Get which model the user chooses
    print("## Choosing Your Model ##")
    while model_choice != "w" and model_choice != "s":
        model_choice = input("What model are you using (W)STATT/(S)TATT:? ").strip().lower()

    if model_choice == "w":
        print("WSTATT Selected!")
    else:
        print("STATT Selected!")
    
    # Get the number of timestamps the user is testing with
    print("## Choosing Your Timestamps ##")
    while timestamps not in time_choices:
        timestamps = int(input("How many timestamps are you measuring (6,12,18,24)?: ").strip())
    
    # When applicable, get the bands of weather the user is testing for
    if model_choice == "w":
        print("## Choosing Your Weather Bands ##")
        print("Weather bands: [dayl, prcp, srad, swe, tmax, tmin, vp]")
        bands = input("What bands of weather do you need (Ex. 0 2 4 5)?: ")
        bands = bands.split()
        bands = [int(band) for band in bands]

        in_channels_weather = len(bands)
        
        # Create the WSTATT model
        model = WSTATT(
            in_channels=in_channels,
            in_channels_w=in_channels_weather,
            out_channels=out_channels,
        )
        model_file = f"Wstatt-{timestamps}-{bands.sort().join("-")}.pt"
    else:
        # Create the STATT model
        model = STATT(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        model_file = f"Statt-{timestamps}.pt"

    # Load the model if a file already exists in the same directory
    if os.path.isfile(model_file):
        model.load_state_dict(torch.load(model_file),strict = False)
        print(f"{model} LOADED")
    else:
        print(f"{model} COMPLETE")

    early_stopper = EarlyStopper(patience, min_delta)
    print(f"Early Stopper Created with PATIENCE: {patience} and MAX EPOCHS: {max_epochs}")

    for epoch in np.arange(max_epochs):
        epoch_train_loss = train_epoch(
            epoch=epoch+1,
            model=model,
            unknown_class=unknown_class,
            learning_rate=learning_rate,
            dataset=train_dataset,
            batch_size=batch_size,
            timestamps=timestamps,
            bands=bands,
        )
        # Saves the model with its current parameters when its epoch_train_loss is less than the previous epoch
        if(len(train_loss)==0 or epoch_train_loss < train_loss[-1]):
            torch.save(model.state_dict(), model_file)            
        train_loss.append(epoch_train_loss)

        epoch_val_loss = validate_epoch(
            epoch=epoch,
            model=model,
            unknown_class=unknown_class,
            learning_rate=learning_rate,
            val_dataset=val_dataset,
            batch_size=batch_size,
            timestamps=timestamps,
            threshold=threshold,
            class_names=class_names,
            labels_list=labels_list,
            bands=bands,
        )
        val_loss.append(epoch_val_loss)

        if early_stopper.early_stop(epoch_val_loss):
            print(f"Early Stopped Activated at EPOCH {epoch}")
            break
        
    print("Model COMPLETE")

    