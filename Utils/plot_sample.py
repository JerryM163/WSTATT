import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

class_color_list = ['#ffd300','#ff2626','#00a8e2','#ffff00','#e2007c','#a57000','#d6d600','#a50000','#ffcc66','#f2a377','#ff00ff','#704489','#ffff7c',
                '#00a582','#e8d6af','#00ff8c','#ff6666','#334933','#af9970','#ffa5e2','#a5f28c','#ccbfa3','#bfbf77','#93cc93','#93cc93','#93cc93',
                '#e8bfff','#c6d69e','#e8ffbf','#7cafaf','#7cafaf','#4970a3','#9a9a9a',"#9773cd"] ## add unknown to end and subtract one from vamx in imshow()
colormap = mpl.colors.ListedColormap(class_color_list)

#weather_bands = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
weather_bands = ['dayl', 'srad', 'tmax', 'tmin']
#weather_bands = ['prcp', 'swe', 'vp']

def plot_sample():
    grid = input("Grid to visualize (T11SKA_20##_#_#): ")

    combined_label_path = f"G:/.shortcut-targets-by-id/1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX/WSTATT_DATA/LABEL_DATA/NUMPY/COMBINED_LABELS/{grid}_combined_label.npy"
    label_path = f"G:/.shortcut-targets-by-id/1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX/WSTATT_DATA/LABEL_DATA/NUMPY/ERODED_LABELS/{grid}_label.npy"
    sat_path = f"G:/.shortcut-targets-by-id/1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX/WSTATT_DATA/SATELLITE/NUMPY/{grid}_image.npy"
    weather_path = f"G:/.shortcut-targets-by-id/1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX/WSTATT_DATA/WEATHER/DAYMET/{grid}_daymet_10980_global_normalised_year_day_average_grid_array.npy"

    # Load in grid data
    sat_array = np.load(sat_path)
    combined_label_array = np.load(combined_label_path)
    eroded_label_array = np.load(label_path)
    weather_array = np.load(weather_path)

    # Take note of the array shapes
    print("Satellite Data Shape:", sat_array.shape)
    print("Combined Label Shape:", combined_label_array.shape)
    print("Eroded Label Shape:", eroded_label_array.shape)
    print("Weather Data Shape:", weather_array.shape)

    timestamps = int(input("How many timestamps are you measuring (6,12,18,24)?: ").strip())

    # Generates 'timestamps' evenly spaced numbers between 0 to 24
    timesteps = np.linspace(0, 23, timestamps, dtype=int)

    time_steps_w = weather_array.shape[0]

    cols = 6
    rows = len(timesteps)//6

    plt.figure(figsize=(24, 2*rows)) # VISUALIZING SATELLITE DATA

    for idx, step in enumerate(timesteps):
        color_bands = sat_array[step,0:3,:,:]
        clip = 1
        channel_stats = {}
        image = []
        for channel in [2,1,0]:
            array = color_bands[channel]
            channel_stats[channel] = {
                "Max": np.percentile(array, 100-clip), 
                "Min": np.percentile(array, clip)
            }
            array[array>channel_stats[channel]["Max"]] = channel_stats[channel]["Max"]
            array[array<channel_stats[channel]["Min"]] = channel_stats[channel]["Min"]
            array = (array - channel_stats[channel]["Min"])*255/(channel_stats[channel]["Max"]-channel_stats[channel]["Min"])
            image.append(array)
        image = np.array(image).astype(np.uint8)
        image = image.transpose(1,2,0)

        plt.subplot(rows, cols, idx + 1)
        plt.axis('off')
        plt.imshow(image)

    plt.tight_layout(pad = 0.1)

    plt.figure(figsize=(16, 8)) # VISUALIZING LABELS

    plt.subplot(2, 1, 1)
    plt.imshow(combined_label_array, cmap=colormap, interpolation='none', vmin=0, vmax=len(class_color_list)-1)
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(eroded_label_array, cmap=colormap, interpolation='none', vmin=0, vmax=len(class_color_list)-1)
    plt.axis('off')

    plt.tight_layout(pad = 0.1)

    fig = plt.figure(figsize=(5 * len(weather_bands), 5)) # VISUALIZING WEATHER DATA

    for band_num,band in enumerate(weather_bands):

        band_sequence = np.squeeze(weather_array[:, band_num, :, :])
        step_size_w = 1

        plt.subplot(1, len(weather_bands), band_num + 1)

        x_ticks = np.arange(time_steps_w, step=step_size_w) + 1
        x = np.arange(time_steps_w, step=step_size_w) + 1
        y = band_sequence[::step_size_w]

        plt.plot(x,y,color='#FF0000',linewidth=2)
        plt.ylabel(band,fontsize = 10)
        plt.xlabel('Time',fontsize = 10)
        plt.title(str(band),fontsize = 12)
        plt.xticks(x_ticks,fontsize = 2, rotation='vertical')

    plt.show()

if __name__ == "__main__":
    plot_sample()