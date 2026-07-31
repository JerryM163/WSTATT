# AI4Earth 2026: WSTATT

This repository contains the project files for the WSTATT group as a part of the University of Minnesota's AI4Earth research program in 2026.

## Repository Contents

```text
.
├── Models/
|    ├── statt.py
|    └── stnet.py
├── References/ (Jupyter notebooks from the original WSTATT repository)
|    ├── Explore_data.ipynb
|    ├── Explore_STATT.ipynb
|    └── Explore_WSTATT.ipynb
├── Results/
|    ├── Predictions/ (contains pngs of different models' predicted segmentation masks)
|    ├── SavedModels/ (trained models saved as pt files if you want to see how they predict)
|    └── AI4EarthWstatt.pptx (final slideshow presented at the end of the program)
├── Utils/
|    ├── data.py
|    ├── device.py
|    ├── early_stopper.py
|    └── plot_sample.py
├── main.py
├── test.py
├── train.py
└── val.py
```

# Models

```statt.py``` contains baseline STATT and WSTATT crop classification models as outlined in their original papers.

1. STATT - [Attention-augmented Spatio-Temporal Segmentation for Land Cover Mapping](https://arxiv.org/pdf/2105.02963)

2. WSTATT - [Combining Satellite and Weather Data for Crop Type Mapping](https://arxiv.org/pdf/2401.15875)

```stnet.py``` contains our custom transformer-based crop classification model.

# References

For more info on the notebooks in particular, visit the original WSTATT repository where these were created:

https://github.com/praveen-ravirathinam/WSTATT


1. `Explore_data.ipynb`  
   Used for exploring the dataset, including satellite images, crop labels, eroded labels, and weather variables.

2. `Explore_STATT.ipynb`  
   Implements and tests the STATT model, which uses multi-temporal satellite imagery for crop classification.

3. `Explore_WSTATT.ipynb`  
   Implements and tests the WSTATT model, which extends STATT by adding weather information along with satellite imagery.

# Results

Folder containing our trained models and final predictions at the end of the program.

# Utils

```data.py``` holds all the functions required for preparing the data to be loaded into the models

```device.py``` contains a global variable that informs all other dependent files whether *cuda* is available

```early_stopper.py``` outlines an Early Stopper class that improves model training

```plot_sample.py``` contains many useful tools for visualizations of a sample satellite grid

# Model Training Loop

```main.py``` central loop that prompts the user to initialize a model then trains and validates it until either the max number of epochs is reached or the early stopper activates

```train.py``` trains a model for a single epoch and saves its losses

```val.py``` validates a model for a single epoch, saves it losses, and drafts a classification report

```test.py``` allows you to visualize a trained model's performance

# Dataset

Preprocessed data is present at: https://drive.google.com/drive/folders/1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX?usp=drive_link

The dataset contains satellite imagery, weather data, and crop labels.

The dataset structure should be:

```text
WSTATT_DATA/
├── DISTRIBUTION/
|    └── T11SKA/
|         ├── test_set_T11SKA_DISTRI1.npy
|         ├── train_set_T11SKA_DISTRI1.npy
|         └── validation_set_T11SKA_DISTRI1.npy
├── LABEL_DATA/
|    └── NUMPY/
|         ├── COMBINED_LABELS/
|         └── ERODED_LABELS/
├── LABEL_MAPS/
├── SATELLITE/
|    └── NUMPY/
└── WEATHER/
     └── DAYMET/
```

## Data Format

Satellite data is stored as NumPy arrays with the format:

```text
[timesteps, channels, height, width]
```

Label data is stored as 2D NumPy arrays with the format:

```text
[height, width]
```

Weather data includes variables such as:

```text
dayl, prcp, srad, swe, tmax, tmin, vp
```

## Crop Classes

The project uses multiple crop and land-cover classes, including crops such as corn, cotton, rice, wheat, tomatoes, grapes, almonds, pistachio, alfalfa, and others.

Unknown or ignored pixels are represented using:

```python
unknown_class = 100
```

These pixels are ignored during training and evaluation.

## Acknowledgement

This repository is prepared for experiments with STATT and WSTATT-style spatio-temporal crop classification using satellite imagery and weather data.

## Citation

This repository was based off of Praveen Ravirathinam's WSTATT repository, containing 3 Jupyter Notebooks we used as references.

That repository also cites the following 2 papers which the provided baseline models were based off of: 

```bibtex
@inproceedings{ravirathinam2024wstatt,
  title={Combining Satellite and Weather Data for Crop Type Mapping: An Inverse Modelling Approach},
  author={Ravirathinam, Praveen and Ghosh, Rahul and Khandelwal, Ankush and Jia, Xiaowei and Mulla, David and Kumar, Vipin},
  booktitle={Proceedings of the 2024 SIAM International Conference on Data Mining},
  pages={445--453},
  year={2024},
  publisher={SIAM},
  doi={10.1137/1.9781611978032.52}
}

@inproceedings{ghosh2021statt,
  title={Attention-augmented Spatio-Temporal Segmentation for Land Cover Mapping},
  author={Ghosh, Rahul and Ravirathinam, Praveen and Jia, Xiaowei and Lin, Chenxi and Jin, Zhenong and Kumar, Vipin},
  booktitle={Proceedings of the 2021 IEEE International Conference on Big Data},
  pages={1399--1408},
  year={2021},
  publisher={IEEE},
  doi={10.1109/BigData52589.2021.9671974}
}


```
