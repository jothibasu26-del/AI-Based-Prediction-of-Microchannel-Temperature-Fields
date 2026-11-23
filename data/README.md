# Dataset

This directory is intended to document the CFD simulation data used for ANN-based microchannel temperature-field prediction.

## Dataset Overview

The dataset was generated from numerical simulations of microchannel heat transfer with localized hotspot heating.

| Parameter | Values |
|---|---|
| Number of CFD cases | 32 |
| Spatial observations | 161,376 |
| Grid per case | 41 × 123 |
| Peclet number (Pe) | 1, 5, 10, 55, 100 |
| Knudsen number (Kn) | 0, 0.01, 0.1 |
| TMAC | 0.5, 0.7, 0.9, 1.0 |

## Data Organization

The original CFD simulation files are expected to be organized in a folder named:

`hotspot_data/`

The EDA notebook uses these files to construct the combined spatial dataset and perform data analysis.

## Data Availability

Raw CFD data files are not included in this repository.

This keeps the repository focused on the analysis workflow, model implementations and evaluation results while avoiding unnecessary storage of large simulation datasets.

## Usage

To reproduce the analysis:

1. Obtain the original CFD simulation CSV files.
2. Place them in the expected `hotspot_data/` directory.
3. Run `notebooks/01_CFD_Data_Analytics_EDA.ipynb`.

Refer to the notebook documentation for further instructions.
