# Notebooks

This folder contains the Jupyter notebooks used for data analysis and ANN model development in the microchannel hotspot-cooling project.

## 1. CFD Data Analytics and Exploratory Data Analysis

**File:** `01_CFD_Data_Analytics_EDA.ipynb`

This notebook performs exploratory data analysis on the CFD simulation dataset.

### Analyses performed

- Dataset inspection and parameter extraction
- Duplicate parameter-combination checks
- Spatial dataset construction
- Data quality and coverage analysis
- Overall and case-level descriptive statistics
- Temperature distribution analysis
- Parameter-wise comparisons
- Parameter sensitivity analysis
- Correlation analysis
- Knudsen number and TMAC interaction analysis
- Export of analysis tables

### Input data

The notebook expects the raw CFD CSV files in a folder named:

`hotspot_data/`

Place this folder in the working directory from which the notebook is executed.

### Output

The notebook exports analysis tables and figures into:

`results/data_analytics/`

## 2. Microchannel AI ANN Model Development

**File:** `02_Microchannel_AI_ANN_V16.py`

This file documents the V16 ANN model development workflow.

The final training implementations for V16, V16-A and V16-B are maintained separately in:

`src/models/`

Refer to the model comparison documentation in `results/model_metrics/` for the evaluation results.

## Requirements

Install the project dependencies using:

```bash
pip install -r requirements.txt
```

Run the notebooks using Jupyter:

```bash
jupyter notebook
```

**Note:** The raw CFD dataset is not included in this repository. Users must provide the simulation CSV files in the expected input directory before executing the EDA notebook.
