# Model Metrics and Evaluation Results

This directory contains quantitative evaluation results from the ANN surrogate models developed for microchannel temperature-field prediction.

## 1. Model Comparison

`model_comparison.md`

Provides the comparative analysis of eight ANN configurations:

- V13-A baseline
- V15-A, V15-B, V15-C and V15-D feature-ablation models
- V16, V16-A and V16-B Softplus models

## 2. Global Performance Results

The following CSV files contain global model evaluation metrics:

| File | Model |
|---|---|
| `ann_v16_global_results.csv` | V16 |
| `ann_v16a_global_results.csv` | V16-A |
| `ann_v16b_global_results.csv` | V16-B |

## 3. Peak Temperature Results

The following files contain peak-temperature prediction evaluation across 32 CFD cases:

| File | Model |
|---|---|
| `ann_v16_peak_temperature_results.csv` | V16 |
| `ann_v16a_peak_temperature_results.csv` | V16-A |
| `ann_v16b_peak_temperature_results.csv` | V16-B |

## 4. Evaluation Method

The models were evaluated using five-fold Leave-One-Peclet-Number-Out (LOPO) validation.

Performance was assessed using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Maximum prediction error
- Regional prediction accuracy
- Relative peak-temperature error

## 5. Main Findings

- V16-A achieved the best global prediction accuracy.
- V16 achieved the best near-wall hotspot prediction accuracy.
- V16-B achieved the lowest median relative peak-temperature error.

Refer to `model_comparison.md` for the detailed results and interpretation.

## Note

The reported metrics correspond to the current dataset and validation procedure.
