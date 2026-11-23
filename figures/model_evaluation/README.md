# Model Evaluation Figures

This directory contains visualizations generated from the out-of-fold predictions of the ANN models developed for microchannel temperature-field prediction.

## Figures

| Figure | Description |
|---|---|
| 01_actual_vs_predicted.png | Comparison of CFD reference temperatures and ANN predictions |
| 02_temperature_field_comparison.png | CFD temperature field, V16-A prediction and absolute error |
| 03_absolute_error_distributions.png | Distribution of absolute prediction errors |
| 04_global_model_comparison.png | Comparison of global MAE, RMSE and R² |
| 05_r2_across_peclet_numbers.png | Model performance across different Peclet numbers |
| 06_peak_temperature_errors.png | Peak-temperature prediction errors across CFD cases |

## Supporting Data

The CSV files contain:
- Global model evaluation metrics
- Peak-temperature prediction errors for individual CFD cases
- R² values across Peclet numbers

## Evaluation Method

The figures are based on out-of-fold predictions from five-fold Leave-One-Peclet-Out (LOPO) evaluation.

The purpose is to assess model generalization across different operating conditions and compare the performance of V16, V16-A and V16-B.
