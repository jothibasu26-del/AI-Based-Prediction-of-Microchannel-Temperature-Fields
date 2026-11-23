# AI-Based Prediction of Microchannel Temperature Fields

### ANN-Based Surrogate Modelling for Microchannel Hotspot Cooling

## Project Overview

This project develops Artificial Neural Network (ANN) surrogate models to predict temperature distributions in a microchannel with localized hotspot heating.

The work explores the application of machine learning to computational fluid dynamics (CFD), with the objective of predicting temperature fields from spatial coordinates and operating parameters without repeatedly solving the complete numerical problem.

The project combines CFD data analysis, exploratory data analysis (EDA), feature engineering, neural network development and physics-regularized learning.

## Research Problem

Microchannel cooling is an important thermal management approach for compact electronic systems. Localized hotspots can create steep temperature gradients and elevated wall temperatures.

Numerical simulation provides detailed temperature-field information but can be computationally expensive when repeated across different operating conditions.

An ANN-based surrogate model offers a potential way to approximate these temperature fields efficiently.

## Dataset

The dataset was generated from numerical simulations of microchannel heat transfer with localized hotspot heating.

| Property | Description |
|---|---|
| CFD cases | 32 |
| Spatial observations | 161,376 |
| Grid per case | 41 × 123 |
| Peclet numbers (Pe) | 1, 5, 10, 55, 100 |
| Knudsen numbers (Kn) | 0, 0.01, 0.1 |
| TMAC values | 0.5, 0.7, 0.9, 1.0 |
| Prediction target | Dimensionless temperature field |

The dataset contains spatial coordinates and operating parameters associated with the corresponding temperature values.

## Project Workflow

1. **Data Preparation:** Combined and organized CFD simulation data from multiple operating conditions.
2. **Exploratory Data Analysis:** Examined data distributions, parameter sensitivity, correlations and interactions between operating parameters.
3. **Feature Engineering:** Investigated additional physical and spatial features.
4. **ANN Development:** Developed and evaluated multiple neural network configurations.
5. **Physics Regularization:** Investigated PDE residual and boundary-condition losses.
6. **Model Evaluation:** Compared global accuracy, operating-regime generalization, regional prediction accuracy and hotspot peak-temperature errors.

## Model Development

Eight ANN configurations were evaluated across three development generations.

| Generation | Models | Description |
|---|---|---|
| V13 | V13-A | Baseline ANN |
| V15 | V15-A, V15-B, V15-C, V15-D | Feature-engineering and ablation experiments |
| V16 | V16, V16-A, V16-B | Softplus-based ANN and physics-regularization experiments |

### Final Model Configurations

- **V16-A:** Softplus ANN without physics-loss regularization.
- **V16:** Softplus ANN with PDE and boundary-condition regularization.
- **V16-B:** Alternative physics-regularized Softplus ANN configuration.

## Model Evaluation

The models were evaluated using five-fold Leave-One-Peclet-Number-Out (LOPO) validation.

Each evaluation fold holds out one Peclet-number group to assess generalization to an unseen flow regime.

### Global Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| V16-A | **0.9834** | **2.2165** | **0.8994** |
| V16-B | 1.0045 | 2.2701 | 0.8945 |
| V16 | 1.0019 | 2.2816 | 0.8934 |
| V13-A | 1.4599 | 3.4108 | 0.7618 |

### Key Results

- V16-A achieved the highest global R² of 0.8994.
- V16-A achieved the lowest mean relative peak-temperature error of 13.08%.
- V16 achieved the best near-wall hotspot R² of 0.8500.
- V16-B achieved the lowest median relative peak-temperature error of 12.54%.
- All three V16 models kept all 32 evaluated peak-temperature predictions within 30% relative error.

## Key Findings

1. Softplus-based V16 models substantially improved global prediction performance compared with the V13-A baseline.
2. Additional engineered features in the V15 experiments did not improve global generalization on the current dataset.
3. Physics regularization provided a promising improvement in localized near-wall hotspot prediction.
4. Global temperature-field accuracy and localized hotspot prediction require separate evaluation.

## Repository Structure

```text
AI-Based-Prediction-of-Microchannel-Temperature-Fields/
│
├── notebooks/
│   └── 01_CFD_Data_Analytics_EDA.ipynb
│
├── src/
│   └── models/
│       ├── train_ann_v16_physics_regularized.py
│       ├── train_ann_v16a_softplus_data_only.py
│       └── train_ann_v16b_softplus_physics.py
│
├── data/
├── figures/
├── results/
│   ├── analysis_tables/
│   └── model_metrics/
│
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- Jupyter Notebook

## Current Best Models

| Objective | Model |
|---|---|
| General-purpose temperature prediction | V16-A |
| Near-wall hotspot prediction | V16 |
| Peak-temperature median accuracy | V16-B |

## Future Work

- Expand the CFD dataset with additional operating conditions.
- Improve prediction accuracy at low Peclet numbers.
- Investigate physics-informed loss weighting and boundary-condition enforcement.
- Evaluate computational speed against conventional CFD simulations.
- Develop additional visualizations comparing predicted and reference temperature fields.

---

**Note:** The reported results represent the current dataset and validation setup. Further testing on additional CFD cases is needed to establish broader generalization.
