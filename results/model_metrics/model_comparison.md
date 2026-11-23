# ANN Model Development and Performance Comparison

## 1. Overview

This project investigates artificial neural network (ANN) surrogate models for predicting temperature fields in a microchannel with localized hotspot heating.

Eight model configurations were evaluated using a dataset containing 161,376 spatial observations from 32 CFD cases.

The models were evaluated using 5-fold Leave-One-Peclet-Number-Out (LOPO) validation, testing generalization across five Peclet numbers: 1, 5, 10, 55 and 100.

The model development included three generations:

- **V13-A:** Baseline ANN.
- **V15:** Feature-engineering and feature-ablation experiments.
- **V16:** Softplus-based ANN models, including data-only and physics-regularized variants.

## 2. Global Performance Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| V16-A Softplus Data-Only | **0.9834** | **2.2165** | **0.8994** |
| V16-B Softplus Physics | 1.0045 | 2.2701 | 0.8945 |
| V16 Physics Softplus | 1.0019 | 2.2816 | 0.8934 |
| V13-A Baseline | 1.4599 | 3.4108 | 0.7618 |
| V15-C Near-Wall Hotspot | 1.5288 | 3.5603 | 0.7404 |
| V15-B Hotspot | 1.5710 | 3.6157 | 0.7323 |
| V15-D All Features | 1.6754 | 3.8686 | 0.6935 |
| V15-A Effective Kn | 1.6766 | 3.8693 | 0.6934 |

V16-A achieved the best global performance, with the lowest MAE and RMSE and the highest R².

Compared with V13-A, V16-A improved R² from 0.7618 to 0.8994.

## 3. Effect of Model Development

### 3.1 Softplus activation — V16 series

The V16 models introduced Softplus activation and a revised ANN architecture.

All three V16 configurations substantially outperformed the V13-A baseline in global prediction accuracy.

This indicates that the V16 modelling changes provided a major improvement over the earlier baseline.

### 3.2 Physics regularization

V16 and V16-B incorporated PDE and boundary-condition residual losses.

Compared with V16-A, the physics-regularized models had slightly lower global accuracy. However, V16 achieved the best prediction accuracy in the localized near-wall hotspot region.

This highlights a trade-off between overall temperature-field accuracy and localized near-wall prediction.

### 3.3 Feature engineering — V15 series

The V15 experiments investigated additional engineered features:

- Effective Knudsen number
- Hotspot indicator
- Near-wall hotspot indicator
- Combination of engineered features

All four V15 configurations performed worse globally than V13-A.

The results suggest that adding engineered features did not improve generalization for the current dataset, particularly at low Peclet numbers.

## 4. Generalization Across Peclet Numbers

The V16 models showed particularly strong improvements at low and moderate Peclet numbers.

| Model | Pe = 1 R² | Pe = 5 R² | Pe = 10 R² | Pe = 55 R² | Pe = 100 R² |
|---|---:|---:|---:|---:|---:|
| V16-A | 0.8101 | 0.9159 | 0.9766 | 0.8343 | 0.8650 |
| V16-B | 0.8005 | 0.9286 | 0.9794 | 0.8873 | 0.6942 |
| V16 | 0.7981 | 0.9487 | 0.9761 | 0.8300 | 0.7159 |
| V13-A | 0.5496 | 0.8248 | 0.9669 | 0.9720 | 0.8523 |

The V16 family substantially improved generalization at Pe = 1 compared with the earlier baseline.

At Pe = 100, V16-A retained stronger performance than the two physics-regularized V16 configurations.

## 5. Spatial Region Performance

| Model | Entire Domain R² | Heated Section R² | Hotspot Band R² | Near-Wall Hotspot R² |
|---|---:|---:|---:|---:|
| V16-A | **0.8994** | **0.8833** | **0.8444** | 0.8160 |
| V16 | 0.8934 | 0.8773 | 0.8392 | **0.8500** |
| V16-B | 0.8945 | 0.8694 | 0.8231 | 0.8168 |
| V13-A | 0.7618 | 0.7183 | 0.7070 | 0.6855 |

V16-A performed best across the entire domain, heated section and hotspot band.

V16 achieved the highest R² in the near-wall hotspot region (Y ≤ 0.2), with R² = 0.8500 and MAE = 1.5172.

This suggests that physics regularization may improve localized near-wall temperature prediction, even when global accuracy is slightly lower.

## 6. Peak Temperature Prediction

Peak-temperature predictions were evaluated across all 32 CFD cases.

| Model | Mean Relative Peak Error | Median Relative Peak Error | Within 10% | Within 20% |
|---|---:|---:|---:|---:|
| V16-A | **13.08%** | 14.54% | **13/32** | 20/32 |
| V16-B | 15.11% | **12.54%** | 10/32 | **22/32** |
| V16 | 16.27% | 18.00% | 11/32 | 19/32 |
| V13-A | 22.79% | 22.68% | 4/32 | 15/32 |

All three V16 models kept all 32 evaluated cases within 30% peak-temperature error.

V16-A achieved the lowest mean relative peak error, while V16-B achieved the lowest median error and the most cases within 20%.

## 7. Final Model Selection

### V16-A — General-purpose surrogate

Selected as the primary model because it achieved:

- Best global R²: 0.8994
- Lowest global MAE: 0.9834
- Lowest global RMSE: 2.2165
- Lowest mean relative peak-temperature error: 13.08%

### V16 — Near-wall gradient surrogate

Retained as the physics-regularized alternative because it achieved the best near-wall hotspot prediction:

- Near-wall hotspot R²: 0.8500
- Near-wall hotspot MAE: 1.5172

### V16-B — Alternative physics-regularized model

Retained for comparison because it achieved:

- Lowest median relative peak-temperature error: 12.54%
- 22 out of 32 peak predictions within 20% error.

## 8. Conclusion

The comparison demonstrates that the V16 Softplus models provide a substantial improvement over the earlier ANN configurations.

V16-A is the strongest general-purpose surrogate for predicting the microchannel temperature field. V16 provides the best localized near-wall hotspot prediction, while V16-B offers competitive peak-temperature prediction consistency.

The results also demonstrate that additional feature engineering does not necessarily improve model generalization. Model selection should consider global accuracy, operating-regime generalization, localized temperature prediction and hotspot peak accuracy rather than relying on a single performance metric.

---

**Note:** Results are based on the current dataset and evaluation procedure. Further validation with additional CFD cases and operating conditions is required before treating the models as broadly generalizable surrogates.
