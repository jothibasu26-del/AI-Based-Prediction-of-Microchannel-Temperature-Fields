# AI-Based Prediction of Microchannel Temperature Fields

An Artificial Neural Network (ANN)-based surrogate modelling project for predicting temperature distributions in a microchannel with localized hotspot heating, using CFD-generated simulation data.

The project combines computational heat transfer, exploratory data analysis, feature engineering, and machine learning to investigate whether an ANN can reproduce CFD temperature fields and generalize to unseen operating conditions.

## Project Overview

Numerical simulations provide detailed temperature fields, but performing CFD simulations for every operating condition can be computationally expensive. This project explores an ANN-based surrogate model that learns the relationship between spatial coordinates, flow conditions, rarefaction parameters, and temperature.

The model predicts the dimensionless temperature field across the microchannel using spatial and operating-condition inputs.

### Engineering Problem

The study considers heat transfer in a microchannel with a localized hotspot along the heated bottom wall.

- Computational domain: \(X \in [-10,30]\) as represented by the simulation dataset's coordinate range.
- Transverse domain: \(Y \in [0,1]\).
- Heated section: \(X \in [0,10]\).
- Localized hotspot: \(X \in [4,6]\).
- Flow and rarefaction conditions are varied using Peclet number, Knudsen number, and tangential momentum accommodation
