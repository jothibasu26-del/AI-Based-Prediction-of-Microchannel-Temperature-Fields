# CFD Dataset

## Dataset Overview

This project uses numerical simulation data generated from a microchannel heat-transfer study involving localized hotspot cooling under slip-flow conditions.

## Dataset Details

- Number of simulation cases: 32
- Spatial grid: 41 × 123
- Spatial observations: 161,376
- Input parameters: Peclet number (Pe), Knudsen number (Kn), and TMAC
- Spatial coordinates: X and Y
- Target variable: Dimensionless temperature

## Data Organization

Each simulation case represents a different combination of operating parameters. The dataset contains temperature values across the computational domain.

## Usage

Place the original simulation CSV files in the `hotspot_data/` directory before running the exploratory data analysis notebook.

**Note:** Raw simulation files are not included in this repository. Add them only if you have permission to share the data.
