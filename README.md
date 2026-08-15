# Solar Irradiance Forecasting with CNN-WNN-MMHA

This project implements a hybrid deep learning framework for short-term global irradiance forecasting based on meteorological and solar radiation variables collected in Tunisia. The model is designed to capture both local patterns and long-range temporal dependencies in irradiance data, which are essential for reliable forecasting under changing weather and daylight conditions.

## Abstract

Accurate prediction of global irradiance is critical for optimizing energy management in photovoltaic (PV) systems, particularly in solar-powered electric vehicles (ESVs). However, traditional models struggle to capture the complex spatial and temporal dependencies in irradiance data, limiting prediction accuracy under varying weather conditions. Existing approaches, including statistical methods, conventional machine learning models, and standalone deep learning techniques like LSTM, fail to integrate local features and long-term dependencies simultaneously, creating a need for more robust solutions.

This paper introduces a novel hybrid framework, CNN-WNN-MMHA, that combines Convolutional Neural Networks (CNN), Wavelet Neural Networks (WNN), and a Masked Multi-Head Attention (MMHA) mechanism. The CNN extracts spatial and local features, WNN performs frequency decomposition to capture multi-scale variations, and MMHA models temporal dependencies while encoding positional information. The model is trained and evaluated on a real-world climatic dataset from Tunisia, collected over eight years. Experimental results demonstrate that the proposed model significantly outperforms state-of-the-art methods such as LSTM, BiLSTM, and CNN-LSTM, achieving a 79% reduction in MAPE and superior generalization performance across diverse weather scenarios. This advancement enhances energy forecasting reliability, supporting smarter route planning and energy optimization for solar-powered vehicles, with potential extensions to other renewable energy systems.

## Dataset

The data used for the forecasting model comes from the Energy Research Center, located at the Borj Cedria Science and Technology Park in the north of Tunisia (Latitude: 36.717°, Longitude: 10.427°). The dataset spans from January 1, 2015, to December 31, 2022, with hourly resolution.

The recorded variables include:

- Global Irradiance (GI): total solar radiation on a horizontal surface, W/m²
- Clearsky Global Horizontal Irradiance (Clearsky GHI): modeled clear-sky GHI, W/m²
- Direct Normal Irradiance (DNI): radiation normal to the sun rays, W/m²
- Clearsky Direct Normal Irradiance (Clearsky DNI): modeled clear-sky DNI, W/m²
- Diffuse Horizontal Irradiance (DHI): diffuse radiation on a horizontal plane, W/m²
- Clearsky Diffuse Horizontal Irradiance (Clearsky DHI): modeled clear-sky DHI, W/m²
- Solar Height (SH): sun elevation angle, degrees
- Solar Zenith Angle: angle between the vertical direction and the sun, degrees
- Is Day: binary indicator for daytime or nighttime
- Ambient Temperature (T): air temperature at 2 m, °C
- Wind Speed (WS): wind speed at 10 m, m/s
- Wind Direction (WD): wind direction, degrees
- Relative Humidity (RH): humidity, %
- Precipitation (P): rainfall or snowfall, mm/h
- Cloud Cover (CC): cloud obstruction percentage
- Atmospheric Pressure (PA): pressure, hPa

## Modeling approach

The forecasting framework combines the following modules:

1. CNN block: captures local temporal and meteorological patterns from the multivariate input window.
2. WNN block: performs frequency-aware decomposition and extracts multi-scale variations.
3. MMHA block: models long-range dependencies and temporal interactions while preserving positional information.
4. Regression head: outputs the predicted global irradiance value for the next step.

The model is trained on historical multivariate time sequences and evaluated on unseen cities or hold-out periods to assess generalization.

## Project structure

- `src/train.py`: model training pipeline
- `src/evaluate.py`: evaluation and plotting of predictions
- `src/preprocessing.py`: data loading and feature engineering
- `src/dataset.py`: sequence generation for time-series forecasting
- `src/models/`: model implementations
- `configs/solar.yaml`: configuration for the irradiance forecasting experiments
- `results/`: saved model checkpoints, metrics, and prediction outputs

## Usage

1. Prepare the dataset as a CSV file with the required columns and store it in `data/raw/solar.csv`.
2. Update the configuration in `configs/solar.yaml` if needed.
3. Train the model:

```bash
python -m src.train --config configs/solar.yaml
```

4. Evaluate the trained model:

```bash
python -m src.evaluate --config configs/solar.yaml
```

## Expected outputs

The training pipeline produces:

- model checkpoint in `results/`
- training history JSON
- scaler artifact
- evaluation metrics (MAE, RMSE, MAPE, R²)
- prediction CSV and visualization plots

## Notes

This repository provides a reproducible baseline for hybrid irradiance forecasting using the Tunisian climatic dataset and a CNN-WNN-MMHA architecture tailored for renewable energy prediction tasks.
