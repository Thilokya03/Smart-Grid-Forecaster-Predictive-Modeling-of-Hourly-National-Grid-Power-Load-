# Predictive Model 

This directory contains the machine learning pipeline for predicting the hourly national grid power load. The forecasting model is built using XGBoost.

## Pipeline Scripts

The modeling process is split into three sequential scripts that should be run in order:

1. **`01_features.py`**: Handles feature engineering and data preprocessing. It takes raw data and generates features for the model, saving intermediate outputs like `features.pkl`.
2. **`02_train.py`**: Performs model training and hyperparameter tuning to find the optimal XGBoost model parameters. It saves the best configuration to `best_params.json`.
3. **`03_final_train_eval.py`**: Trains the final model using the best parameters, evaluates its performance, and saves the final model artifacts.

## Output Artifacts

Running the pipeline generates several files:

- **Model Artifacts**: `xgb_demand_model.json` and `xgb_demand_model.ubj` (The saved XGBoost model).
- **Data Files**: `master_training_data.csv` (combined training data) and `features.pkl`.
- **Metrics & Metadata**: `metrics.json` (evaluation metrics), `feature_columns.json` (ordered list of features used), and `best_params.json`.
- **Analysis**: `feature_importance.csv` and `actual_vs_predicted.png` for visualizing model behavior and performance.


## How to Run

Run the XGBoost pipeline scripts in the following order:

```bash
python 01_features.py
python 02_train.py
python 03_final_train_eval.py