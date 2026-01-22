# Production Planning - Material Yield Prediction System

A machine learning system for predicting material yield in manufacturing operations using SAP data. This tool helps production planners estimate how much finished product (output) can be produced from raw materials (input).

## Overview

This system uses historical SAP manufacturing data to predict yield percentages when converting raw materials into finished goods. It follows SAP's standard movement type logic:

- **261 (Goods Issue)** = INPUT - Raw materials consumed from stock
- **101 (Goods Receipt)** = OUTPUT - Finished goods received into stock
- **Yield** = Output / Input (expressed as a percentage)

## Features

- Multi-model comparison (Random Forest, Gradient Boosting, XGBoost, Ridge)
- Support for multiple input materials per prediction
- Confidence intervals on predictions
- Forward planning (input → expected output)
- Reverse planning (required output → needed input)
- Anomaly detection for unusual yield values
- Interactive Streamlit web interface

## Project Structure

```
Production_Planning/
├── train_model.py       # Main training script
├── data_preparation.py  # Data loading, cleaning, and feature engineering
├── model_training.py    # ML model training and prediction classes
├── prediction_utils.py  # Utility functions for predictions
├── app.py               # Streamlit web application
├── 261.csv              # Input data (Goods Issue records)
├── 101.csv              # Output data (Goods Receipt records)
└── README.md            # This file
```

## Installation

1. Create a conda environment (recommended):
   ```bash
   conda create -n production_planning python=3.10
   conda activate production_planning
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn joblib streamlit plotly xgboost
   ```

## Usage

### 1. Train the Model

Place your SAP data files (`261.csv` and `101.csv`) in the project directory, then run:

```bash
python train_model.py
```

**Command line options:**
```bash
python train_model.py --input261 path/to/261.csv --output101 path/to/101.csv --output-dir ./models
```

This will:
- Load and clean the data
- Join input/output records by Manufacturing Order
- Calculate yield percentages
- Train multiple regression models
- Save model artifacts for the Streamlit app

### 2. Run the Web Application

```bash
streamlit run app.py
```

The app allows you to:
- Select input materials, species, grades, and quantities
- Run yield predictions
- View predicted output materials and quantities
- Compare model performance
- Download results as CSV

## Input Data Requirements

### 261.csv (Goods Issue - Input Materials)
Required columns:
- `MANUFACTURINGORDER` - Order identifier (join key)
- `MATERIAL` - Material code
- `MATERIALSPECIE` - Wood species
- `TALLYGRADE` - Material grade
- `PLANT` - Plant code
- `MATERIALTHICKNESS`, `TALLYLENGTH`, `TALLYWIDTH` - Dimensions
- `BFIN` - Board feet (or calculated from dimensions)

### 101.csv (Goods Receipt - Output Materials)
Required columns:
- `MANUFACTURINGORDER` - Order identifier (join key)
- `MATERIAL` - Output material code
- `MATERIALSPECIE` - Wood species
- `TALLYGRADE` - Material grade
- `MATERIALTHICKNESS`, `TALLYLENGTH`, `TALLYWIDTH` - Dimensions
- `BFIN` - Board feet for output

## Model Outputs

After training, the following files are created:

| File | Description |
|------|-------------|
| `yield_model.joblib` | Best performing trained model |
| `all_models.joblib` | All trained models for comparison |
| `encoders.joblib` | Label encoders for categorical features |
| `feature_columns.joblib` | List of feature columns used |
| `metrics.joblib` | Model performance metrics |
| `scalers.joblib` | Feature scalers |
| `test_results.csv` | Actual vs predicted on test set |
| `feature_importance.csv` | Feature importance rankings |

## Model Features

The models use these features for prediction:
- Input Material (encoded)
- Input Species (encoded)
- Input Grade (encoded)
- Input Plant (encoded)
- Input Thickness
- Input Length
- Input Width
- Total Input Board Feet

## Performance Metrics

Models are evaluated using:
- **R-squared (R2)** - Explained variance
- **MAE** - Mean Absolute Error (in yield %)
- **RMSE** - Root Mean Squared Error (in yield %)
- **MAPE** - Mean Absolute Percentage Error
- **Cross-validation** - 5-fold CV for robust evaluation

## Example Output

```
======================================================================
MATERIAL YIELD PREDICTION - MODEL TRAINING
======================================================================

Configuration:
  Input (261 - Goods Issue): 261.csv
  Output (101 - Goods Receipt): 101.csv

Data Summary:
  Records: 50,000
  Columns: 15

Yield Statistics:
  Mean: 78.45%
  Std:  8.23%

Model Comparison:
      Model  Train R²  Test R²   MAE   RMSE
    XGBoost    0.9234   0.8912  3.21   4.56
RandomForest   0.9456   0.8845  3.35   4.72
```

## SAP Integration Notes

- Input and output materials are **different** - never match by material code
- Join records **only** on `MANUFACTURINGORDER`
- Board feet formula: `(Thickness x Width x Length) / 144`
- Typical lumber yields range from 50% to 95% depending on grade and species

## License

Internal use only.
