"""
Unified Training Script for Material Yield Prediction Models

Usage:
    python training/train.py --config yield-2024
    python training/train.py --config yield-2025
    python training/train.py --config yield-2year
    python training/train.py --config yield-5year
    python training/train.py --config yield-6month
    python training/train.py --config yield-multi
    python training/train.py --all  # Train all models

This script reads configuration from training/configs/{model_name}.yaml
and trains the model accordingly.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ONNX conversion
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort


def load_config(config_name: str) -> dict:
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "training" / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {config_name}")
    print(f"  Description: {config.get('description', 'N/A')}")
    print(f"  Years: {config['data'].get('years', 'N/A')}")

    return config


def load_data(config: dict) -> tuple:
    """Load and prepare data based on configuration."""
    years = config['data'].get('years', [])
    min_order_count = config['data'].get('min_order_count', 5)

    print(f"\nLoading data for years: {years}")

    # Import data loading functions
    from src.data_preparation import load_raw_csv_data, prepare_full_dataset

    # Load raw data
    df_261_raw, df_101_raw = load_raw_csv_data(years=years)

    # Prepare full dataset with encoding
    historical_data, encoders = prepare_full_dataset(years=years)

    print(f"  Loaded {len(historical_data):,} records")
    print(f"  Features: {list(historical_data.columns)}")

    return df_261_raw, df_101_raw, historical_data, encoders


def prepare_features(
    historical_data: pd.DataFrame,
    config: dict,
    encoders: dict
) -> tuple:
    """Prepare feature matrix and target variable."""

    categorical_features = config['features']['categorical']
    numerical_features = config['features']['numerical']
    target = config['target']

    # Build feature columns list
    feature_columns = []

    # Add encoded categorical features
    for cat in categorical_features:
        encoded_col = f"{cat}_Encoded"
        if encoded_col in historical_data.columns:
            feature_columns.append(encoded_col)
        elif cat in historical_data.columns:
            # Need to encode
            if cat not in encoders:
                encoders[cat] = LabelEncoder()
                historical_data[encoded_col] = encoders[cat].fit_transform(
                    historical_data[cat].astype(str)
                )
            else:
                historical_data[encoded_col] = encoders[cat].transform(
                    historical_data[cat].astype(str)
                )
            feature_columns.append(encoded_col)

    # Add numerical features
    for num in numerical_features:
        if num in historical_data.columns:
            feature_columns.append(num)

    print(f"\nFeature columns: {feature_columns}")
    print(f"Target: {target}")

    # Create feature matrix
    X = historical_data[feature_columns].fillna(0)
    y = historical_data[target]

    # Remove any infinite values
    mask = ~(np.isinf(X).any(axis=1) | np.isinf(y))
    X = X[mask]
    y = y[mask]

    print(f"  Training samples: {len(X):,}")

    return X, y, feature_columns


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
) -> tuple:
    """Train the model based on configuration."""

    model_config = config['model']
    validation_config = config['validation']

    # Split data
    test_size = validation_config.get('test_size', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"\nTraining {model_config['algorithm']}...")
    print(f"  Train size: {len(X_train):,}")
    print(f"  Test size: {len(X_test):,}")

    # Create model
    hyperparams = model_config.get('hyperparameters', {})
    model = GradientBoostingRegressor(**hyperparams)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'trained_at': datetime.now().isoformat(),
        'config': config['model_name']
    }

    print(f"\nMetrics:")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")

    # Cross-validation
    cv_folds = validation_config.get('cross_validation_folds', 5)
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    metrics['cv_r2_mean'] = float(cv_scores.mean())
    metrics['cv_r2_std'] = float(cv_scores.std())

    print(f"  CV R² (mean ± std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

    return model, metrics, (X_test, y_test, y_pred_test)


def convert_to_onnx(
    model,
    feature_columns: list,
    output_path: Path
) -> None:
    """Convert sklearn model to ONNX format."""

    print(f"\nConverting to ONNX...")

    # Define input type
    n_features = len(feature_columns)
    initial_type = [('X', FloatTensorType([None, n_features]))]

    # Convert
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save
    onnx_path = output_path / "yield_model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"  Saved to: {onnx_path}")

    # Validate
    validate_onnx_conversion(model, onnx_path, n_features)


def validate_onnx_conversion(
    sklearn_model,
    onnx_path: Path,
    n_features: int
) -> None:
    """Validate ONNX model produces identical predictions."""

    # Create test input
    test_input = np.random.rand(1, n_features).astype(np.float32)

    # sklearn prediction
    sklearn_pred = sklearn_model.predict(test_input.astype(np.float64))

    # ONNX prediction
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    onnx_pred = session.run(None, {"X": test_input})[0]

    # Compare
    if np.allclose(sklearn_pred, onnx_pred.flatten(), rtol=1e-4):
        print(f"  ONNX validation passed")
    else:
        print(f"  WARNING: ONNX predictions differ!")
        print(f"    sklearn: {sklearn_pred}")
        print(f"    ONNX: {onnx_pred}")


def compute_precomputed_data(
    df_261_raw: pd.DataFrame,
    df_101_raw: pd.DataFrame,
    config: dict
) -> dict:
    """Compute precomputed data for deployment."""

    print("\nComputing precomputed data...")

    # Import from prepare_deployment
    from scripts.prepare_deployment import (
        compute_material_stats,
        compute_grade_breakdown_data
    )

    min_order_count = config['data'].get('min_order_count', 5)

    material_stats = compute_material_stats(df_261_raw, df_101_raw, min_order_count)
    grade_breakdown = compute_grade_breakdown_data(df_261_raw, df_101_raw, min_order_count)

    precomputed = {
        'material_stats': material_stats,
        'grade_breakdown_data': grade_breakdown,
        'generated_at': datetime.now().isoformat(),
        'config': config['model_name'],
        'years': config['data'].get('years', [])
    }

    print(f"  Materials: {len(material_stats)}")

    return precomputed


def save_artifacts(
    model,
    encoders: dict,
    feature_columns: list,
    metrics: dict,
    precomputed: dict,
    config: dict,
    test_results: tuple = None
) -> None:
    """Save all model artifacts."""
    import json

    output_dir = PROJECT_ROOT / config['output']['model_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving artifacts to: {output_dir}")

    # Save sklearn model
    joblib.dump(model, output_dir / "yield_model.joblib")
    print(f"  Saved: yield_model.joblib")

    # Save encoders
    joblib.dump(encoders, output_dir / "encoders.joblib")
    print(f"  Saved: encoders.joblib")

    # Save feature columns
    joblib.dump(feature_columns, output_dir / "feature_columns.joblib")
    print(f"  Saved: feature_columns.joblib")

    # Save metrics
    joblib.dump(metrics, output_dir / "metrics.joblib")
    print(f"  Saved: metrics.joblib")

    # Save precomputed data
    joblib.dump(precomputed, output_dir / "historical_data_precomputed.joblib", compress=3)
    print(f"  Saved: historical_data_precomputed.joblib")

    # Save test results CSV
    if test_results:
        X_test, y_test, y_pred_test = test_results
        test_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred_test
        })
        test_df.to_csv(output_dir / "test_results.csv", index=False)
        print(f"  Saved: test_results.csv")

    # Save model_metadata.json
    metadata = {
        'trained_years': config['data'].get('years', []),
        'trained_at': datetime.now().isoformat(),
        'best_model': config['model']['algorithm'],
        'test_r2': metrics.get('test_r2', 0),
        'test_rmse': metrics.get('test_rmse', 0),
        'test_mae': metrics.get('test_mae', 0),
        'num_features': len(feature_columns),
        'feature_columns': feature_columns,
        'config_name': config.get('model_name', '')
    }
    with open(output_dir / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: model_metadata.json")

    # Convert to ONNX
    convert_to_onnx(model, feature_columns, output_dir)


def train_single_model(config_name: str) -> dict:
    """Train a single model from config name."""

    print("=" * 70)
    print(f"TRAINING MODEL: {config_name}")
    print("=" * 70)

    # Load config
    config = load_config(config_name)

    # Route KD distribution model to its own training pipeline
    if config.get('model_type') == 'kd_distribution':
        from src.kd_material_prediction import train_kd_distribution_model
        metrics = train_kd_distribution_model(
            years=config['data']['years'],
            min_order_threshold=config['data'].get('min_order_threshold', 20),
            save_dir=str(PROJECT_ROOT / config['output']['model_dir']),
        )
        return metrics

    # Load data
    df_261_raw, df_101_raw, historical_data, encoders = load_data(config)

    # Prepare features
    X, y, feature_columns = prepare_features(historical_data, config, encoders)

    # Train model
    model, metrics, test_results = train_model(X, y, config)

    # Compute precomputed data
    precomputed = compute_precomputed_data(df_261_raw, df_101_raw, config)

    # Save artifacts
    save_artifacts(model, encoders, feature_columns, metrics, precomputed, config, test_results)

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE: {config_name}")
    print(f"{'=' * 70}")

    return metrics


def train_all_models() -> dict:
    """Train all configured models."""

    config_dir = PROJECT_ROOT / "training" / "configs"
    configs = [f.stem for f in config_dir.glob("*.yaml")]

    print(f"Found {len(configs)} model configurations:")
    for c in configs:
        print(f"  - {c}")

    results = {}
    for config_name in configs:
        try:
            metrics = train_single_model(config_name)
            results[config_name] = {'status': 'success', 'metrics': metrics}
        except Exception as e:
            print(f"ERROR training {config_name}: {e}")
            results[config_name] = {'status': 'error', 'error': str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        status = result['status']
        if status == 'success':
            r2 = result['metrics'].get('test_r2', 0)
            print(f"  {name}: SUCCESS (Test R² = {r2:.4f})")
        else:
            print(f"  {name}: ERROR - {result['error']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Material Yield Prediction Models")
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Config name to train (e.g., yield-2024)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Train all configured models'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available configurations'
    )

    args = parser.parse_args()

    if args.list:
        config_dir = PROJECT_ROOT / "training" / "configs"
        configs = [f.stem for f in config_dir.glob("*.yaml")]
        print("Available configurations:")
        for c in configs:
            print(f"  - {c}")
        return

    if args.all:
        train_all_models()
    elif args.config:
        train_single_model(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
