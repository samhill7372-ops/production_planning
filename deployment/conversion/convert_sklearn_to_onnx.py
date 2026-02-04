"""Convert sklearn GradientBoostingRegressor to ONNX format."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
from pathlib import Path


def convert_model(year: str = "2024"):
    """Convert sklearn model to ONNX format."""
    base_dir = Path(__file__).parent.parent.parent

    # Load original model
    model_path = base_dir / f"models/{year}/yield_model.joblib"
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Load feature columns to get count
    feature_cols = joblib.load(base_dir / f"models/{year}/feature_columns.joblib")
    n_features = len(feature_cols)
    print(f"Model has {n_features} features: {feature_cols}")

    # Define input shape
    initial_type = [('X', FloatTensorType([None, n_features]))]

    # Convert to ONNX
    print("Converting to ONNX...")
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Create output directory
    output_dir = base_dir / f"deployment/models/{year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ONNX model
    output_path = output_dir / "yield_model.onnx"
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX model to {output_path}")

    # Copy encoders and feature_columns to deployment folder
    import shutil
    shutil.copy(base_dir / f"models/{year}/encoders.joblib", output_dir / "encoders.joblib")
    shutil.copy(base_dir / f"models/{year}/feature_columns.joblib", output_dir / "feature_columns.joblib")
    print(f"Copied encoders and feature_columns to {output_dir}")

    # Validate: compare predictions
    validate_conversion(model, str(output_path), year)

    return str(output_path)


def validate_conversion(original_model, onnx_path: str, year: str):
    """Ensure ONNX model produces identical predictions."""
    base_dir = Path(__file__).parent.parent.parent

    # Load feature columns
    feature_cols = joblib.load(base_dir / f"models/{year}/feature_columns.joblib")
    n_features = len(feature_cols)

    # Create sample input (random values for each feature)
    test_input = np.array([[1.0, 2.0, 3.0, 4.0, 4.0, 96.0, 8.0, 1000.0]], dtype=np.float32)

    # Original prediction
    sklearn_pred = original_model.predict(test_input.astype(np.float64))

    # ONNX prediction
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_pred = session.run(None, {"X": test_input})[0]

    # Compare
    sklearn_val = float(sklearn_pred[0])
    onnx_val = float(onnx_pred[0][0]) if len(onnx_pred[0].shape) > 0 else float(onnx_pred[0])

    if np.allclose(sklearn_val, onnx_val, rtol=1e-5):
        print(f"Validation PASSED: sklearn={sklearn_val:.4f}, onnx={onnx_val:.4f}")
    else:
        raise ValueError(f"Predictions differ! sklearn={sklearn_val}, onnx={onnx_val}")


if __name__ == "__main__":
    print("=" * 50)
    print("Converting 2024 model...")
    print("=" * 50)
    convert_model("2024")

    print("\n" + "=" * 50)
    print("Converting 2025 model...")
    print("=" * 50)
    convert_model("2025")

    print("\n" + "=" * 50)
    print("All models converted successfully!")
    print("=" * 50)
