"""Load encoders and transform raw input to model features."""
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any


class Preprocessor:
    """Handles encoding of raw input values to model features."""

    def __init__(self, model_dir: str):
        """
        Initialize preprocessor with encoders and feature columns.

        Args:
            model_dir: Path to directory containing encoders.joblib and feature_columns.joblib
        """
        self.model_dir = Path(model_dir)
        self.encoders = joblib.load(self.model_dir / "encoders.joblib")
        self.feature_columns = joblib.load(self.model_dir / "feature_columns.joblib")

        # Map request fields to encoder names
        self._field_mapping = {
            'input_material': 'Input_Material',
            'input_specie': 'Input_Specie',
            'input_grade': 'Input_Grade',
            'input_plant': 'Input_Plant',
        }

    def encode(self, request) -> np.ndarray:
        """
        Transform raw request values to encoded feature array.

        Args:
            request: PredictionRequest object with raw values

        Returns:
            numpy array of shape (1, n_features) ready for model inference
        """
        features = {}

        # Encode categorical fields
        features['Input_Material_Encoded'] = self._safe_encode('Input_Material', request.input_material)
        features['Input_Specie_Encoded'] = self._safe_encode('Input_Specie', request.input_specie)
        features['Input_Grade_Encoded'] = self._safe_encode('Input_Grade', request.input_grade)
        features['Input_Plant_Encoded'] = self._safe_encode('Input_Plant', request.input_plant)

        # Numeric fields (no encoding needed)
        features['Input_Thickness'] = float(request.input_thickness)
        features['Input_Length'] = float(request.input_length)
        features['Input_Width'] = float(request.input_width)
        features['Total_Input_BF'] = float(request.total_input_bf)

        # Build array in correct feature order
        feature_array = [features[col] for col in self.feature_columns]
        return np.array([feature_array], dtype=np.float32)

    def _safe_encode(self, field: str, value: str) -> int:
        """
        Encode categorical value safely.

        Args:
            field: Encoder field name (e.g., 'Input_Material')
            value: Raw string value to encode

        Returns:
            Encoded integer value

        Raises:
            ValueError: If field doesn't exist or value is unknown
        """
        encoder = self.encoders.get(field)
        if encoder is None:
            raise ValueError(f"No encoder found for field: {field}")

        if value in encoder.classes_:
            return int(encoder.transform([value])[0])
        else:
            valid_values = list(encoder.classes_)[:10]  # Show first 10 for brevity
            more = f" (and {len(encoder.classes_) - 10} more)" if len(encoder.classes_) > 10 else ""
            raise ValueError(
                f"Unknown {field} value: '{value}'. "
                f"Valid values include: {valid_values}{more}"
            )

    def get_valid_values(self, field: str) -> list:
        """Get list of valid values for a categorical field."""
        encoder = self.encoders.get(field)
        if encoder is None:
            return []
        return list(encoder.classes_)
