"""ONNX Runtime inference wrapper."""
import onnxruntime as ort
import numpy as np
from pathlib import Path


class ONNXModelService:
    """Loads ONNX model once and provides inference method."""

    def __init__(self, model_path: str):
        """
        Initialize ONNX model service.

        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Configure session options for performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        # Load model ONCE at initialization
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        # Cache input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Log model info
        print(f"ONNX model loaded: {self.model_path}")
        print(f"  Input: {self.input_name} (shape: {self.session.get_inputs()[0].shape})")
        print(f"  Output: {self.output_name}")

    def predict(self, features: np.ndarray) -> float:
        """
        Run inference and return yield percentage.

        Args:
            features: numpy array of shape (1, n_features)

        Returns:
            Predicted yield percentage as float
        """
        result = self.session.run(None, {self.input_name: features})

        # Handle different output shapes
        output = result[0]
        if isinstance(output, np.ndarray):
            if output.ndim == 0:
                return float(output)
            elif output.ndim == 1:
                return float(output[0])
            else:
                return float(output[0][0])
        return float(output)

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Run batch inference.

        Args:
            features: numpy array of shape (batch_size, n_features)

        Returns:
            Array of predicted yield percentages
        """
        result = self.session.run(None, {self.input_name: features})
        return result[0].flatten()
