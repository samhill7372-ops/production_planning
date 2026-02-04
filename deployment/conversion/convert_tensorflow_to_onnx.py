"""Reference: Convert TensorFlow/Keras model to ONNX.

This is a reference implementation showing how to convert TensorFlow models.
Not used in this project (which uses sklearn), but provided for future reference.
"""


def convert_tensorflow_model(model_path: str, output_path: str):
    """
    Convert a TensorFlow/Keras model to ONNX format.

    Args:
        model_path: Path to the saved Keras model (.h5 or SavedModel directory)
        output_path: Path to save the ONNX model

    Example:
        convert_tensorflow_model('my_model.h5', 'model.onnx')
    """
    import tf2onnx
    import tensorflow as tf

    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Get input specification
    spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)

    # Convert to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=17,
        output_path=output_path
    )

    print(f"TensorFlow model converted and saved to {output_path}")


# Example usage (uncomment to use):
# if __name__ == "__main__":
#     convert_tensorflow_model('keras_model.h5', 'tensorflow_model.onnx')
