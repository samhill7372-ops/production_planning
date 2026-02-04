"""Reference: Convert PyTorch model to ONNX.

This is a reference implementation showing how to convert PyTorch models.
Not used in this project (which uses sklearn), but provided for future reference.
"""
import torch
import torch.onnx


def convert_pytorch_model(model, input_shape: tuple, output_path: str):
    """
    Convert a PyTorch model to ONNX format.

    Args:
        model: PyTorch model (nn.Module)
        input_shape: Input tensor shape excluding batch dimension (e.g., (8,) for 8 features)
        output_path: Path to save the ONNX model

    Example:
        model = MyModel()
        model.load_state_dict(torch.load('model.pt'))
        convert_pytorch_model(model, (8,), 'model.onnx')
    """
    model.eval()

    # Create dummy input with batch dimension
    dummy_input = torch.randn(1, *input_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"PyTorch model converted and saved to {output_path}")


# Example usage (uncomment to use):
# if __name__ == "__main__":
#     import torch.nn as nn
#
#     class SimpleModel(nn.Module):
#         def __init__(self, n_features):
#             super().__init__()
#             self.fc1 = nn.Linear(n_features, 64)
#             self.fc2 = nn.Linear(64, 1)
#
#         def forward(self, x):
#             x = torch.relu(self.fc1(x))
#             return self.fc2(x)
#
#     model = SimpleModel(n_features=8)
#     convert_pytorch_model(model, (8,), 'pytorch_model.onnx')
