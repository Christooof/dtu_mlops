import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytest

import sys
sys.path.append('..')  # Adds the parent directory to the system path

from vae_mnist_working import Model,encoder,decoder, DEVICE  # Now you can import the module

x_dim=1, 28, 28
batch_size = 100
model = Model(encoder=encoder, decoder=decoder).to(DEVICE)
# Define the test function
def test_model_output_shape():
    # Define your model, load data, etc. (similar to your model setup code)
    # ...

    # Create a sample input tensor with shape X
    sample_input = torch.randn(batch_size, 1, 28, 28)  # Replace with desired input shape X

    # Forward pass to obtain the output from the model
    with torch.no_grad():
        output, _, _ = model(sample_input.to(DEVICE))

    # Check if the output shape matches the expected shape Y
    expected_output_shape = (batch_size, 1, 28, 28)  # Replace with desired output shape Y
    assert output.shape == expected_output_shape

# Run the test function
if __name__ == "__main__":
    pytest.main([__file__])
