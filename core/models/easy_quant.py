from re import I
from typing import TypedDict

import torch
from torch import Tensor
from torch.optim import Adam

from core.common.utils import compute_quantization_error, quantize_vector_with_range


class EasyQuantConfig(TypedDict):
    learning_rate: float
    num_epochs: int


def quantize_tensor_easy_quant(
    x: Tensor, num_bits: int, config: EasyQuantConfig, verbose: bool = False
) -> tuple[Tensor, list[float], list[float]]:
    """Quantize a tensor using the EasyQuant algorithm proposed by Tang et al. (2024)

    Args:
        x (Tensor): input tensor to be quantized
        num_bits (int): number of bits to quantize the tensor
        config (dict): configuration parameters for the EasyQuant algorithm
        verbose (bool): whether to print debug information

    Returns:
        Tensor: dequantized (reconstructed) tensor
        list[float]: history of quantization range values
        list[float]: history of reconstruction error values
    """

    ## Optimize the quantization range using gradient descent and Adam optimizer
    # Extract configuration parameters
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]

    if verbose:
        print(f"Optimizing quantization range using EasyQuant algorithm")
        print(f"Learning rate: {learning_rate}")
        print(f"Number of epochs: {num_epochs}")

    # Initialize the quantization range to cover the entire range of the input tensor
    q_range = (
        (torch.max(torch.abs(x)) / (2 ** (num_bits - 1)))
        .clone()
        .detach()
        .requires_grad_(True)
    )

    if verbose:
        print(f"Initial quantization range: {q_range.item()}")

    # track quantization range and reconstruction error
    q_range_history_tensor = torch.empty(num_epochs, device=x.device)
    reconstruction_error_history_tensor = torch.empty(num_epochs, device=x.device)

    # Define the optimizer
    optimizer = Adam([q_range], lr=learning_rate)

    # Optimize the quantization range using gradient descent
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        q_range_history_tensor[epoch] = q_range

        # Dequantize the input tensor using the current quantization range
        quantized_x = quantize_vector_with_range(x, q_range, num_bits)

        # Compute the quantization error
        reconstruction_error = compute_quantization_error(x, quantized_x)
        reconstruction_error_history_tensor[epoch] = reconstruction_error

        # Compute the gradient of the reconstruction error
        # Using automatic differentiation has been confirmed to generate the same result as using Equation (2) in the paper
        reconstruction_error.backward()

        # Compute the gradient of the quantization error
        optimizer.step()

        if verbose:
            print(
                f"Epoch {epoch + 1}: Quantization range = {q_range.item()}, Reconstruction error = {reconstruction_error}"
            )

        # Ensure q_range remains positive to avoid division by zero
        with torch.no_grad():
            q_range.clamp_(min=1e-6)

    # Convert the history tensors to lists
    q_range_history = q_range_history_tensor.cpu().tolist()
    reconstruction_error_history = reconstruction_error_history_tensor.cpu().tolist()

    # Quantize the input tensor using the optimized quantization range
    x_reconstructed = quantize_vector_with_range(x, q_range, num_bits, detach=True)

    return x_reconstructed, q_range_history, reconstruction_error_history
