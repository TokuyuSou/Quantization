from typing import Literal, TypedDict

import torch
from torch.optim import Adam

QuantizationAlgorithm = Literal["EasyQuant"]


class EasyQuantConfig(TypedDict):
    learning_rate: float
    num_epochs: int


def _quantize_vector_with_range(
    x: torch.tensor, q_range: float, num_bits: int, detach: bool = False
) -> torch.tensor:
    if detach:
        return (
            torch.clamp(
                torch.round(x / q_range.detach()),
                min=-(2 ** (num_bits - 1)) + 1,
                max=2 ** (num_bits - 1),
            )
            * q_range.detach()
        )
    else:
        return (
            torch.clamp(
                torch.round(x / q_range),
                min=-(2 ** (num_bits - 1)) + 1,
                max=2 ** (num_bits - 1),
            )
            * q_range
        )


def compute_quantization_error(x: torch.tensor, quantized_x: torch.tensor) -> float:
    return torch.sum((quantized_x - x) ** 2).item()


def quantize_tensor_easy_quant(
    x: torch.tensor, num_bits: int, config: EasyQuantConfig, verbose: bool = False
) -> tuple[torch.tensor, list[float], list[float]]:
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

    print(f"Optimizing quantization range using EasyQuant algorithm")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")

    # Initialize the quantization range to cover the entire range of the input tensor
    q_range = (
        ((x.max() - x.min()) / (2**num_bits - 1)).clone().detach().requires_grad_(True)
    )

    if verbose:
        print(f"Initial quantization range: {q_range.item()}")

    # track quantization range and reconstruction error
    q_range_history = []
    reconstruction_error_history = []

    # Define the optimizer
    optimizer = Adam([q_range], lr=learning_rate)

    # Optimize the quantization range using gradient descent
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        q_range_history.append(q_range.item())

        # Dequantize the input tensor using the current quantization range
        quantized_x = _quantize_vector_with_range(x, q_range, num_bits)

        # Compute the quantization error
        reconstruction_error = compute_quantization_error(x, quantized_x)
        reconstruction_error_history.append(reconstruction_error)

        # Compute explicitly the gradient of the quantization error with respect to the quantization range
        grad_of_error = (
            2 * torch.sum((quantized_x - x) * torch.round(x / q_range)).item()
        )

        q_range.grad = torch.tensor(grad_of_error)

        # Compute the gradient of the quantization error
        optimizer.step()

        if verbose:
            print(f"Epoch {epoch + 1}: Quantization range = {q_range.item()}")

        # Ensure q_range remains positive to avoid division by zero
        with torch.no_grad():
            q_range.clamp_(min=1e-6)

    # Dequantize the input tensor using the optimized quantization range
    x_reconstructed = _quantize_vector_with_range(x, q_range, num_bits, detach=True)

    return x_reconstructed, q_range_history, reconstruction_error_history


def quantize_tensor(
    x: torch.tensor,
    num_bits: int,
    quantization_algorithm: QuantizationAlgorithm,
    verbose: bool,
    **kwargs,
) -> tuple[torch.tensor, list[float], list[float]]:
    """Quantize a tensor using a given number of bits

    Args:
        x (Tensor): input tensor to be quantized
        num_bits (int): number of bits to quantize the tensor

    Returns:
        Tensor: dequantized (reconstructed) tensor
        list[float]: history of quantization range values
        list[float]: history of reconstruction error values
    """

    ## Switch between different quantization algorithms
    if quantization_algorithm == "EasyQuant":
        try:
            config = EasyQuantConfig(**kwargs)
            return quantize_tensor_easy_quant(x, num_bits, config, verbose)
        except TypeError:
            raise ValueError(
                f"Invalid configuration parameters for the EasyQuant algorithm: {kwargs}"
            )
    else:
        raise ValueError(f"Unknown quantization algorithm: {quantization_algorithm}")
