from typing import TypedDict

import torch
from torch import Tensor
from torch.optim import Adam

from core.common.utils import (
    compute_quantization_error,
    quantize_vector_with_range,
    quantize_and_reconstruct_vector_with_range,
)
from core.models.base_model import QuantizationMethod


class EasyQuantConfig(TypedDict):
    learning_rate: float
    num_epochs: int


class EasyQuantReconstructionSettings(TypedDict):
    scale: float


class EasyQuant(QuantizationMethod[EasyQuantConfig, EasyQuantReconstructionSettings]):
    def __init__(
        self,
        num_bits: int,
        config: EasyQuantConfig,
        quantized_dtype: torch.dtype | None = None,
        verbose: bool = False,
    ):
        """
        EasyQuant algorithm proposed by Tang et al. (2024)

        Parameters:
        - num_bits (int): Number of bits used to quantize the data.
        - config (ConfigType): A dictionary containing configuration parameters specific to the quantization method.
        """
        super().__init__(num_bits, config, quantized_dtype, verbose)

        self.learning_rate = config["learning_rate"]
        self.num_epochs = config["num_epochs"]

    def optimize(self, data: Tensor) -> tuple[float, list[float], list[float]]:
        """Optimize the quantization range using the EasyQuant algorithm

        Args:
            data (Tensor): The data to be quantized.

        Returns:
            tuple[float, list[float], list[float]]: Tuple containing the optimized quantization range, history of quantization range values, and history of reconstruction error values
        """

        ## Optimize the quantization range using gradient descent and Adam optimizer
        if self.verbose:
            print(f"Optimizing quantization range using EasyQuant algorithm")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Number of epochs: {self.num_epochs}")

        data = data.detach()

        # Initialize the quantization range to cover the entire range of the input tensor
        q_range = (
            (torch.max(torch.abs(data)) / (2 ** (self.num_bits - 1)))
            .clone()
            .detach()
            .requires_grad_(True)
        )

        if self.verbose:
            print(f"Initial quantization range: {q_range.item()}")

        # track quantization range and reconstruction error
        q_range_history_tensor = torch.empty(self.num_epochs, device=data.device)
        reconstruction_error_history_tensor = torch.empty(
            self.num_epochs, device=data.device
        )

        # Define the optimizer
        optimizer = Adam([q_range], lr=self.learning_rate)

        # Optimize the quantization range using gradient descent
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            q_range_history_tensor[epoch] = q_range

            # Dequantize the input tensor using the current quantization range
            quantized_data = quantize_and_reconstruct_vector_with_range(
                data, q_range, self.num_bits
            )

            # Compute the quantization error
            reconstruction_error = compute_quantization_error(data, quantized_data)
            reconstruction_error_history_tensor[epoch] = reconstruction_error

            # Compute the gradient of the reconstruction error
            # Using automatic differentiation has been confirmed to generate the same result as using Equation (2) in the paper
            reconstruction_error.backward()

            # Compute the gradient of the quantization error
            optimizer.step()

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}: Quantization range = {q_range.item()}, Reconstruction error = {reconstruction_error}"
                )

            # Ensure q_range remains positive to avoid division by zero
            with torch.no_grad():
                q_range.clamp_(min=1e-6)

        # Convert the history tensors to lists
        q_range_history = q_range_history_tensor.cpu().tolist()
        reconstruction_error_history = (
            reconstruction_error_history_tensor.cpu().tolist()
        )

        return q_range.item(), q_range_history, reconstruction_error_history

    def quantize(self, data: Tensor, **kwargs) -> tuple[float, Tensor]:
        """Quantize the input tensor and return the low-bit representation

        Args:
            data (Tensor): The data to be quantized.

        Returns:
            tuple[float, Tensor]: Tuple containing the final quantization range and the quantized data.
        """
        # Raise error if the quantized data type is not specified
        if self.quantized_dtype is None:
            raise ValueError(
                "The quantized data type must be specified to store the quantized weights in low-bit format"
            )

        # Optimize the quantization range using the EasyQuant algorithm
        q_range, _, _ = self.optimize(data)

        # Quantize the input tensor using the optimized quantization range
        quantized_data = quantize_vector_with_range(
            data, q_range, self.num_bits, self.quantized_dtype
        )

        # Save the reconstruction settings
        self.reconstruction_settings = EasyQuantReconstructionSettings(scale=q_range)

        return q_range, quantized_data

    def quantize_and_reconstruct(
        self, data: Tensor, **kwargs
    ) -> tuple[float, Tensor, list[float], list[float]]:
        """Quantize and reconstruct the input tensor

        Args:
            data (Tensor): The data to be quantized and reconstructed.

        Returns:
            tuple[float, Tensor, list[float], list[float]]: Tuple containing the final quantization range, the reconstructed data, history of quantization range values, and history of reconstruction error values
        """

        # Optimize the quantization range using the EasyQuant algorithm
        q_range, q_range_history, reconstruction_error_history = self.optimize(data)

        # Save the reconstruction settings
        self.reconstruction_settings = EasyQuantReconstructionSettings(scale=q_range)

        # Quantize and reconstruct the input tensor using the optimized quantization range
        reconstructed_data = quantize_and_reconstruct_vector_with_range(
            data, q_range, self.num_bits, detach=True
        )

        return (
            q_range,
            reconstructed_data,
            q_range_history,
            reconstruction_error_history,
        )

    def reconstruct(self, data: Tensor, **kwargs) -> Tensor:
        """Reconstruct the quantized data back to the original data space

        Args:
            data (Tensor): The quantized data to be reconstructed.

        Returns:
            Tensor: The reconstructed data.
        """
        if self.reconstruction_settings is None:
            raise ValueError(
                "Reconstruction settings not found. Please quantize the data first."
            )

        # Get the quantization range from the reconstruction settings
        scale = self.reconstruction_settings["scale"]

        # Reconstruct the quantized data using the quantization range
        reconstructed_data = quantize_and_reconstruct_vector_with_range(
            data, scale, self.num_bits
        )

        return reconstructed_data
