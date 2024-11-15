import time
from multiprocessing import Pool, cpu_count
from typing import TypedDict

import torch
from torch import Tensor
from torch.optim import Adam

from core.models.base_model import LayerQuantization, QuantizationMethod
from core.utils.quantization_helpers import (
    compute_quantization_error, quantize_and_reconstruct_vector_with_range,
    quantize_vector_with_range)


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

        # Optimize the quantization range using gradient descent and Adam optimizer
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

    def quantize(self, data: Tensor, **kwargs) -> Tensor:
        """Quantize the input tensor and return the low-bit representation

        Args:
            data (Tensor): The data to be quantized.

        Returns:
            Tensor: The quantized data (low-bit representation).
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

        return quantized_data

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


def _quantize_channel(
    args: tuple[int, Tensor, int, torch.dtype, bool, EasyQuantConfig]
) -> tuple[float, Tensor]:
    """Quantizes a single channel of a weight matrix using the EasyQuant algorithm

    Args:
        args (tuple[int, Tensor, int, torch.dtype, bool, EasyQuantConfig]): Tuple containing the channel index, the weight matrix, the number of bits, the quantized data type, a flag to print verbose output, and the configuration dictionary.

    Returns:
        tuple[float, Tensor]: Tuple containing the quantization range and the quantized channel.
    """
    i, W, num_bits, quantized_dtype, verbose, config = args

    # Extract the i-th channel of the weight matrix
    channel = W[i]

    # Quantize the channel using the EasyQuant algorithm
    quantization_executor = EasyQuant(num_bits, config, quantized_dtype, verbose)

    quantized_channel = quantization_executor.quantize(channel)

    return quantization_executor.reconstruction_settings["scale"], quantized_channel


class EasyQuantLayerQuantization(
    LayerQuantization[EasyQuantConfig, EasyQuantReconstructionSettings]
):
    def __init__(
        self,
        num_bits: int,
        config: EasyQuantConfig,
        quantized_dtype: torch.dtype | None = None,
        retain_outliers: bool = False,
        outlier_threshold: int = 3,
        num_workers: int = cpu_count() // 2,
        verbose: bool = False,
    ):
        """
        Initializes the layer quantization method with a configuration dictionary.

        Parameters:
        - num_bits (int): Number of bits used to quantize the data.
        - config (EasyQuantConfig): A dictionary containing configuration parameters specific to the quantization method.
        - quantized_dtype (torch.dtype | None): The data type used to store the quantized weights.
        - retain_outliers (bool): Whether to retain outliers in the quantized weights.
        - outlier_threshold (int): The threshold used to detect outliers.
        - num_workers (int): The number of workers used for parallel processing.
        - verbose (bool): Whether to print detailed information during quantization.
        """
        super().__init__(
            num_bits,
            config,
            quantized_dtype,
            retain_outliers,
            outlier_threshold,
            num_workers,
            verbose,
        )

        self.device = "cpu"  # MPS backend does not support sparse tensors

        self.q_ranges = None
        self.normal_weights = None
        self.outlier_weights = None

    def quantize_layer(self, W: Tensor, **kwargs) -> None:
        """Quantizes the weights of a layer. Each row of the weight matrix is quantized independently. (per-channel)
        The quantized weights will be stored as a member variable of the class instance.

        Args:
            W (Tensor): The weights to be quantized.

        """

        ## Outlier detection and masking
        start_time = time.time()

        W = W.detach().to(self.device)

        if self.retain_outliers:
            if W.dim() != 2:
                raise ValueError(
                    f"Outlier retention is only supported for 2D tensors (weight matrices): {W.shape}"
                )
            # Identify outliers in the input tensor
            outlier_mask = self.mask_outliers(W)

            # Extract outliers and normal weights
            outlier_weights = W.clone()
            outlier_weights[~outlier_mask] = 0
            normal_weights = W.clone()
            normal_weights[outlier_mask] = 0

            # Store outlier weights in CSR format for efficient processing
            outlier_weights = outlier_weights.to_sparse_csr()
        else:
            normal_weights = W
            outlier_weights = None

        # Allocate memory for the quantized weights and quantization ranges
        quantized_normal_weights = torch.empty_like(
            normal_weights, device=self.device
        ).to(self.quantized_dtype)

        self.q_ranges = torch.empty(W.size(0), device=self.device)

        ## Quantization per channel
        if self.verbose:
            print(f"Quantizing {W.size(0)} output channels")

        # Prepare arguments for parallel processing
        args = [
            (
                i,
                normal_weights,
                self.num_bits,
                self.quantized_dtype,
                self.verbose,
                self.config,
            )
            for i in range(W.size(0))
        ]

        # Using Pool to parallelize the quantization of each channel
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(_quantize_channel, args)

        # Unpacking results
        for i, (range_, quantized_weights) in enumerate(results):
            self.q_ranges[i] = range_
            quantized_normal_weights[i] = quantized_weights

        self.q_ranges = self.q_ranges.view(-1, 1)

        if self.verbose:
            print(f"Quantization completed in {time.time() - start_time:.2f} seconds")

        self.normal_weights = quantized_normal_weights
        self.outlier_weights = outlier_weights

    def reconstruct_layer(self, **kwargs) -> Tensor:
        """Reconstructs the quantized data back to the original data space.

        Args:
            data (Tensor): The quantized data to be reconstructed.

        Returns:
            Tensor: The reconstructed data.
        """

        if (
            self.q_ranges is None
            or self.normal_weights is None
            or (self.retain_outliers and self.outlier_weights is None)
        ):
            raise ValueError(
                "Quantization ranges or weights not found. Please quantize the data first."
            )

        # Reconstruct the quantized data using the quantization ranges
        weights_full_precision = self.normal_weights * self.q_ranges

        if self.retain_outliers:
            weights_full_precision += self.outlier_weights.to_dense()

        return weights_full_precision
