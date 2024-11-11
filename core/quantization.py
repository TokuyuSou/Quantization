from typing import Literal

import torch
import torch.mps
from torch import Tensor

from core.models.easy_quant import EasyQuantConfig, quantize_tensor_easy_quant

QuantizationAlgorithm = Literal["EasyQuant"]


def mask_outliers(tensor: Tensor, n_sigma: int = 3) -> Tensor:
    """Mask outliers in a tensor based on the specified number of standard deviations
    Args:
        tensor (Tensor): input tensor
        n_sigma (int, optional): number of standard deviations to consider as outliers. Defaults to 3.

    Returns:
        Tensor: boolean mask of the same shape as the input tensor (True for outliers, False otherwise)
    """
    mean = tensor.mean()
    std = tensor.std()
    outlier_mask = torch.abs(tensor - mean) >= (n_sigma * std)
    return outlier_mask


def quantize_layer(
    W: Tensor,
    num_bits: int,
    quantization_algorithm: QuantizationAlgorithm,
    retain_outliers: bool,
    outlier_threshold: int = 3,
    verbose: bool = False,
    **kwargs,
) -> tuple[Tensor, Tensor | None]:
    """Quantize a layer of a neural network (all weights in the layer)

    Args:
        W (Tensor): weight tensor to be quantized (2D tensor)
        num_bits (int): number of bits to quantize the tensor
        quantization_algorithm (QuantizationAlgorithm): quantization algorithm to use
        retain_outliers (bool): whether to retain outliers during quantization
        outlier_threshold (int, optional): threshold for identifying outliers. Defaults to 3.
        verbose (bool, optional): whether to print debug information. Defaults to False.

    Returns:
        Tensor: quantized tensor of the normal weights (same shape as the input tensor, with outliers set to zero)
        Tensor | None: retained outlier weights (same shape as the input tensor, sparse matrix stored in CSR format)
    """
    # Check if mps is available
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    ## Outlier detection and masking
    if retain_outliers:
        if W.dim() != 2:
            raise ValueError(
                f"Outlier retention is only supported for 2D tensors (weight matrices): {W.shape}"
            )
        # Identify outliers in the input tensor
        outlier_mask = mask_outliers(W, outlier_threshold)

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

    # Allocate memory for the quantized weights
    quantized_normal_weights = torch.empty_like(normal_weights, device=device)

    ## Quantization per channel
    args = [
        (
            normal_weights[i].to(device),
            num_bits,
            quantization_algorithm,
            verbose,
        )
        for i in range(W.size(0))
    ]

    for i in range(W.size(0)):
        quantized_normal_weights[i] = quantize_single_tensor(*args[i], **kwargs)

    return quantized_normal_weights, outlier_weights


def quantize_single_tensor(
    x: Tensor,
    num_bits: int,
    quantization_algorithm: QuantizationAlgorithm,
    verbose: bool = False,
    **kwargs,
) -> Tensor:
    """Quantize a tensor using a given number of bits

    Args:
        x (Tensor): input tensor to be quantized
        num_bits (int): number of bits to quantize the tensor
        quantization_algorithm (QuantizationAlgorithm): quantization algorithm to use
        per_channel (bool): whether to quantize per channel (for 2D tensors) or per tensor

    Returns:
        Tensor: quantized tensor
    """

    ## Switch between different quantization algorithms
    if quantization_algorithm == "EasyQuant":
        try:
            config = EasyQuantConfig(**kwargs)
            quantized_x, _, _ = quantize_tensor_easy_quant(x, num_bits, config, verbose)
            return quantized_x
        except TypeError:
            raise ValueError(
                f"Invalid configuration parameters for the EasyQuant algorithm: {kwargs}"
            )
    else:
        raise ValueError(f"Unknown quantization algorithm: {quantization_algorithm}")
