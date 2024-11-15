from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from typing import Generic, TypeVar

import torch
from torch import Tensor

from core.utils.quantization_helpers import get_best_int_type

ConfigType = TypeVar("ConfigType", bound=dict[str, any])
ReconstructionSettings = TypeVar("ReconstructionSettings")


class QuantizationMethod(ABC, Generic[ConfigType, ReconstructionSettings]):
    def __init__(
        self,
        num_bits: int,
        config: ConfigType,
        quantized_dtype: torch.dtype | None = None,
        verbose: bool = False,
    ):
        """
        Initializes the quantization method with a configuration dictionary.

        Parameters:
        - num_bits (int): Number of bits used to quantize the data.
        - config (ConfigType): A dictionary containing configuration parameters specific to the quantization method.
        - quantized_dtype (torch.dtype): Data type of the quantized weights. (Used for saving the weights in low-bit format)
        - verbose (bool): Whether to print verbose output during optimization.
        """
        self.quantized_dtype = quantized_dtype
        self.num_bits = num_bits
        self.quantized_dtype = quantized_dtype
        self.verbose = verbose
        self.reconstruction_settings: ReconstructionSettings | None = None

    @abstractmethod
    def quantize(self, data: Tensor, **kwargs) -> Tensor:
        """Quantizes the input data and returns the low-bit representation.
        Every element in the input tensor is used as a separate (1-D) data point.

        Args:
            data (Tensor): The data to be quantized.

        Returns:
            Tensor: The quantized data (low-bit representation).

        """

    @abstractmethod
    def reconstruct(self, data: Tensor, **kwargs) -> Tensor:
        """Reconstructs the quantized data back to the original data space.

        Args:
            data (Tensor): The quantized data to be reconstructed.

        Returns:
            Tensor: The reconstructed data.
        """

    @abstractmethod
    def quantize_and_reconstruct(self, data: Tensor, **kwargs) -> tuple[Tensor, any]:
        """Quantizes and reconstructs the input data.

        Args:
            data (Tensor): The data to be quantized and reconstructed.

        Returns:
            tuple[Tensor, any]: Tuple containing the quantized data and any additional information
        """


class LayerQuantization(ABC, Generic[ConfigType, ReconstructionSettings]):
    def __init__(
        self,
        num_bits: int,
        config: ConfigType,
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
        - config (ConfigType): A dictionary containing configuration parameters specific to the quantization method.
        - quantized_dtype (torch.dtype): Data type of the quantized weights. (Used for saving the weights in low-bit format)
        - retain_outliers (bool): Whether to retain outliers in the quantized data.
        - outlier_threshold (int): The threshold for detecting outliers.
        - num_workers (int): Number of workers to use for parallel processing.
        - verbose (bool): Whether to print verbose output during optimization.
        """

        self.num_bits = num_bits
        self.quantized_dtype = quantized_dtype
        self.retain_outliers = retain_outliers
        self.outlier_threshold = outlier_threshold
        self.num_workers = num_workers
        self.verbose = verbose
        self.config = config

        if self.quantized_dtype is None:
            # Use the least memory consuming integer type for quantization (if not specified otherwise)
            self.quantized_dtype = get_best_int_type(self.num_bits)

            print(f"Using {self.quantized_dtype} for quantization.")

    def mask_outliers(self, tensor: Tensor) -> Tensor:
        """Mask outliers in a tensor based on the specified number of standard deviations
        Args:
            tensor (Tensor): input tensor

        Returns:
            Tensor: boolean mask of the same shape as the input tensor (True for outliers, False otherwise)
        """
        mean = tensor.mean()
        std = tensor.std()
        outlier_mask = torch.abs(tensor - mean) >= (self.outlier_threshold * std)
        return outlier_mask

    @abstractmethod
    def quantize_layer(self, W: Tensor, **kwargs):
        """Quantizes the weights of a layer. Each row of the weight matrix is quantized independently. (per-channel)
        The quantized weights will be stored as a member variable of the class instance.

        Args:
            W (Tensor): The weights to be quantized.

        """

    @abstractmethod
    def reconstruct_layer(self, **kwargs) -> Tensor:
        """Reconstructs the quantized data back to the original data space.
        Assumes data and reconstruction settings are stored in the class instance.

        Returns:
            Tensor: The reconstructed data.
        """
