from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import torch
from torch import Tensor

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
            tuple[Tensor]: The quantized data (low-bit representation).

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
