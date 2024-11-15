# This part was partly inspired by the article in https://pub.towardsai.net/how-i-built-my-own-custom-8-bit-quantizer-from-scratch-a-step-by-step-guide-using-pytorch-a913cd12e85d
import os
from typing import Literal

import torch
import torch.mps
import torch.nn.functional as F
from torch import Tensor, nn

from core.models.easy_quant import EasyQuantConfig, EasyQuantLayerQuantization
from core.models.squeeze_llm import SqueezeQuantConfig, SqueezeQuantLayerQuantization
from core.utils.model_copy import copy_model_with_gradients

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

QuantizationAlgorithm = Literal["EasyQuant", "SqueezeQuant"]


class QuantizedLinearLayer(nn.Module):
    """
    A linear layer with quantized weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        quantization_algorithm: QuantizationAlgorithm = "EasyQuant",
        original_dtype: torch.dtype = torch.float32,
        num_bits: int = 8,
        retain_outliers: bool = True,
        outlier_threshold: int = 3,
        verbose: bool = False,
        quantization_config: dict | None = None,
    ):
        """Initialize the quantized linear layer

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            use_bias (bool, optional): Whether to include bias. Defaults to True.
            quantization_algorithm (QuantizationAlgorithm, optional): Quantization algorithm to use. Defaults to "EasyQuant".
            original_dtype (torch.dtype, optional): Data type of the original weights. Defaults to torch.float32.
            num_bits (int, optional): Number of bits to quantize the weights. Defaults to 8.
            retain_outliers (bool, optional): Whether to retain outliers during quantization. Defaults to True.
            outlier_threshold (int, optional): Threshold for identifying outliers. Defaults to 3.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
            quantization_config (dict, optional): Configuration parameters for the quantization algorithm. Defaults to None.
        """

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.quantization_algorithm = quantization_algorithm
        self.original_dtype = original_dtype
        self.num_bits = num_bits
        self.retain_outliers = retain_outliers
        self.outlier_threshold = outlier_threshold
        self.verbose = verbose
        self.quantization_config = quantization_config or {}
        self.quantization_executor = None

        # Use buffers to store the quantized weights
        self.register_buffer("normal_weights", None)
        self.register_buffer("outlier_weights", None)

        # scale will have the same shape as the output features
        self.register_buffer("scales", None)

        # For simplicity, biases are not quantized
        if use_bias:
            self.register_buffer("bias", None)
        else:
            self.bias = None

    def quantize(self, weights: Tensor, **kwargs) -> None:
        """Quantize the weights of the linear layer

        Args:
            weights (Tensor): input weights to be quantized
        """
        # Quantize the weights using the EasyQuant algorithm
        if self.quantization_algorithm == "EasyQuant":
            self.quantization_executor = EasyQuantLayerQuantization(
                num_bits=self.num_bits,
                config=EasyQuantConfig(**self.quantization_config),
                retain_outliers=self.retain_outliers,
                outlier_threshold=self.outlier_threshold,
                verbose=self.verbose,
            )

            self.quantization_executor.quantize_layer(weights)

        elif self.quantization_algorithm == "SqueezeQuant":
            self.quantization_executor = SqueezeQuantLayerQuantization(
                num_bits=self.num_bits,
                config=SqueezeQuantConfig(**self.quantization_config),
                retain_outliers=self.retain_outliers,
                outlier_threshold=self.outlier_threshold,
                verbose=self.verbose,
            )

            if self.quantization_config.get("use_sensitivity"):
                # retrieve gradients with respect to the weights
                weights_sensitivity = kwargs.get("weights_sensitivity")
                if weights_sensitivity is None:
                    raise ValueError(
                        "You need to provide the gradients with respect to the weights for performing sensitivity-based quantization"
                    )
                if (
                    not isinstance(weights_sensitivity, Tensor)
                    or weights_sensitivity.size() != weights.size()
                ):
                    raise ValueError(
                        "Weights sensitivity should be a tensor of the same shape as the weights"
                    )

                self.quantization_executor.quantize_layer(weights, weights_sensitivity)
            else:
                self.quantization_executor.quantize_layer(weights)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the linear layer

        Args:
            input (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        if input.dtype != self.original_dtype:
            raise ValueError(
                f"Input tensor data type {input.dtype} does not match the original data type {self.original_dtype}"
            )
        # Quantize the weights if they are not already quantized
        if self.quantization_executor is None:
            raise ValueError(
                "Weights have not been quantized. Run the quantize method first."
            )

        # Convert the normal weights to the original data type (do calculation in the original data type)
        weights_full_precision = self.quantization_executor.reconstruct_layer()

        if self.use_bias:
            if self.bias is None:
                raise ValueError(
                    "Bias has not been initialized. Please initialize the bias."
                )
            output = F.linear(
                input,
                weights_full_precision,
                self.bias,
            )
        else:
            output = F.linear(input, weights_full_precision)

        return output


def quantize_model(
    base_model: nn.Module,
    quantization_algorithm: QuantizationAlgorithm = "EasyQuant",
    quantization_config: dict | None = None,
    num_bits: int = 8,
    retain_outliers: bool = True,
    outlier_threshold: int = 3,
    verbose: bool = False,
    exclude_list: list[str] | None = None,
    quantized: bool = True,
) -> nn.Module:
    """Quantize all linear layers in a model. This function returns a new model with quantized weights, with the original model unchanged.

    Args:
        base_model (nn.Module): input model
        quantization_algorithm (QuantizationAlgorithm, optional): quantization algorithm to use. Defaults to "EasyQuant".
        quantization_config (dict, optional): configuration parameters for the quantization algorithm. Defaults to None.
        num_bits (int, optional): number of bits to quantize the weights. Defaults to 8.
        retain_outliers (bool, optional): whether to retain outliers during quantization. Defaults to True.
        outlier_threshold (int, optional): threshold for identifying outliers. Defaults to 3.
        verbose (bool, optional): whether to print debug information. Defaults to False.
        exclude_list (list[str], optional): list of layer names to exclude from quantization. Defaults to [].
        quantized (bool, optional): whether to quantize the weights. Defaults to True.

    Returns:
        nn.Module: quantized model
    """
    # Create a deep copy of the base model (with gradients if available)
    quantized_model = copy_model_with_gradients(base_model)

    # Replace all linear layers in the model with quantized linear layers
    replace_linear_layer_with_quantized(
        quantized_model,
        quantization_algorithm=quantization_algorithm,
        quantization_config=quantization_config,
        num_bits=num_bits,
        retain_outliers=retain_outliers,
        outlier_threshold=outlier_threshold,
        verbose=verbose,
        exclude_list=exclude_list,
        quantized=quantized,
    )

    return quantized_model


def replace_linear_layer_with_quantized(
    base_model: nn.Module,
    quantization_algorithm: QuantizationAlgorithm = "EasyQuant",
    quantization_config: dict | None = None,
    num_bits: int = 8,
    retain_outliers: bool = True,
    outlier_threshold: int = 3,
    verbose: bool = False,
    exclude_list: list[str] | None = None,
    quantized: bool = True,
) -> None:
    """Replace all linear layers in a model with quantized linear layers

    Args:
        base_model (nn.Module): input model
        quantization_algorithm (QuantizationAlgorithm, optional): quantization algorithm to use. Defaults to "EasyQuant".
        quantization_config (dict, optional): configuration parameters for the quantization algorithm. Defaults to None.
        num_bits (int, optional): number of bits to quantize the weights. Defaults to 8.
        retain_outliers (bool, optional): whether to retain outliers during quantization. Defaults to True.
        outlier_threshold (int, optional): threshold for identifying outliers. Defaults to 3.
        verbose (bool, optional): whether to print debug information. Defaults to False.
        exclude_list (list[str], optional): list of layer names to exclude from quantization. Defaults to [].
        quantized (bool, optional): whether to quantize the weights. Defaults to True.
    """
    if exclude_list is None:
        exclude_list = []
    # Iterate through all the modules in the model
    for name, module in base_model.named_children():
        if name in exclude_list:
            continue

        if verbose:
            print(f"Quantizing module: {name}")

        # Replace linear layers with quantized linear layers
        if isinstance(module, nn.Linear):
            setattr(
                base_model,
                name,
                QuantizedLinearLayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    use_bias=module.bias is not None,
                    quantization_algorithm=quantization_algorithm,
                    original_dtype=module.weight.dtype,
                    num_bits=num_bits,
                    retain_outliers=retain_outliers,
                    outlier_threshold=outlier_threshold,
                    verbose=verbose,
                    quantization_config=quantization_config,
                ),
            )
            if quantized:
                if (
                    quantization_algorithm == "SqueezeQuant"
                    and quantization_config.get("use_sensitivity")
                ):
                    # retrieve gradients with respect to the weights and calculate the sensitivity
                    weights_sensitivity = torch.square(module.weight.grad)
                    if weights_sensitivity is None:
                        raise ValueError(
                            "SqueezeQuant requires the gradients with respect to the weights to be available"
                        )

                    getattr(base_model, name).quantize(
                        module.weight, weights_sensitivity=weights_sensitivity
                    )
                else:
                    getattr(base_model, name).quantize(module.weight)
            if module.bias is not None:
                getattr(base_model, name).bias = module.bias

        else:
            print(f"Module {name} is not a linear layer. Going deeper...")
            # Recursively apply quantization to the submodules
            replace_linear_layer_with_quantized(
                module,
                quantization_algorithm=quantization_algorithm,
                quantization_config=quantization_config,
                num_bits=num_bits,
                retain_outliers=retain_outliers,
                outlier_threshold=outlier_threshold,
                verbose=verbose,
                exclude_list=exclude_list,
                quantized=quantized,
            )
