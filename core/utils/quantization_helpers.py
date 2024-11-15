import torch
from torch import Tensor


def quantize_and_reconstruct_vector_with_range(
    x: Tensor, q_range: Tensor | float, num_bits: int, detach: bool = False
) -> Tensor:
    # Quantization assumes using symmetric quantization levels
    result = (
        torch.clamp(
            torch.round(x / q_range),
            min=-(2 ** (num_bits - 1)) + 1,
            max=2 ** (num_bits - 1),
        )
        * q_range
    )

    if detach:
        return result.detach()
    return result


def quantize_vector_with_range(
    x: Tensor,
    q_range: Tensor | float,
    num_bits: int,
    quantized_dtype: torch.dtype,
    detach: bool = False,
) -> Tensor:
    result = torch.clamp(
        torch.round(x / q_range),
        min=-(2 ** (num_bits - 1)) + 1,
        max=2 ** (num_bits - 1),
    ).to(quantized_dtype)

    if detach:
        return result.detach()
    return result


def compute_quantization_error(x: Tensor, quantized_x: Tensor) -> float:
    return torch.sum((quantized_x - x) ** 2)


def get_best_int_type(num_bits: int) -> torch.dtype:
    if num_bits <= 8:
        return torch.int8
    elif num_bits <= 16:
        return torch.int16
    elif num_bits <= 32:
        return torch.int32
    else:
        return torch.int64
