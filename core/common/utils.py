import torch


def quantize_vector_with_range(
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
    return torch.sum((quantized_x - x) ** 2)
