import torch
import os
import sys

sys.path.append(os.path.abspath("/Users/deyucao/Quantization"))
print(sys.path)

from core.quantization import quantize_layer
from core.models.easy_quant import EasyQuantConfig

if __name__ == "__main__":
    # Define the weight tensor
    W = torch.randn(10, 10)

    # Define the configuration for EasyQuant
    config = EasyQuantConfig(learning_rate=0.1, num_epochs=100)

    # Quantize the weight tensor using the EasyQuant algorithm
    quantized_W, outliers_W = quantize_layer(
        W,
        num_bits=3,
        quantization_algorithm="EasyQuant",
        retain_outliers=True,
        outlier_threshold=2,
        verbose=False,
        **config,
    )

    for i in range(W.size(0)):
        print(f"Original tensor {i}:\n{W[i]}")
        print(f"Quantized tensor {i}:\n{quantized_W[i]}")
        print(f"Outlier tensor {i}:\n{outliers_W.to_dense()[i]}")
        print("\n")
