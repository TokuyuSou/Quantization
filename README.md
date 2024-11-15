# Quantization

This repository contains code for quantizing neural network models, specifically focusing on linear layers. The quantization methods implemented include EasyQuant and SqueezeQuant, which are designed to reduce the memory footprint and computational requirements of models without significantly sacrificing accuracy. Since CUDA is not supported, I mainly focused on measuring the accuracy degradation part.

## Directory Structure

- **core**: Contains the core implementation of the quantization algorithms.
  - **models**: Includes the EasyQuant and SqueezeQuant implementations.
  - **utils**: Utility functions for model manipulation and quantization.
- **script**: Jupyter notebooks demonstrating the usage of the quantization methods.

## Getting Started

### Installation

Clone the repository and install the required packages:

```bash
# Clone the repository
git clone https://github.com/TokuyuSou/Quantization.git
cd Quantization

# Install the required packages using poetry
poetry install
```

### Usage

#### Quantizing a vector

Refer to [script/quantize_vector.ipynb](/script/quantize_vector.ipynb) for a demonstration of quantizing a vector.

#### Quantizing a Model

Refer to [script/quantize_model.ipynb](/script/quantize_simple_model.ipynb) for a demonstration of quantizing a model. Please note that only the quantization of linear layers is supported.

## References

- [How I Built My Own Custom 8-bit Quantizer from Scratch](https://pub.towardsai.net/how-i-built-my-own-custom-8-bit-quantizer-from-scratch-a-step-by-step-guide-using-pytorch-a913cd12e85d)
- [Tang et al. (2024). EasyQuant: A Simple and Effective Quantization Method for Neural Networks.](https://arxiv.org/abs/2403.02775)
- [Kim et al. (2024). SqueezeQuant: SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629)
