import time
from multiprocessing import Pool, cpu_count
from typing import TypedDict

import torch
from sklearn.cluster import KMeans
from torch import Tensor

from core.models.base_model import LayerQuantization, QuantizationMethod
from core.utils.quantization_helpers import get_best_int_type


class SqueezeQuantConfig(TypedDict):
    k_means_max_iter: int
    use_sensitivity: bool


class SqueezeQuantReconstructionSettings(TypedDict):
    cluster_centers: Tensor


class SqueezeQuant(
    QuantizationMethod[SqueezeQuantConfig, SqueezeQuantReconstructionSettings]
):
    def __init__(
        self,
        num_bits: int,
        config: SqueezeQuantConfig,
        quantized_dtype: torch.dtype | None = None,
        verbose: bool = False,
    ):
        """SqueezeLLM algorithm proposed by Kim et al. (2024)


        Args:
        - num_bits (int): Number of bits used to quantize the data.
        - config (ConfigType): A dictionary containing configuration parameters specific to the quantization method.
        - quantized_dtype (torch.dtype): Data type of the quantized weights. (Used for saving the weights in low-bit format)
        - verbose (bool): Whether to print verbose output during optimization.

        """
        super().__init__(num_bits, config, quantized_dtype, verbose)

        self.k_means_max_iter = config["k_means_max_iter"]
        self.use_sensitivity = config["use_sensitivity"]

        if self.quantized_dtype is None:
            # Use the least memory consuming integer type for quantization (if not specified otherwise)
            self.quantized_dtype = get_best_int_type(self.num_bits)

    def optimize(
        self, data: Tensor, weights: Tensor | None = None
    ) -> tuple[dict[int, Tensor], Tensor]:
        """Run weighted K-means algorithm to optimize the quantized points.

        Args:
            data (Tensor): The data to be quantized.
            weights (Tensor): The weights to be used for weighted K-means algorithm.

        Returns:
            tuple[dict[int, Tensor], Tensor]: Tuple containing the cluster center look-up table and the quantized data.
        """

        # If no weights are provided, use uniform weights
        if self.use_sensitivity:
            if weights is None:
                print("No weights provided. Using uniform weights.")

            if weights.shape != data.shape:
                raise ValueError(
                    f"Data and weights must have the same shape. Got data shape {data.shape} and weights shape {weights.shape}"
                )

        if weights is None:
            weights = torch.ones_like(data)

        # Convert data and weights to numpy for compatibility with sklearn's KMeans
        data_np = data.cpu().numpy().reshape(-1, 1)
        weights_np = weights.cpu().numpy().reshape(-1)

        # Calculate number of clusters based on bit width
        n_clusters = 2**self.num_bits

        # Run weighted K-means using sample weights
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=0, max_iter=self.k_means_max_iter
        )

        kmeans.fit(data_np, sample_weight=weights_np)

        # Retrieve cluster centers and labels
        cluster_centers = torch.tensor(
            kmeans.cluster_centers_, dtype=data.dtype
        ).squeeze()  # shape: (n_clusters,)
        cluster_labels = torch.tensor(kmeans.labels_)  # shape: (n_samples,)

        # shift indices to fit the signed integer range
        cluster_labels = cluster_labels - n_clusters // 2

        cluster_labels = cluster_labels.to(self.quantized_dtype)

        # create a look-up table for retrieving the cluster center from the label
        cluster_center_lut = {
            label - n_clusters // 2: center
            for label, center in enumerate(cluster_centers)
        }

        return cluster_center_lut, cluster_labels

    def quantize(self, data: Tensor, weights: Tensor | None = None, **kwargs) -> Tensor:
        """Quantizes the input data and returns the low-bit representation.

        Args:
            data (Tensor): The data to be quantized.
            weights (Tensor): The weights to be used for weighted quantization.

        Returns:
            Tensor: The quantized data (low-bit representation).

        """

        # If weights are all 0, set them to 1 to avoid division by zero
        if weights is not None and torch.all(weights == 0):
            weights = torch.ones_like(weights)

        # Optimize the quantized points using weighted K-means
        cluster_center_lut, cluster_labels = self.optimize(data, weights)

        self.reconstruction_settings = SqueezeQuantReconstructionSettings(
            cluster_centers=cluster_center_lut
        )

        return cluster_labels

    def reconstruct(self, data: Tensor, **kwargs) -> Tensor:
        """Reconstructs the quantized data back to the original data space.

        Args:
            data (Tensor): The quantized data to be reconstructed.

        Returns:
            Tensor: The reconstructed data.
        """
        if self.reconstruction_settings is None:
            raise ValueError(
                "Reconstruction settings not found. Please quantize the data first."
            )

        cluster_center_lut = self.reconstruction_settings["cluster_centers"]

        # Reconstruct the quantized data using the cluster center look-up table
        flattened_data = data.view(-1)  # flatten the data to iterate over each element
        reconstructed_flat = torch.stack(
            [cluster_center_lut[label.item()] for label in flattened_data]
        )

        reconstructed_data = reconstructed_flat.view(data.shape)

        return reconstructed_data

    def quantize_and_reconstruct(
        self, data: Tensor, weights: Tensor | None = None, **kwargs
    ) -> tuple[Tensor, SqueezeQuantReconstructionSettings]:
        """Quantizes and reconstructs the input data.

        Args:
            data (Tensor): The data to be quantized and reconstructed.
            weights (Tensor): The weights to be used for weighted quantization.

        Returns:
            tuple[Tensor, SqueezeQuantReconstructionSettings]: Tuple containing the quantized data and reconstruction settings.
        """

        # Quantize the input data
        quantized_data = self.quantize(data, weights)  # This is the cluster labels

        # Reconstruct the quantized data using the cluster center look-up table
        reconstructed_data = self.reconstruct(quantized_data)

        return reconstructed_data, self.reconstruction_settings


def _quantize_channel(
    args: tuple[int, Tensor, Tensor | None, int, torch.dtype, bool, SqueezeQuantConfig]
) -> tuple[dict[int, Tensor], Tensor]:
    """Quantizes a single channel of a weight matrix using the SquuezeQuant algorithm.

    Args:
        args (tuple[int, Tensor, Tensor | None, int, torch.dtype, bool, EasyQuantConfig]): Tuple containing the channel index, the weight matrix, sensitivity matrix, the number of bits, the quantized data type, a flag to print verbose output, and the configuration dictionary.

    Returns:
        tuple[dict[int, Tensor], Tensor]: Tuple containing the cluster center look-up table and the quantized channel.
    """
    i, W, sensitivity_matrix, num_bits, quantized_dtype, verbose, config = args

    # Extract the i-th channel of the weight matrix
    channel = W[i]
    channel_sensitivity = (
        sensitivity_matrix[i] if sensitivity_matrix is not None else None
    )

    # Quantize the channel using the EasyQuant algorithm
    quantization_executor = SqueezeQuant(num_bits, config, quantized_dtype, verbose)

    quantized_channel = quantization_executor.quantize(channel, channel_sensitivity)

    return (
        quantization_executor.reconstruction_settings["cluster_centers"],
        quantized_channel,
    )


class SqueezeQuantLayerQuantization(
    LayerQuantization[SqueezeQuantConfig, SqueezeQuantReconstructionSettings]
):
    def __init__(
        self,
        num_bits: int,
        config: SqueezeQuantConfig,
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

        self.cluster_center_luts = (
            []
        )  # List of cluster center look-up tables (one per output channel)
        self.normal_weights = None
        self.outlier_weights = None
        self.outlier_mask = None

    def quantize_layer(
        self, W: Tensor, sensitivity_matrix: Tensor | None = None, **kwargs
    ) -> None:
        """Quantizes the layer weights using the SqueezeQuant method.
        The quantized weights will be stored as a member variable of the class instance.

        Args:
            weights (Tensor): The layer weights to be quantized.
            sensitivity_matrix (Tensor | None): The sensitivity matrix used for weighted quantization.
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
            self.outlier_mask = outlier_mask

            # Extract outliers and normal weights
            outlier_weights = W.clone()
            outlier_weights[~outlier_mask] = 0
            normal_weights = W.clone()
            normal_weights[outlier_mask] = 0

            # Store outlier weights in CSR format for efficient processing
            outlier_weights = outlier_weights.to_sparse_csr()

            # Set the sensitivity to 0 for outliers (so that they are not considered in quantization)
            if sensitivity_matrix is not None:
                sensitivity_matrix[outlier_mask] = 0
            else:
                sensitivity_matrix = torch.ones_like(W)
                sensitivity_matrix[outlier_mask] = 0
        else:
            normal_weights = W
            outlier_weights = None

        if sensitivity_matrix is not None:
            sensitivity_matrix = sensitivity_matrix.to(self.device)

        # Allocate memory for the quantized weights and quantization ranges
        quantized_normal_weights = torch.empty_like(
            normal_weights, device=self.device
        ).to(self.quantized_dtype)

        ## Quantization per channel
        if self.verbose:
            print(f"Quantizing {W.size(0)} output channels")

        # Prepare arguments for parallel processing
        args = [
            (
                i,
                normal_weights,
                sensitivity_matrix,
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

        # Extract the cluster center look-up tables and quantized channels
        for i, (cluster_center_lut, quantized_channel) in enumerate(results):
            self.cluster_center_luts.append(cluster_center_lut)
            quantized_normal_weights[i] = quantized_channel

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
            self.cluster_center_luts is None
            or self.normal_weights is None
            or (self.retain_outliers and self.outlier_weights is None)
        ):
            raise ValueError(
                "Quantized weights not found. Please quantize the layer first."
            )

        # Reconstruct the quantized weights using the cluster center look-up tables
        weights_reconstructed = torch.stack(
            [
                torch.tensor(
                    [self.cluster_center_luts[i][label.item()] for label in channel]
                )
                for i, channel in enumerate(self.normal_weights)
            ]
        )

        if self.retain_outliers:
            # Values of the normal weights in outlier positions do not mean anything, so they are set to 0
            weights_reconstructed[self.outlier_mask] = 0
            # Add the outlier weights back to the reconstructed weights
            weights_reconstructed += self.outlier_weights.to_dense()

        return weights_reconstructed
