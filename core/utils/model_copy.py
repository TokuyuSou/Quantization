import torch
import copy


def copy_model_with_gradients(model: torch.nn.Module) -> torch.nn.Module:
    """Copies a PyTorch model and its gradients.

    Args:
        model (torch.nn.Module): The model to copy.

    Returns:
        torch.nn.Module: The copied model.
    """
    copied_model = copy.deepcopy(model)
    for new_param, orig_param in zip(copied_model.parameters(), model.parameters()):
        if orig_param.grad is not None:
            new_param.grad = orig_param.grad.clone()
    return copied_model
