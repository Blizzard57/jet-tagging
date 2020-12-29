import torch


def count_params(model: torch.nn.Module, grad_only: bool = True):
    count: int = 0
    for params in model.parameters():
        if grad_only and params.requires_grad:
            count += params.numel()
    return count
