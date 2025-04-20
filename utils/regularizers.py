# regularizers.py

import torch
import torch.nn as nn


def l1_regularization(model):
    return sum(p.abs().sum() for p in model.parameters())


def l2_regularization(model):
    return sum(p.pow(2).sum() for p in model.parameters())


def adaptive_regularization(model, alpha=1e-4):
    """
    Apply adaptive regularization based on activation variance.
    Assumes model.activations is a list of tensors (one per layer).
    """
    if not hasattr(model, 'activations') or not model.activations:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    act = torch.cat(model.activations, dim=0)
    var = torch.var(act, dim=0, unbiased=False).to(act.device)
    lambdas = 1.0 / (1.0 + var)

    if hasattr(model, 'fc1') and hasattr(model.fc1, 'weight'):
        weights = model.fc1.weight.pow(2).mean(dim=1)
        reg = torch.sum(lambdas * weights)
        return alpha * reg
    else:
        return torch.tensor(0.0, device=act.device)
