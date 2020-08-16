import torch


def mmd_fn(distances):
    min_values, _ = torch.min(distances, dim=0)

    mmd_val = min_values.mean()

    return mmd_val