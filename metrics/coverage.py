import torch


def coverage_fn(distances):
    nb_sample, nb_ref = distances.size()
    _, min_inds = torch.min(distances, dim=1)

    cov_val = float(min_inds.unique().size(0)) / float(nb_ref)

    return torch.tensor(cov_val)