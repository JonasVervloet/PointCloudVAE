import torch


def coverage_quality_fn(distances, max_distance):
    nb_gen, nb_ref = distances.size()
    coverages = torch.zeros(nb_ref)

    min_dists, min_inds = torch.min(distances, dim=1)

    for i in range(nb_ref):
        curr_min_dists = min_dists[min_inds == i]
        if len(curr_min_dists) > 0:
            ceiling = max_distance * torch.ones(len(curr_min_dists))
            clipped = torch.min(curr_min_dists, ceiling)
            clipped_rel = clipped / max_distance

            coverages[i] = 1 - torch.sum(clipped_rel)/len(curr_min_dists)

    return coverages.mean()