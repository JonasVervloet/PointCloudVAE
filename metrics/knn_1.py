import torch


def knn_1_fn(dists_rr, dist_rg, dists_gg):

    nb_r = dists_rr.size(0)
    nb_g = dists_gg.size(0)

    label = torch.cat((torch.ones(nb_r), torch.zeros(nb_g)))

    distances = torch.cat((
        torch.cat((dists_rr, dist_rg), dim=1),
        torch.cat((dist_rg.transpose(0, 1), dists_gg), dim=1)
    ), dim=0)

    infinity = float('inf')

    vals, inds = (distances + torch.diag(infinity * torch.ones(nb_r + nb_g))).min(0)
    preds = label[inds]

    return torch.eq(label, preds).float().mean()