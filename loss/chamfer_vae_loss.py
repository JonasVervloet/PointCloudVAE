import torch
from torch import nn


class VAEChamferDistance(nn.Module):
    def __init__(self, alfa=0.5):
        super(VAEChamferDistance, self).__init__()

        self.alfa = alfa

    def forward(self, in_points_list, in_batch_list, out_points_list, out_batch_list,
                mean, variance):
        assert(in_batch_list is not None)
        assert(out_batch_list is not None)
        assert(torch.max(in_batch_list[0]) == torch.max(out_batch_list[0]))

        input_points = in_points_list[0]
        input_batch = in_batch_list[0]

        output_points = out_points_list[0]
        output_batch = out_batch_list[0]

        nb_batch = torch.max(input_batch)
        chamfer_loss = 0
        for i in range(nb_batch):
            chamfer_loss += self.chamfer_dist(
                input_points[input_batch == i],
                output_points[output_batch == i]
            )

        kl_loss = self.get_kl_divergence(mean, variance)

        return chamfer_loss + self.alfa * kl_loss

    @staticmethod
    def chamfer_dist(cloud1, cloud2):
        with torch.no_grad():
            distances = VAEChamferDistance.get_distances(cloud1, cloud2)
            indices = torch.argmin(distances, dim=0)
        points = cloud2[indices]
        loss = torch.sum((cloud1 - points) ** 2)

        with torch.no_grad():
            distances = torch.transpose(distances, 0, 1)
            indices = torch.argmin(distances, dim=0)
        points = cloud1[indices]
        loss += torch.sum((cloud2 - points) ** 2)

        return loss

    @staticmethod
    def get_distances(cloud1, cloud2):
        distances = []
        for i in range(len(cloud2)):
            dists = (cloud1 - cloud2[i]) ** 2
            distances.append(torch.sum(dists, dim=1))
        return torch.stack(distances)

    @staticmethod
    def get_kl_divergence(mean, variance):
        return 0.5 * torch.sum(torch.exp(variance) + mean**2 - variance)