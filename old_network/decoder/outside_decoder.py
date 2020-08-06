import torch
from torch import nn


class OutsideDecoder(nn.Module):
    def __init__(self, nb_neighbors, radius, neighborhood_decoder):
        super(OutsideDecoder, self).__init__()

        self.nb_neighbors = nb_neighbors
        self.radius = radius

        self.neighborhood_decoder = neighborhood_decoder

    def forward(self, points, features, batch):
        relative_points, output_batch = self.neighborhood_decoder(features, batch)

        anchor_points = torch.repeat_interleave(
            points, self.nb_neighbors, dim=0
        )
        output_points = anchor_points + (relative_points * self.radius)

        return output_points, output_batch
