from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class OutsideEncoder(nn.Module):
    def __init__(self, nb_neighbors, radius, neighborhood_encoder):
        super(OutsideEncoder, self).__init__()

        self.nb_neighbors = nb_neighbors
        self.radius = radius

        self.neighborhood_encoder = neighborhood_encoder

    def forward(self, points, batch):
        ratio = 1/self.nb_neighbors
        fps_indices = gnn.fps(
            x=points,
            batch=batch,
            ratio=ratio
        )
        fps_points = points[fps_indices]
        fps_batch = batch[fps_indices]

        radius_cluster, radius_indices = gnn.radius(
            x=points,
            y=fps_points,
            batch_x=batch,
            batch_y=fps_batch,
            r=self.radius
        )
        anchor_points = fps_points[radius_cluster]
        radius_points = points[radius_indices]

        relative_points = (radius_points - anchor_points) / self.radius

        features = self.neighborhood_encoder(relative_points, radius_cluster)

        return fps_points, features, fps_batch
