import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class MiddleEncoder(nn.Module):
    def __init__(self, nb_neighbors, radius, input_size,
                 neighborhood_encoder, features, features_global):
        super(MiddleEncoder, self).__init__()

        assert(len(features) == 3)
        assert(len(features_global) == 3)

        self.nb_neighbors = nb_neighbors
        self.radius = radius
        self.input_size = input_size

        self.neighborhood_enc = neighborhood_encoder

        self.fc1 = nn.Linear(input_size, features[0])
        self.fc2 = nn.Linear(features[0], features[1])
        self.fc3 = nn.Linear(features[1], features[2])

        self.fc1_global = nn.Linear(features[2], features_global[0])
        self.fc2_global = nn.Linear(features_global[0], features_global[1])
        self.fc3_global = nn.Linear(features_global[1], features_global[2])

    def forward(self, points, features, batch):
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
        radius_features = features[radius_indices]

        relative_points = (radius_points - anchor_points) / self.radius
        rel_encoded = self.neighborhood_enc(relative_points, radius_cluster)
        rel_enc_mapped = rel_encoded[radius_cluster]

        fc_input = torch.cat([relative_points, rel_enc_mapped, radius_features], dim=1)

        fc1_features = F.relu(self.fc1(fc_input))
        fc2_features = F.relu(self.fc2(fc1_features))
        fc3_features = F.relu(self.fc3(fc2_features))

        max_features = gnn.global_max_pool(
            x=fc3_features,
            batch=radius_cluster
        )

        fc1_global_features = F.relu(self.fc1_global(max_features))
        fc2_global_features = F.relu(self.fc2_global(fc1_global_features))
        fc3_global_features = F.relu(self.fc3_global(fc2_global_features))

        output_features = torch.cat([rel_encoded, fc3_global_features], dim=1)

        return fps_points, output_features, fps_batch