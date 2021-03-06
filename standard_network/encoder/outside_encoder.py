from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class OutsideEncoder(nn.Module):
    def __init__(self, nb_neighbors, radius,
                 features, features_global):
        super(OutsideEncoder, self).__init__()

        assert(len(features) == 3)
        assert(len(features_global) == 3)

        self.nb_neighbors = nb_neighbors
        self.radius = radius

        self.fc1 = nn.Linear(3, features[0])
        self.fc2 = nn.Linear(features[0], features[1])
        self.fc3 = nn.Linear(features[1], features[2])

        self.fc1_global = nn.Linear(features[2], features_global[0])
        self.fc2_global = nn.Linear(features_global[0], features_global[1])
        self.fc3_global = nn.Linear(features_global[1], features_global[2])

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

        fc1_features = F.relu(self.fc1(relative_points))
        fc2_features = F.relu(self.fc2(fc1_features))
        fc3_features = F.relu(self.fc3(fc2_features))

        max_features = gnn.global_max_pool(
            x=fc3_features,
            batch=radius_cluster
        )

        fc1_global_features = F.relu(self.fc1_global(max_features))
        fc2_global_features = F.relu(self.fc2_global(fc1_global_features))
        fc3_global_features = F.relu(self.fc3_global(fc2_global_features))

        return fps_points, fc3_global_features, fps_batch
