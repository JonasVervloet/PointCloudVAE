import torch
from torch import nn
from torch.nn import functional as F
import math


class MiddleDecoder(nn.Module):
    def __init__(self, nb_neighbors, radius, input_size,
                 features_global, features):
        super(MiddleDecoder, self).__init__()

        assert(len(features_global) == 3)
        assert(len(features) == 3)

        self.nb_neighbors = nb_neighbors
        self.radius = radius
        self.input_size = input_size

        self.fc1_global = nn.Linear(input_size, features_global[0])
        self.fc2_global = nn.Linear(features_global[0], features_global[1])
        self.fc3_global = nn.Linear(features_global[1], features_global[2])

        self.fc1 = nn.Linear(features_global[2] + 2, features[0])
        self.fc2 = nn.Linear(features[0], features[1])
        self.fc3_feature = nn.Linear(features[1], features[2])
        self.fc3_point = nn.Linear(features[1], 3)

    def forward(self, points, features, batch):
        fc1_global_features = F.relu(self.fc1_global(features))
        fc2_global_features = F.relu(self.fc2_global(fc1_global_features))
        fc3_global_features = F.relu(self.fc3_global(fc2_global_features))

        repeated_features = torch.repeat_interleave(
            fc3_global_features, self.nb_neighbors, dim=0
        )
        grid_repeated = self.create_grid().repeat(
            (features.size(0), 1)
        )
        concat_features = torch.cat(
            [grid_repeated, repeated_features], dim=1
        )

        fc1_features = F.relu(self.fc1(concat_features))
        fc2_features = F.relu(self.fc2(fc1_features))
        fc3_features = F.relu(self.fc3_feature(fc2_features))
        fc3_points = torch.tanh(self.fc3_point(fc2_features))

        anchor_points = torch.repeat_interleave(
            points, self.nb_neighbors, dim=0
        )
        output_batch = torch.repeat_interleave(
            batch, self.nb_neighbors, dim=0
        )
        output_points = anchor_points + (fc3_points * self.radius)

        return output_points, fc3_features, output_batch

    def create_grid(self):
        nb = int(math.sqrt(self.nb_neighbors))
        dist = 1/nb
        grid = []
        for i in range(nb):
            for j in range(nb):
                grid.append(
                    dist * torch.tensor(
                        [i + 0.5, j + 0.5]
                    )
                )

        return torch.stack(grid)
