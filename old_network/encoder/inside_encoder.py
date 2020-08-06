import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class InsideEncoder(nn.Module):
    def __init__(self, radius, input_size, neighborhood_encoder,
                 features, features_global):
        super(InsideEncoder, self).__init__()

        assert(len(features) == 3)
        assert(len(features_global) == 3)

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
        relative_points = points / self.radius
        rel_encoded = self.neighborhood_enc(relative_points, batch)
        rel_enc_mapped = rel_encoded[batch]

        fc_input = torch.cat([rel_enc_mapped, features], dim=1)

        fc1_features = F.relu(self.fc1(fc_input))
        fc2_features = F.relu(self.fc2(fc1_features))
        fc3_features = F.relu(self.fc3(fc2_features))

        max_features = gnn.global_max_pool(
            x=fc3_features,
            batch=batch
        )

        fc1_global_features = F.relu(self.fc1_global(max_features))
        fc2_global_features = F.relu(self.fc2_global(fc1_global_features))
        fc3_global_features = F.relu(self.fc3_global(fc2_global_features))

        return fc3_global_features
