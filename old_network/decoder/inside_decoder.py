import torch
from torch import nn
from torch.nn import functional as F


class InsideDecoder(nn.Module):
    def __init__(self, nb_neighbors, radius, split,
                 feature_size, neighborhood_decoder, features):
        super(InsideDecoder, self).__init__()
        assert(len(features) == 3)

        self.nb_neighbors = nb_neighbors
        self.radius = radius
        self.split = split
        self.feature_size = feature_size

        self.neighborhood_dec = neighborhood_decoder

        self.fc1 = nn.Linear(feature_size + 3, features[0])
        self.fc2 = nn.Linear(features[0], features[1])
        self.fc3 = nn.Linear(features[1], features[2])

    def forward(self, features):
        assert(features.size(1) == self.split + self.feature_size)

        point_features = features[:, :self.split]
        feature_features = features[:, self.split:]

        batch = torch.arange(feature_features.size(0))

        relative_points, output_batch = self.neighborhood_dec(point_features, batch)
        feature_repeated = feature_features.repeat_interleave(self.nb_neighbors, dim=0)
        concat_features = torch.cat([relative_points, feature_repeated], dim=1)

        fc1_features = F.relu(self.fc1(concat_features))
        fc2_features = F.relu(self.fc2(fc1_features))
        fc3_features = F.relu(self.fc3(fc2_features))

        output_points = relative_points * self.radius

        return output_points, fc3_features, output_batch
