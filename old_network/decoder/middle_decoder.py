import torch
from torch import nn
from torch.nn import functional as F
import math


class MiddleDecoder(nn.Module):
    def __init__(self, nb_neighbors, radius, input_size_neigh_dec,
                 input_size_feats_dec, neighborhood_decoder, features):
        super(MiddleDecoder, self).__init__()

        assert(len(features) == 3)

        self.nb_neighbors = nb_neighbors
        self.radius = radius
        self.input_size_neigh_dec = input_size_neigh_dec
        self.input_size_feats_dec = input_size_feats_dec

        self.neighborhood_dec = neighborhood_decoder

        self.fc1 = nn.Linear(input_size_feats_dec, features[0])
        self.fc2 = nn.Linear(features[0], features[1])
        self.fc3 = nn.Linear(features[1], features[2])

    def forward(self, points, features, batch):
        point_features = features[:, :self.input_size_neigh_dec]
        feature_features = features[:, self.input_size_neigh_dec:]

        relative_points, output_batch = self.neighborhood_dec(points, point_features, batch)
        features_repeated = feature_features.repeat_interleave(self.nb_neighbors, dim=0)
        concat_features = torch.cat([relative_points, features_repeated], dim=1)

        fc1_features = F.relu(self.fc1(concat_features))
        fc2_features = F.relu(self.fc2(fc1_features))
        fc3_features = F.relu(self.fc3(fc2_features))

        anchor_points = torch.repeat_interleave(
            points, self.nb_neighbors, dim=0
        )
        output_points = anchor_points + (relative_points * self.radius)

        return output_points, fc3_features, output_batch
