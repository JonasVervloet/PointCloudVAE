import torch
from torch import nn
from torch.nn import functional as F


class NeighborhoodDecoder(nn.Module):
    """
    This decoder takes in features the batch they belong to and outputs
        relative points and the batch they belong to.
    """
    def __init__(self, nb_neighbors, input_size,
                 features_global, features):
        super(NeighborhoodDecoder, self).__init__()

        assert(len(features_global) == 3)
        assert(len(features) == 3)

        self.nb_neighbors = nb_neighbors
        self.input_size = input_size
        self.latent_feature = features_global[2]

        self.fc1_global = nn.Linear(input_size, features_global[0])
        self.fc2_global = nn.Linear(features_global[0], features_global[1])
        self.fc3_global = nn.Linear(features_global[1], features_global[2]*self.nb_neighbors)

        self.fc1 = nn.Linear(features_global[2], features[0])
        self.fc2 = nn.Linear(features[0], features[1])
        self.fc3_point = nn.Linear(features[1], features[2])

    def forward(self, features, batch):
        fc1_global_features = F.relu(self.fc1_global(features))
        fc2_global_features = F.relu(self.fc2_global(fc1_global_features))
        fc3_global_features = F.relu(self.fc3_global(fc2_global_features))

        resized = fc3_global_features.view(-1, self.latent_feature)

        fc1_features = F.relu(self.fc1(resized))
        fc2_features = F.relu(self.fc2(fc1_features))
        output_points = torch.tanh(self.fc3_point(fc2_features))

        output_batch = torch.repeat_interleave(
            batch, self.nb_neighbors, dim=0
        )

        return output_points, output_batch


class OldNeighborhoodDecoder(nn.Module):
    """
    This decoder takes in features the batch they belong to and outputs
        relative points and the batch they belong to.
    """
    def __init__(self, nb_neighbors, input_size, features_global):
        super(OldNeighborhoodDecoder, self).__init__()

        assert(len(features_global) == 2)

        self.nb_neighbors = nb_neighbors
        self.input_size = input_size

        self.latent_feature = features_global[1]

        self.fc1_global = nn.Linear(input_size, features_global[0])
        self.fc2_global = nn.Linear(features_global[0], features_global[1])
        self.fc3_global = nn.Linear(features_global[1], features_global[1]*self.nb_neighbors)

        self.fc1 = nn.Linear(features_global[1], 3)

    def forward(self, features, batch):
        fc1_global_features = F.relu(self.fc1_global(features))
        fc2_global_features = F.relu(self.fc2_global(fc1_global_features))
        fc3_global_features = F.relu(self.fc3_global(fc2_global_features))

        resized = fc3_global_features.view(-1, self.latent_feature)

        output_points = torch.tanh(self.fc1(resized))

        output_batch = torch.repeat_interleave(
            batch, self.nb_neighbors, dim=0
        )

        return output_points, output_batch
