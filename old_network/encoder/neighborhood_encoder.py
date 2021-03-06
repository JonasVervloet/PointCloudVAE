from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class NeighborhoodEncoder(nn.Module):
    """
    This encoder takes in relative points and the cluster to which they belong and outputs
        a single feature vector for each cluster.
    """
    def __init__(self, features, features_global):
        super(NeighborhoodEncoder, self).__init__()

        assert(len(features) == 3)
        assert(len(features_global) == 3)

        self.fc1 = nn.Linear(3, features[0])
        self.fc2 = nn.Linear(features[0], features[1])
        self.fc3 = nn.Linear(features[1], features[2])

        self.fc1_global = nn.Linear(features[2], features_global[0])
        self.fc2_global = nn.Linear(features_global[0], features_global[1])
        self.fc3_global = nn.Linear(features_global[1], features_global[2])

    def forward(self, relative_points, cluster):

        fc1_features = F.relu(self.fc1(relative_points))
        fc2_features = F.relu(self.fc2(fc1_features))
        fc3_features = F.relu(self.fc3(fc2_features))

        max_features = gnn.global_max_pool(
            x=fc3_features,
            batch=cluster
        )

        fc1_global_features = F.relu(self.fc1_global(max_features))
        fc2_global_features = F.relu(self.fc2_global(fc1_global_features))
        fc3_global_features = F.relu(self.fc3_global(fc2_global_features))

        return fc3_global_features


class OldNeighborhoodEncoder(nn.Module):
    """
    This encoder takes in relative points and the cluster to which they belong and outputs
        a single feature vector for each cluster.
    """
    def __init__(self, feature, features_global):
        super(OldNeighborhoodEncoder, self).__init__()

        assert(len(features_global) == 2)

        self.fc1 = nn.Linear(3, feature)

        self.fc1_global = nn.Linear(feature, features_global[0])
        self.fc2_global = nn.Linear(features_global[0], features_global[1])

    def forward(self, relative_points, cluster):

        fc1_features = F.relu(self.fc1(relative_points))

        max_features = gnn.global_max_pool(
            x=fc1_features,
            batch=cluster
        )

        fc1_global_features = F.relu(self.fc1_global(max_features))
        fc2_global_features = F.relu(self.fc2_global(fc1_global_features))

        return fc2_global_features
