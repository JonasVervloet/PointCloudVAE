import torch
from torch_geometric.data import Batch

from standard_network.network.network_generator import NetworkGenerator
from dataset.primitive_shapes import PrimitiveShapes

generator = NetworkGenerator(
    nbs_neighbors=[25, 16, 9],
    radii=[0.25, 1.0, 2.0]
)
network = generator.generate_network()

print(network)

input_points = torch.randn((7200, 3))
input_batch = torch.arange(2).repeat_interleave(3600)
batch_obj = Batch(
    pos=input_points,
    batch=input_batch
)

print(input_points.size())
print(input_batch.size())
print(torch.max(input_batch))
print(batch_obj)
print()

in_points, in_batch, out_points, out_batch, latent_features = network(batch_obj)

print()
print(in_points.size())
print(in_batch.size())
print(torch.max(in_batch))
print(out_points.size())
print(out_batch.size())
print(torch.max(out_batch))
print(latent_features.size())

dataset_generator = PrimitiveShapes(
    train_size=10,
    val_size=2,
    nb_points=3600,
    batch_size=5
)

train_loader, val_loader = dataset_generator.generate_loaders()
print()
print(train_loader)
for batch in train_loader:
    print(batch)

print()
print(val_loader)
for batch in val_loader:
    print(batch)

train_size, val_size = dataset_generator.get_sizes()
print()
print(train_size)
print(val_size)


