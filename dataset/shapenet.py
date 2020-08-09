import torch
from torch_geometric.datasets.shapenet import ShapeNet
from torch_geometric.transforms import LinearTransformation

from dataset.dataset_generator import DatasetGenerator


class ShapeNetGenerator(DatasetGenerator):
    PATH = "D:/Documenten/Documenten Mie/jonas/ShapeNet/"

    def __init__(self, batch_size, shuffle=True, categories="Airplane"):
        DatasetGenerator.__init__(self, batch_size, shuffle)
        self.categories = categories
        self.train_size = None
        self.val_size = None

    def generate_train_dataset(self):
        train_set = ShapeNet(
            ShapeNetGenerator.PATH,
            categories=self.categories,
            include_normals=False,
            split='train',
            transform=LinearTransformation(
                torch.tensor([
                    [2.0, 0, 0],
                    [0, 2.0, 0],
                    [0, 0, 2.0]
                ])
            )
        )
        self.train_size = len(train_set)
        return train_set

    def generate_validation_dataset(self):
        val_set = ShapeNet(
            ShapeNetGenerator.PATH,
            categories=self.categories,
            include_normals=False,
            split='val',
            transform=LinearTransformation(
                torch.tensor([
                    [2.0, 0, 0],
                    [0, 2.0, 0],
                    [0, 0, 2.0]
                ])
            )
        )
        self.val_size = len(val_set)
        return val_set

    def get_train_size(self):
        return self.train_size

    def get_val_size(self):
        return self.val_size