from torch_geometric.datasets.shapenet import ShapeNet

from dataset.dataset_generator import DatasetGenerator

PATH = "D:/Shapenet/"


class ShapeNetGenerator(DatasetGenerator):
    def __init__(self, batch_size, shuffle=True, category="Airplane"):
        DatasetGenerator.__init__(self, batch_size, shuffle)
        self.category = category
        self.train_size = None
        self.val_size = None

    def generate_train_dataset(self):
        return ShapeNet(
            root=PATH,
            categories=self.category
        )


    def generate_validation_dataset(self):
        print()

    def get_train_size(self):
        return self.train_size

    def get_val_size(self):
        return self.val_size