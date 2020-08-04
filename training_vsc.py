from training.training_manager import TrainingManager
from standard_network.network.network_generator import NetworkGenerator
from dataset.primitive_shapes import PrimitiveShapes


map_path = "/data/leuven/335/vsc33597/StandardNetwork/Test0408/"

network_generator = NetworkGenerator(
    [25, 16, 9],
    [0.3, 1.0, 2.0]
)
network = network_generator.generate_network()
print(network)

dataset_generator = PrimitiveShapes(
    train_size=20,
    val_size=2,
    nb_points=3600,
    batch_size=5,
    shuffle=True,
    shapes=[True, True, True, True, True]
)

training_manager = TrainingManager()
training_manager.set_map_path(map_path)
training_manager.set_network(network)
training_manager.set_dataset_generator(dataset_generator)

training_manager.train(
    end_epoch=20,
    lr=0.001,
    weight_decay=0
)