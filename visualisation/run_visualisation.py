from visualisation.visualise_cloud_ae import auto_encode_clouds
from standard_network.network.network_generator import NetworkGenerator
from training.save_manager import SaveManager
from dataset.primitive_shapes import PrimitiveShapes


map_path = "D:/Resultaten/Test/"
save_manager = SaveManager(map_path)

network_generator = NetworkGenerator(
    [25, 16, 9],
    [0.3, 1.0, 2.0]
)
network = network_generator.generate_network()
save_manager.load_network(
    network=network,
    epoch_nb=50
)

dataset_generator = PrimitiveShapes(
    train_size=1,
    val_size=1,
    nb_points=3600,
    batch_size=5,
    shuffle=True,
    shapes=[True, True, True, True, True]
)

auto_encode_clouds(network, dataset_generator, save_manager, name="epoch50")


