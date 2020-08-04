from standard_network.encoder.middle_encoder import MiddleEncoder
from standard_network.encoder.outside_encoder import OutsideEncoder
from standard_network.encoder.inside_encoder import InsideEncoder

from standard_network.decoder.middle_decoder import MiddleDecoder
from standard_network.decoder.outside_decoder import OutsideDecoder
from standard_network.decoder.inside_decoder import InsideDecoder

from standard_network.network.vae_network import PointCloudVAE


class NetworkGenerator:
    def __init__(self, nbs_neighbors, radii):
        assert(len(nbs_neighbors) == 3)
        assert(len(radii) == 3)

        self.nbs_neighbors = nbs_neighbors
        self.radii = radii

    def generate_network(self):
        outside_encoder = OutsideEncoder(
            nb_neighbors=self.nbs_neighbors[0],
            radius=self.radii[0],
            features=[8, 16, 64],
            features_global=[32, 16, 8]
        )
        middle_encoder = MiddleEncoder(
            nb_neighbors=self.nbs_neighbors[1],
            radius=self.radii[1],
            input_size=8 + 3,
            features=[16, 32, 128],
            features_global=[64, 32, 16]
        )
        inside_encoder = InsideEncoder(
            radius=self.radii[2],
            input_size=16 + 3,
            features=[32, 64, 256],
            features_global=[128, 64, 32]
        )
        encoders = [outside_encoder, middle_encoder, inside_encoder]

        inside_decoder = InsideDecoder(
            nb_neighbors=self.nbs_neighbors[2],
            radius=self.radii[2],
            input_size=32,
            features_global=[64, 128, 256],
            features=[64, 32, 16]
        )
        middle_decoder = MiddleDecoder(
            nb_neighbors=self.nbs_neighbors[1],
            radius=self.radii[1],
            input_size=16,
            features_global=[32, 64, 128],
            features=[32, 16, 8]
        )
        outside_decoder = OutsideDecoder(
            nb_neighbors=self.nbs_neighbors[0],
            radius=self.radii[0],
            input_size=8,
            features_global=[16, 32, 64],
            features=[16, 8]
        )
        decoders = [inside_decoder, middle_decoder, outside_decoder]

        return PointCloudVAE(
            encoders=encoders,
            decoders=decoders,
            lat_size=32
        )