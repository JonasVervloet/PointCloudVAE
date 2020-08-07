from old_network.encoder.middle_encoder import MiddleEncoder
from old_network.encoder.outside_encoder import OutsideEncoder
from old_network.encoder.inside_encoder import InsideEncoder
from old_network.encoder.neighborhood_encoder import NeighborhoodEncoder

from old_network.decoder.middle_decoder import MiddleDecoder
from old_network.decoder.outside_decoder import OutsideDecoder
from old_network.decoder.inside_decoder import InsideDecoder
from old_network.decoder.neighborhood_decoder import NeighborhoodDecoder

from full_network.network import PointCloudAE


class NetworkGenerator:
    def __init__(self, nbs_neighbors, radii):
        assert(len(nbs_neighbors) == 3)
        assert(len(radii) == 3)

        self.nbs_neighbors = nbs_neighbors
        self.radii = radii

    def generate_network(self):
        outside_neigh_enc = NeighborhoodEncoder(
            features=[8, 16, 64],
            features_global=[32, 16, 8]
        )
        outside_encoder = OutsideEncoder(
            nb_neighbors=self.nbs_neighbors[0],
            radius=self.radii[0],
            neighborhood_encoder=outside_neigh_enc
        )

        middle_neigh_enc = NeighborhoodEncoder(
            features=[8, 16, 64],
            features_global=[32, 16, 8]
        )
        middle_encoder = MiddleEncoder(
            nb_neighbors=self.nbs_neighbors[1],
            radius=self.radii[1],
            input_size=8+8+3,
            neighborhood_encoder=middle_neigh_enc,
            features=[32, 64, 128],
            features_global=[64, 32, 16]
        )

        inside_neigh_enc = NeighborhoodEncoder(
            features=[8, 16, 64],
            features_global=[32, 16, 8]
        )
        inside_encoder = InsideEncoder(
            radius=self.radii[2],
            input_size=24+8+3,
            neighborhood_encoder=inside_neigh_enc,
            features=[64, 128, 256],
            features_global=[128, 64, 32]
        )

        encoders = [outside_encoder, middle_encoder, inside_encoder]

        inside_neigh_dec = NeighborhoodDecoder(
            nb_neighbors=self.nbs_neighbors[2],
            input_size=8,
            features_global=[16, 32, 64],
            features=[16, 8, 3]
        )
        inside_decoder = InsideDecoder(
            nb_neighbors=self.nbs_neighbors[2],
            radius=self.radii[2],
            split=8,
            feature_size=32,
            neighborhood_decoder=inside_neigh_dec,
            features=[64, 32, 24]
        )

        middle_neigh_dec = NeighborhoodDecoder(
            nb_neighbors=self.nbs_neighbors[1],
            input_size=8,
            features_global=[16, 32, 64],
            features=[16, 8, 3]
        )
        middle_decoder = MiddleDecoder(
            nb_neighbors=self.nbs_neighbors[1],
            radius=self.radii[1],
            split=8,
            feature_size=16,
            neighborhood_decoder=middle_neigh_dec,
            features=[32, 16, 8]
        )

        outside_neigh_dec = NeighborhoodDecoder(
            nb_neighbors=self.nbs_neighbors[0],
            input_size=8,
            features_global=[16, 32, 64],
            features=[16, 8, 3]
        )
        outside_decoder = OutsideDecoder(
            nb_neighbors=self.nbs_neighbors[0],
            radius=self.radii[0],
            neighborhood_decoder=outside_neigh_dec
        )

        decoders = [inside_decoder, middle_decoder, outside_decoder]

        return PointCloudAE(
            encoders=encoders,
            decoders=decoders
        )