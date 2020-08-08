from old_network.encoder.middle_encoder import OldMiddleEncoder
from old_network.encoder.outside_encoder import OutsideEncoder
from old_network.encoder.inside_encoder import OldInsideEncoder
from old_network.encoder.neighborhood_encoder import OldNeighborhoodEncoder

from old_network.decoder.middle_decoder import OldMiddleDecoder
from old_network.decoder.outside_decoder import OutsideDecoder
from old_network.decoder.inside_decoder import OldInsideDecoder
from old_network.decoder.neighborhood_decoder import OldNeighborhoodDecoder

from full_network.network import PointCloudAE


class NetworkGenerator:
    def __init__(self, nbs_neighbors, radii):
        assert(len(nbs_neighbors) == 3)
        assert(len(radii) == 3)

        self.nbs_neighbors = nbs_neighbors
        self.radii = radii

    def generate_network(self):
        outside_neigh_enc = OldNeighborhoodEncoder(
            feature=80,
            features_global=[40, 20]
        )
        outside_encoder = OutsideEncoder(
            nb_neighbors=self.nbs_neighbors[0],
            radius=self.radii[0],
            neighborhood_encoder=outside_neigh_enc
        )

        middle_neigh_enc = OldNeighborhoodEncoder(
            feature=80,
            features_global=[40, 20]
        )
        middle_encoder = OldMiddleEncoder(
            nb_neighbors=self.nbs_neighbors[1],
            radius=self.radii[1],
            input_size=20+20+3,
            feature=30,
            neighborhood_encoder=middle_neigh_enc
        )

        inside_neigh_enc = OldNeighborhoodEncoder(
            feature=80,
            features_global=[40, 20]
        )
        inside_encoder = OldInsideEncoder(
            radius=self.radii[2],
            input_size=20+50+3,
            feature=80,
            neighborhood_encoder=inside_neigh_enc
        )

        encoders = [outside_encoder, middle_encoder, inside_encoder]

        inside_neigh_dec = OldNeighborhoodDecoder(
            nb_neighbors=self.nbs_neighbors[2],
            input_size=20,
            features_global=[40, 80]
        )
        inside_decoder = OldInsideDecoder(
            nb_neighbors=self.nbs_neighbors[2],
            radius=self.radii[2],
            split=20,
            neighborhood_decoder=inside_neigh_dec,
            feature_size=80,
            output_size=50
        )

        middle_neigh_dec = OldNeighborhoodDecoder(
            nb_neighbors=self.nbs_neighbors[1],
            input_size=20,
            features_global=[40, 80]
        )
        middle_decoder = OldMiddleDecoder(
            nb_neighbors=self.nbs_neighbors[1],
            radius=self.radii[1],
            split=20,
            neighborhood_decoder=middle_neigh_dec,
            feature_size=30,
            output_size=20
        )

        outside_neigh_dec = OldNeighborhoodDecoder(
            nb_neighbors=self.nbs_neighbors[0],
            input_size=20,
            features_global=[40, 80]
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