from old_network.encoder.middle_encoder import OldMiddleEncoder
from old_network.encoder.outside_encoder import OutsideEncoder
from old_network.encoder.inside_encoder import OldInsideEncoder
from old_network.encoder.neighborhood_encoder import OldNeighborhoodEncoder

from old_network.decoder.middle_decoder import OldMiddleDecoder
from old_network.decoder.outside_decoder import OutsideDecoder
from old_network.decoder.inside_decoder import OldInsideDecoder
from old_network.decoder.neighborhood_decoder import OldNeighborhoodDecoder

from full_network.vae_network import PointCloudVAE


class NetworkGenerator:
    def __init__(self, nbs_neighbors, radii, multiplier):
        assert(len(nbs_neighbors) == 3)
        assert(len(radii) == 3)
        assert(isinstance(multiplier, int))

        self.nbs_neighbors = nbs_neighbors
        self.radii = radii

        self.multiplier = multiplier

    def generate_network(self):
        outside_neigh_enc = OldNeighborhoodEncoder(
            feature=40 * self.multiplier,
            features_global=[20 * self.multiplier, 10 * self.multiplier]
        )
        outside_encoder = OutsideEncoder(
            nb_neighbors=self.nbs_neighbors[0],
            radius=self.radii[0],
            neighborhood_encoder=outside_neigh_enc
        )

        middle_neigh_enc = OldNeighborhoodEncoder(
            feature=40 * self.multiplier,
            features_global=[20 * self.multiplier, 10 * self.multiplier]
        )
        middle_encoder = OldMiddleEncoder(
            nb_neighbors=self.nbs_neighbors[1],
            radius=self.radii[1],
            input_size=10 * self.multiplier + 10 * self.multiplier + 3,
            feature=15 * self.multiplier,
            neighborhood_encoder=middle_neigh_enc
        )

        inside_neigh_enc = OldNeighborhoodEncoder(
            feature=40 * self.multiplier,
            features_global=[20 * self.multiplier, 10 * self.multiplier]
        )
        inside_encoder = OldInsideEncoder(
            radius=self.radii[2],
            input_size=10 * self.multiplier + 25 * self.multiplier + 3,
            feature=40 * self.multiplier,
            neighborhood_encoder=inside_neigh_enc
        )

        encoders = [outside_encoder, middle_encoder, inside_encoder]

        inside_neigh_dec = OldNeighborhoodDecoder(
            nb_neighbors=self.nbs_neighbors[2],
            input_size=10 * self.multiplier,
            features_global=[20 * self.multiplier, 40 * self.multiplier]
        )
        inside_decoder = OldInsideDecoder(
            nb_neighbors=self.nbs_neighbors[2],
            radius=self.radii[2],
            split=10 * self.multiplier,
            neighborhood_decoder=inside_neigh_dec,
            feature_size=40 * self.multiplier,
            output_size=25 * self.multiplier
        )

        middle_neigh_dec = OldNeighborhoodDecoder(
            nb_neighbors=self.nbs_neighbors[1],
            input_size=10 * self.multiplier,
            features_global=[20 * self.multiplier, 40 * self.multiplier]
        )
        middle_decoder = OldMiddleDecoder(
            nb_neighbors=self.nbs_neighbors[1],
            radius=self.radii[1],
            split=10 * self.multiplier,
            neighborhood_decoder=middle_neigh_dec,
            feature_size=15 * self.multiplier,
            output_size=10 * self.multiplier
        )

        outside_neigh_dec = OldNeighborhoodDecoder(
            nb_neighbors=self.nbs_neighbors[0],
            input_size=10 * self.multiplier,
            features_global=[20 * self.multiplier, 40 * self.multiplier]
        )
        outside_decoder = OutsideDecoder(
            nb_neighbors=self.nbs_neighbors[0],
            radius=self.radii[0],
            neighborhood_decoder=outside_neigh_dec
        )

        decoders = [inside_decoder, middle_decoder, outside_decoder]

        return PointCloudVAE(
            encoders=encoders,
            decoders=decoders,
            lat_size=50 * self.multiplier
        )