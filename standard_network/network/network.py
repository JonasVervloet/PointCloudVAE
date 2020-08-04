import torch
from torch import nn

from standard_network.encoder.middle_encoder import MiddleEncoder
from standard_network.encoder.outside_encoder import OutsideEncoder
from standard_network.encoder.inside_encoder import InsideEncoder

from standard_network.decoder.middle_decoder import MiddleDecoder
from standard_network.decoder.outside_decoder import OutsideDecoder
from standard_network.decoder.inside_decoder import InsideDecoder


class PointCloudAE(nn.Module):
    def __init__(self, encoders, decoders):
        super(PointCloudAE, self).__init__()

        assert(len(encoders) == 3)
        assert(isinstance(encoders[0], OutsideEncoder))
        assert(isinstance(encoders[1], MiddleEncoder))
        assert(isinstance(encoders[2], InsideEncoder))

        assert(len(decoders) == 3)
        assert(isinstance(decoders[0], InsideDecoder))
        assert(isinstance(decoders[1], MiddleDecoder))
        assert(isinstance(decoders[2], OutsideDecoder))

        self.outside_encoder = encoders[0]
        self.middle_encoder = encoders[1]
        self.inside_encoder = encoders[2]

        self.inside_decoder = decoders[0]
        self.middle_decoder = decoders[1]
        self.outside_decoder = decoders[2]

    def forward(self, batch_obj):
        out_enc_points, out_enc_features, out_enc_batch = self.outside_encoder(
            batch_obj.pos, batch_obj.batch
        )
        middle_enc_points, middle_enc_features, middle_enc_batch = self.middle_encoder(
            out_enc_points, out_enc_features, out_enc_batch
        )
        in_enc_features = self.inside_encoder(
            middle_enc_points, middle_enc_features, middle_enc_batch
        )

        in_dec_points, in_dec_features, in_dec_batch = self.inside_decoder(
            in_enc_features
        )
        middle_dec_points, middle_dec_features, middle_dec_batch = self.middle_decoder(
            in_dec_points, in_dec_features, in_dec_batch
        )
        out_dec_points, out_dec_batch = self.outside_decoder(
            middle_dec_points, middle_dec_features, middle_dec_batch
        )

        input_points_list = [batch_obj.pos, out_enc_points, middle_enc_points]
        input_batch_list = [batch_obj.batch, out_enc_batch, middle_enc_batch]

        output_points_list = [out_dec_points, middle_dec_points, in_dec_points]
        output_batch_list = [out_dec_batch, middle_dec_batch, in_dec_batch]

        return input_points_list, input_batch_list, output_points_list, output_batch_list, in_enc_features

