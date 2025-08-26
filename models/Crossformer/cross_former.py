import torch
import torch.nn as nn
from einops import rearrange, repeat

from math import ceil

from models.Crossformer.cross_decoder import Decoder
from models.Crossformer.cross_embed import DSW_embedding
from models.Crossformer.cross_encoder import Encoder


class Crossformer(nn.Module):
    def __init__(self, config, win_size=2,
                 factor=10, d_model=256, d_ff=512, n_heads=4, e_layers=3,
                 dropout=0, baseline=False, device=torch.device('cuda:0')):
        super(Crossformer, self).__init__()
        d_model = config.d_model
        d_ff = config.d_ff
        dropout = config.dropout
        self.data_dim = config.enc_in
        self.in_len = config.seq_len
        self.out_len = config.pred_len
        self.seg_len = config.seg_len
        e_layers=config.e_layers
        self.merge_win = config.win_size

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(self.seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_in_len // self.seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth=1, \
                               dropout=dropout, in_seg_num=(self.pad_in_len // self.seg_len), factor=factor)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, (self.pad_out_len // self.seg_len), d_model))
        self.decoder = Decoder(self.seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                               out_seg_num=(self.pad_out_len // self.seg_len), factor=factor)

    def forward(self, x_enc=None, x_mark_enc=None):
        x_seq = x_enc
        if (self.baseline):
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        return base + predict_y[:, :self.out_len, :]
