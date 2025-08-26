import os
import sys
import torch.nn as nn
from einops import rearrange, repeat

from models.Crossformer.attn import TwoStageAttentionLayer, AttentionLayer

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


class DecoderLayer(nn.Module):

    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num=10, factor=10):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, \
                                                     d_ff, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):

        batch = x.shape[0]

        # Apply self-attention to the decoder input.
        x = self.self_attention(x)

        # Reshape for cross-attention.
        x = x.reshape(-1, x.shape[2], x.shape[3])
        cross = cross.reshape(-1, cross.shape[2], cross.shape[3])

        # Apply cross-attention with the encoder output.
        tmp = self.cross_attention(x, cross, cross)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        # Reshape back to original batch and dimension structure.
        dec_output = dec_output.reshape(batch, -1, dec_output.shape[1], dec_output.shape[2])

        # Generate a prediction for the current scale.
        layer_predict = self.linear_pred(dec_output)
        layer_predict = layer_predict.reshape(layer_predict.shape[0], -1, layer_predict.shape[3])

        return dec_output, layer_predict


class Decoder(nn.Module):
    """
    The Decoder of Crossformer, which aggregates predictions from multiple scales
    to form the final forecast.
    """

    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout, \
                 router=False, out_seg_num=10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, \
                                                   out_seg_num, factor))

    def forward(self, x, cross):
        final_predict = None
        i = 0
        ts_d = x.shape[1]

        # Process through each decoder layer and accumulate the predictions.
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        # Reshape the final prediction to the standard [B, L, D] format.
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)

        return final_predict