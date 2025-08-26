import torch
import torch.nn as nn
import math


class Time_branch_2d(nn.Module):

    def __init__(self, seq_R, freq, c_in, c_out, windows_size, use_time_features=True):
        super(Time_branch_2d, self).__init__()

        self.seq_R = seq_R
        self.c_out = c_out
        self.windows_size = windows_size
        self.use_time_features = use_time_features

        # Determine the dimension of time features based on frequency
        dim_time = 0
        if self.use_time_features:
            if freq == 't': dim_time = 5
            elif freq == 'h': dim_time = 4
            elif freq == 'd': dim_time = 3

        # separate processing for data and time features
        self.hidden_MLP_data = nn.Conv1d(
            in_channels=c_in * 1,  # Input channels for raw features only
            out_channels=c_in * c_out,
            kernel_size=windows_size, stride=1, groups=c_in)

        # separate processing for data and time features
        self.gate_data = nn.Conv1d(
            in_channels=c_in * 1,
            out_channels=c_in * 2 * c_out,
            kernel_size=1, stride=1, groups=c_in)

        if self.use_time_features and dim_time > 0:
            self.hidden_MLP_mark = nn.Conv1d(
                in_channels=c_in * dim_time,
                out_channels=c_in * c_out,
                kernel_size=windows_size, stride=1, groups=c_in)
            self.gate_mark = nn.Conv1d(
                in_channels=c_in * dim_time,
                out_channels=c_in * 2 * c_out,
                kernel_size=1, stride=1, groups=c_in)
        else:
            self.use_time_features = False

        # Gate Layer 2 (for historical information hid)
        self.gate_hid = nn.Conv1d(
            in_channels=c_in * c_out,
            out_channels=c_in * 2 * c_out,
            kernel_size=1, stride=1, groups=c_in)

        # Long-term information aggregation layer
        self.fc = nn.Conv1d(
            in_channels=c_in * c_out,
            out_channels=c_in * c_out,
            kernel_size=seq_R, stride=1, groups=c_in)

    def deal(self, x, x_mark):
        B, R, C, c_in, _ = x.shape

        # Process data x
        x_supply = torch.zeros(B, self.windows_size, C, c_in, 1).to(x.device)
        x_all = torch.cat([x_supply, x], dim=1)
        x_all = x_all.permute(0, 2, 3, 4, 1).reshape(B * C, c_in, R + self.windows_size)
        H_data = self.hidden_MLP_data(x_all[:, :, :-1])

        # Conditionally process time features x_mark
        if self.use_time_features and x_mark is not None:
            c_time = x_mark.shape[-1]
            mark_supply = torch.zeros(B, self.windows_size, C, c_in, c_time).to(x.device)
            mark_all = torch.cat([mark_supply, x_mark], dim=1)
            mark_all = mark_all.permute(0, 2, 3, 4, 1).reshape(B * C, c_in * c_time, R + self.windows_size)
            H_mark = self.hidden_MLP_mark(mark_all[:, :, :-1])
            hid = H_data + H_mark # Combine historical information from data and time features
        else:
            hid = H_data # Historical information comes from data only

        hid = hid.reshape(B, C, c_in, self.c_out, R).permute(0, 4, 1, 2, 3)
        return hid

    def gated_unit(self, x, x_mark, hid):
        B, R, C, c_in, _ = x.shape

        # Gate for data x
        x_reshaped = x.reshape(B * R * C, c_in, 1)
        gate_out_data = self.gate_data(x_reshaped)

        # Conditional Gate for x_mark
        if self.use_time_features and x_mark is not None:
            c_time = x_mark.shape[-1]
            x_mark_reshaped = x_mark.reshape(B * R * C, c_in * c_time, 1)
            gate_out_mark = self.gate_mark(x_mark_reshaped)
            x_embed_input = gate_out_data + gate_out_mark
        else:
            x_embed_input = gate_out_data

        # Gate for history hid
        hid_reshaped = hid.reshape(B * R * C, c_in * self.c_out, 1)
        gate_out_hid = self.gate_hid(hid_reshaped)

        # Combine input gate and history gate
        x_embed = x_embed_input + gate_out_hid
        x_embed = x_embed.reshape(B, R, C, c_in, -1)

        sigmod_gate, tanh_gate = torch.split(x_embed, self.c_out, dim=-1)
        sigmod_gate = torch.sigmoid(sigmod_gate)
        tanh_gate = torch.tanh(tanh_gate)

        hid = hid * sigmod_gate + (1 - sigmod_gate) * tanh_gate
        return hid

    def forward(self, x, x_mark):
        B, R, C, c_in, _ = x.shape
        out = self.deal(x, x_mark)
        out = self.gated_unit(x, x_mark, out)
        out = self.fc(out.permute(0, 2, 3, 4, 1).reshape(
            B * C, c_in * self.c_out, R)).reshape(B, C, c_in, self.c_out)
        return out


class Short_Term_Extractor(nn.Module):

    def __init__(self, seq_R, freq, c_in, c_out, period, use_time_features=True):
        super(Short_Term_Extractor, self).__init__()

        self.seq_R = seq_R
        self.c_out = c_out
        self.period = period
        self.use_time_features = use_time_features

        dim_time = 0
        if self.use_time_features:
            if freq == 't': dim_time = 5
            elif freq == 'h': dim_time = 4
            elif freq == 'd': dim_time = 3

        # separate for data and time features
        self.fc_row_data = nn.Conv1d(
            in_channels=c_in * 1,
            out_channels=c_in * c_out,
            kernel_size=period, stride=1, groups=c_in)

        if self.use_time_features and dim_time > 0:
            self.fc_row_mark = nn.Conv1d(
                in_channels=c_in * dim_time,
                out_channels=c_in * c_out,
                kernel_size=period, stride=1, groups=c_in)
        else:
            self.use_time_features = False

        # Column aggregation layer
        self.fc_col = nn.Conv1d(
            in_channels=c_in * c_out,
            out_channels=c_in * c_out,
            kernel_size=seq_R, stride=1, groups=c_in)

    def forward(self, x, x_mark):
        B, R, C, c_in, _ = x.shape

        # Process data x
        x_reshaped = x.permute(0, 1, 3, 4, 2).reshape(B * R, c_in, C)
        out_data = self.fc_row_data(x_reshaped)

        if self.use_time_features and x_mark is not None:
            c_time = x_mark.shape[-1]
            x_mark_reshaped = x_mark.permute(0, 1, 3, 4, 2).reshape(B * R, c_in * c_time, C)
            out_mark = self.fc_row_mark(x_mark_reshaped)
            out = out_data + out_mark
        else:
            out = out_data

        out = out.reshape(B, R, c_in * self.c_out)

        # Column aggregation
        out = self.fc_col(out.permute(0, 2, 1)).reshape(
            B, c_in, 1, self.c_out).repeat(1, 1, self.period, 1)
        return out.permute(0, 2, 1, 3)


class Time_branch(nn.Module):

    def __init__(self, seq_R, freq, c_in, c_out, windows_size,
                 period, pred_R, need_short=True, use_time_features=True):
        super(Time_branch, self).__init__()

        self.need_short = need_short
        self.pred_R = pred_R
        # Instantiate the long-term branch
        self.long_term_branch = Time_branch_2d(seq_R, freq, c_in, c_out, windows_size, use_time_features)

        if self.need_short:
            # Instantiate the short-term branch
            self.short_term_branch = Short_Term_Extractor(seq_R,
                                                         freq, c_in, c_out, period, use_time_features)
            fc_in_channels = c_in * 2 * c_out
        else:
            fc_in_channels = c_in * c_out

        self.fc = nn.Conv1d(
            in_channels=fc_in_channels,
            out_channels=c_in * pred_R,
            kernel_size=1, stride=1, groups=c_in)

    def forward(self, x, x_mark=None):
        B, R, C, c_in = x.shape
        x = x.unsqueeze(-1)

        if x_mark is not None:
            x_mark = x_mark.unsqueeze(-2).repeat(1, 1, 1, c_in, 1)

        # Get output from the long-term branch
        out_long_term = self.long_term_branch(x, x_mark)

        if self.need_short:
            # Get output from the short-term branch and concatenate
            out_short_term = self.short_term_branch(x, x_mark)
            out_all = torch.cat([out_short_term, out_long_term],
                                dim=-1).reshape(B * C, -1, 1)
        else:
            out_all = out_long_term.reshape(B * C, -1, 1)

        # Final projection to get the prediction
        out_all = self.fc(out_all).reshape(B, C, c_in, self.pred_R).permute(
            0, 3, 1, 2).reshape(B, -1, c_in)
        return out_all


class Time_Branch_Model(nn.Module):

    def __init__(self, configs):
        super(Time_Branch_Model, self).__init__()

        self.configs = configs
        self.freq = configs.freq
        self.period = configs.Time_branch_period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seq_R = int(math.ceil(self.configs.seq_len / self.period))
        self.pred_R = int(math.ceil(self.pred_len / self.period))
        self.c_in = configs.enc_in
        self.d_model = configs.d_model
        self.norm = 1

        # Determine at initialization whether to build components for time features
        dim_time = 0
        if self.freq == 't': dim_time = 5
        elif self.freq == 'h': dim_time = 4
        elif self.freq == 'd': dim_time = 3
        self.use_time_features = dim_time > 0

        self.time_branch = Time_branch(self.seq_R, self.freq, self.c_in,
                                       self.d_model, self.seq_R - 1, self.period,
                                       self.pred_R, use_time_features=self.use_time_features)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        B, L, c_in = x_enc.shape

        if x_mark_enc is not None and torch.all(x_mark_enc == 0):
            x_mark_enc = None

        # Normalization
        if self.norm == 1:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1,
                                         keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Reshape 1D sequence to 2D
        x_enc = x_enc.reshape(B, self.seq_R, self.period, c_in)

        # Reshape time features only if they are valid
        if x_mark_enc is not None:
            c_time = x_mark_enc.shape[-1]
            x_mark_enc = x_mark_enc.reshape(B, self.seq_R, self.period, c_time)

        # Pass data and optional time features to the main model
        output = self.time_branch(x_enc, x_mark_enc)

        # Denormalization
        if self.norm == 1:
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))
            output = output + (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))

        return output