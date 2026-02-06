from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from modules import DfOp, GroupedGRU, GroupedLinear, Mask, convkxf, erb_fb


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        p = params
        layer_width = p.conv_ch
        wf = p.conv_width_f
        assert p.nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        k = p.conv_k_enc
        kwargs = {"batch_norm": True, "depthwise": p.conv_depthwise}
        k0 = 1 if k == 1 and p.conv_lookahead == 0 else max(2, k)
        cl = 1 if p.conv_lookahead > 0 else 0
        self.erb_conv0 = convkxf(1, layer_width, k=k0, fstride=1, lookahead=cl, **kwargs)
        cl = 1 if p.conv_lookahead > 1 else 0
        self.erb_conv1 = convkxf(
            layer_width * wf**0, layer_width * wf**1, k=k, lookahead=cl, **kwargs
        )
        cl = 1 if p.conv_lookahead > 2 else 0
        self.erb_conv2 = convkxf(
            layer_width * wf**1, layer_width * wf**2, k=k, lookahead=cl, **kwargs
        )
        self.erb_conv3 = convkxf(layer_width * wf**2, layer_width * wf**2, k=k, fstride=1, **kwargs)
        self.df_conv0 = convkxf(
            2, layer_width, fstride=1, k=k0, lookahead=p.conv_lookahead, **kwargs
        )
        self.df_conv1 = convkxf(layer_width, layer_width * wf**1, k=k, **kwargs)
        self.erb_bins = p.nb_erb
        self.emb_dim = layer_width * p.nb_erb // 4 * wf**2
        self.df_fc_emb = GroupedLinear(
            layer_width * p.nb_df // 2, self.emb_dim, groups=p.lin_groups
        )
        self.emb_out_dim = p.emb_hidden_dim
        self.emb_n_layers = p.emb_num_layers
        self.gru_groups = p.gru_groups
        self.emb_gru = GroupedGRU(
            self.emb_dim,
            self.emb_out_dim,
            num_layers=p.emb_num_layers,
            batch_first=False,
            groups=p.gru_groups,
            shuffle=p.group_shuffle,
            add_outputs=True,
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encodes erb; erb should be in dB scale + normalized; Fe are number of erb bands.
        # erb: [B, 1, T, Fe]
        # spec: [B, 2, T, Fc]
        b, _, t, _ = feat_erb.shape
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/4]
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc]
        cemb = c1.permute(2, 0, 1, 3).reshape(t, b, -1)  # [T, B, C * Fc/4]
        cemb = self.df_fc_emb(cemb)  # [T, B, C * F/4]
        emb = e3.permute(2, 0, 1, 3).reshape(t, b, -1)  # [T, B, C * F/4]
        emb = emb + cemb
        emb, _ = self.emb_gru(emb)
        emb = emb.transpose(0, 1)  # [B, T, C * F/4]
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        p = params
        layer_width = p.conv_ch
        wf = p.conv_width_f
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        self.emb_width = layer_width * wf**2
        self.emb_dim = self.emb_width * (p.nb_erb // 4)
        self.fc_emb = nn.Sequential(
            GroupedLinear(
                p.emb_hidden_dim, self.emb_dim, groups=p.lin_groups, shuffle=p.group_shuffle
            ),
            nn.ReLU(inplace=True),
        )
        k = p.conv_k_dec
        kwargs = {"k": k, "batch_norm": True, "depthwise": p.conv_depthwise}
        tkwargs = {
            "k": k,
            "batch_norm": True,
            "depthwise": p.convt_depthwise,
            "mode": p.conv_dec_mode,
        }
        pkwargs = {"k": 1, "f": 1, "batch_norm": True}
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = convkxf(layer_width * wf**2, self.emb_width, **pkwargs)
        self.convt3 = convkxf(self.emb_width, layer_width * wf**2, fstride=1, **kwargs)
        self.conv2p = convkxf(layer_width * wf**2, layer_width * wf**2, **pkwargs)
        self.convt2 = convkxf(layer_width * wf**2, layer_width * wf**1, **tkwargs)
        self.conv1p = convkxf(layer_width * wf**1, layer_width * wf**1, **pkwargs)
        self.convt1 = convkxf(layer_width * wf**1, layer_width * wf**0, **tkwargs)
        self.conv0p = convkxf(layer_width, layer_width, **pkwargs)
        self.conv0_out = convkxf(layer_width, 1, fstride=1, k=k, act=nn.Sigmoid())

    def forward(self, emb, e3, e2, e1, e0) -> Tensor:
        # Estimates erb mask
        b, _, t, f8 = e3.shape
        emb = self.fc_emb(emb)
        emb = emb.view(b, t, -1, f8).transpose(1, 2)  # [B, C*8, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C*4, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C*2, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m


class DfDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        p = params
        layer_width = p.conv_ch
        self.emb_dim = p.emb_hidden_dim

        self.df_n_hidden = p.df_hidden_dim
        self.df_n_layers = p.df_num_layers
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_lookahead = p.df_lookahead
        self.gru_groups = p.gru_groups

        self.df_convp = convkxf(
            layer_width, self.df_order * 2, k=1, f=1, complex_in=True, batch_norm=True
        )
        self.df_gru = GroupedGRU(
            p.emb_hidden_dim,
            self.df_n_hidden,
            num_layers=self.df_n_layers,
            batch_first=False,
            groups=p.gru_groups,
            shuffle=p.group_shuffle,
            add_outputs=True,
        )
        self.df_fc_out = nn.Sequential(
            nn.Linear(self.df_n_hidden, self.df_bins * self.df_order * 2), nn.Tanh()
        )
        self.df_fc_a = nn.Sequential(nn.Linear(self.df_n_hidden, 1), nn.Sigmoid())

    def forward(self, emb: Tensor, c0: Tensor) -> Tuple[Tensor, Tensor]:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb.transpose(0, 1))  # [T, B, H], H: df_n_hidden
        c0 = self.df_convp(c0).transpose(1, 2)  # [B, T, O*2, F]
        c = c.transpose(0, 1)  # [B, T, H]
        alpha = self.df_fc_a(c)  # [B, T, 1]
        c = self.df_fc_out(c)  # [B, T, F*O*2], O: df_order
        c = c.view(b, t, self.df_order * 2, self.df_bins)  # [B, T, O*2, F]
        c = c.add(c0).view(b, t, self.df_order, 2, self.df_bins).transpose(3, 4)  # [B, T, O, F, 2]
        return c, alpha


class DfNet(nn.Module):
    def __init__(
        self,
        df_state: Optional[DF] = None,
        run_df: bool = True,
        train_mask: bool = True,
        sr: int = 48_000,
        fft_size: int = 960,
        hop_size: int = 480,
        nb_erb: int = 32,
        nb_df: int = 96,
        norm_tau: float = 1.0,
        lsnr_max: int = 35,
        lsnr_min: int = -15,
        min_nb_freqs: int = 2,
        df_order: int = 5,
        df_lookahead: int = 0,
        pad_mode: str = "input",
        conv_lookahead: int = 0,
        conv_k_enc: int = 2,
        conv_k_dec: int = 1,
        conv_ch: int = 16,
        conv_width_f: int = 1,
        conv_dec_mode: str = "transposed",
        conv_depthwise: bool = True,
        convt_depthwise: bool = True,
        emb_hidden_dim: int = 256,
        emb_num_layers: int = 1,
        df_hidden_dim: int = 256,
        df_num_layers: int = 3,
        gru_groups: int = 1,
        lin_groups: int = 1,
        group_shuffle: bool = True,
        dfop_method: str = "real_unfold",
        mask_pf: bool = False,
    ):
        super().__init__()
        self.sr = sr
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.nb_erb = nb_erb
        self.nb_df = nb_df
        self.norm_tau = norm_tau
        self.lsnr_max = lsnr_max
        self.lsnr_min = lsnr_min
        self.min_nb_freqs = min_nb_freqs
        self.df_order = df_order
        self.df_lookahead = df_lookahead
        self.pad_mode = pad_mode
        self.conv_lookahead = conv_lookahead
        self.conv_k_enc = conv_k_enc
        self.conv_k_dec = conv_k_dec
        self.conv_ch = conv_ch
        self.conv_width_f = conv_width_f
        self.conv_dec_mode = conv_dec_mode
        self.conv_depthwise = conv_depthwise
        self.convt_depthwise = convt_depthwise
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_num_layers = emb_num_layers
        self.df_hidden_dim = df_hidden_dim
        self.df_num_layers = df_num_layers
        self.gru_groups = gru_groups
        self.lin_groups = lin_groups
        self.group_shuffle = group_shuffle
        self.dfop_method = dfop_method
        self.mask_pf = mask_pf
        self.run_df = run_df
        self.train_mask = train_mask

        layer_width = self.conv_ch
        assert self.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        if df_state is None:
            df_state = DF(
                sr=self.sr,
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                nb_bands=self.nb_erb,
            )
        erb_inverse = erb_fb(df_state.erb_widths(), self.sr, inverse=True)
        self.freq_bins = self.fft_size // 2 + 1
        self.emb_dim = layer_width * self.nb_erb
        self.erb_bins = self.nb_erb
        self.enc = Encoder(self)
        self.erb_dec = ErbDecoder(self)
        self.mask = Mask(erb_inverse, post_filter=self.mask_pf)

        self.df_bins = self.nb_df
        self.df_dec = DfDecoder(self)
        self.df_op = DfOp(
            self.nb_df,
            self.df_order,
            self.df_lookahead,
            freq_bins=self.freq_bins,
            method=self.dfop_method,
        )

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,  # Not used, take spec modified by mask instead
        atten_lim: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # re/im into channel axis
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)
        spec = self.mask(spec, m, atten_lim)
        df_coefs, df_alpha = self.df_dec(emb, c0)
        spec = self.df_op(spec, df_coefs, df_alpha)
        return spec, m, lsnr, df_alpha
