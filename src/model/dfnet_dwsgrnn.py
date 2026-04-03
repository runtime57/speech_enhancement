
from typing import Optional, Tuple

from torch import Tensor, nn

from src.model.conv_modules import make_block
from src.model.modules import DfOp, GroupedGRNN, GroupedLinear, Mask, convkxf, erb_filterbank
from src.utils.config_utils import stft_config


BLOCK_NAME = "xdws"


class EncoderGRNN_DWS(nn.Module):
    def __init__(self, params):
        super().__init__()
        p = params
        layer_width = p.conv_ch
        wf = p.conv_width_f
        assert p.nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        k = p.conv_k_enc
        groups = getattr(p, "conv_groups", 1)
        k0 = 1 if k == 1 and p.conv_lookahead == 0 else max(2, k)

        cl = 1 if p.conv_lookahead > 0 else 0
        self.erb_conv0 = make_block(
            BLOCK_NAME, 1, layer_width, p.nb_erb, k0, stride=1, groups=groups, lookahead=cl
        )
        cl = 1 if p.conv_lookahead > 1 else 0
        self.erb_conv1 = make_block(
            BLOCK_NAME,
            layer_width * wf**0,
            layer_width * wf**1,
            p.nb_erb // 2,
            k,
            stride=2,
            groups=groups,
            lookahead=cl,
        )
        cl = 1 if p.conv_lookahead > 2 else 0
        self.erb_conv2 = make_block(
            BLOCK_NAME,
            layer_width * wf**1,
            layer_width * wf**2,
            p.nb_erb // 4,
            k,
            stride=2,
            groups=groups,
            lookahead=cl,
        )
        self.erb_conv3 = make_block(
            BLOCK_NAME,
            layer_width * wf**2,
            layer_width * wf**2,
            p.nb_erb // 4,
            k,
            stride=1,
            groups=groups,
            lookahead=0,
        )
        self.df_conv0 = make_block(
            BLOCK_NAME, 2, layer_width, p.nb_df, k0, stride=1, groups=groups, lookahead=p.conv_lookahead
        )
        self.df_conv1 = make_block(
            BLOCK_NAME,
            layer_width,
            layer_width * wf**1,
            p.nb_df // 2,
            k,
            stride=2,
            groups=groups,
            lookahead=0,
        )

        self.erb_bins = p.nb_erb
        self.emb_dim = layer_width * p.nb_erb // 4 * wf**2
        self.df_fc_emb = GroupedLinear(
            layer_width * p.nb_df // 2, self.emb_dim, groups=p.lin_groups
        )
        self.emb_out_dim = p.emb_hidden_dim
        self.emb_n_layers = p.emb_num_layers
        self.gru_groups = p.gru_groups
        self.emb_grnn = GroupedGRNN(
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
        b, _, t, _ = feat_erb.shape
        e0 = self.erb_conv0(feat_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)
        c0 = self.df_conv0(feat_spec)
        c1 = self.df_conv1(c0)
        cemb = c1.permute(2, 0, 1, 3).reshape(t, b, -1)
        cemb = self.df_fc_emb(cemb)
        emb = e3.permute(2, 0, 1, 3).reshape(t, b, -1)
        emb = emb + cemb
        emb, _ = self.emb_grnn(emb)
        emb = emb.transpose(0, 1)
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder_DWS(nn.Module):
    def __init__(self, params):
        super().__init__()
        p = params
        layer_width = p.conv_ch
        wf = p.conv_width_f
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        groups = getattr(p, "conv_groups", 1)
        self.emb_width = layer_width * wf**2
        self.emb_dim = self.emb_width * (p.nb_erb // 4)
        self.fc_emb = nn.Sequential(
            GroupedLinear(
                p.emb_hidden_dim, self.emb_dim, groups=p.lin_groups, shuffle=p.group_shuffle
            ),
            nn.ReLU(inplace=True),
        )
        k = p.conv_k_dec
        pkwargs = {"k": 1, "f": 1, "batch_norm": True}
        self.conv3p = convkxf(layer_width * wf**2, self.emb_width, **pkwargs)
        self.conv2p = convkxf(layer_width * wf**2, layer_width * wf**2, **pkwargs)
        self.conv1p = convkxf(layer_width * wf**1, layer_width * wf**1, **pkwargs)
        self.conv0p = convkxf(layer_width, layer_width, **pkwargs)
        self.conv0_out = convkxf(layer_width, 1, fstride=1, k=k, act=nn.Sigmoid())

        self.convt3 = make_block(
            BLOCK_NAME,
            self.emb_width,
            layer_width * wf**2,
            p.nb_erb // 4,
            k,
            stride=1,
            groups=groups,
            use_deconv=False,
            lookahead=0,
        )
        self.convt2 = make_block(
            BLOCK_NAME,
            layer_width * wf**2,
            layer_width * wf**1,
            p.nb_erb // 2,
            k,
            stride=2,
            groups=groups,
            use_deconv=(p.conv_dec_mode == "transposed"),
            lookahead=0,
        )
        self.convt1 = make_block(
            BLOCK_NAME,
            layer_width * wf**1,
            layer_width * wf**0,
            p.nb_erb,
            k,
            stride=2,
            groups=groups,
            use_deconv=(p.conv_dec_mode == "transposed"),
            lookahead=0,
        )

    def forward(self, emb, e3, e2, e1, e0) -> Tensor:
        b, _, t, f8 = e3.shape
        emb = self.fc_emb(emb)
        emb = emb.view(b, t, -1, f8).transpose(1, 2)
        e3 = self.convt3(self.conv3p(e3) + emb)
        e2 = self.convt2(self.conv2p(e2) + e3)
        e1 = self.convt1(self.conv1p(e1) + e2)
        m = self.conv0_out(self.conv0p(e0) + e1)
        return m


class DfDecoderGRNN(nn.Module):
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
        self.df_grnn = GroupedGRNN(
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
        c, _ = self.df_grnn(emb.transpose(0, 1))
        c0 = self.df_convp(c0).transpose(1, 2)
        c = c.transpose(0, 1)
        alpha = self.df_fc_a(c)
        c = self.df_fc_out(c)
        c = c.view(b, t, self.df_order * 2, self.df_bins)
        c = c.add(c0).view(b, t, self.df_order, 2, self.df_bins).transpose(3, 4)
        return c, alpha


class DfNetDWSGRNN(nn.Module):
    def __init__(
        self,
        run_df: bool = True,
        train_mask: bool = True,
        conv_lookahead: int = 2,
        conv_k_enc: int = 2,
        conv_k_dec: int = 2,
        conv_ch: int = 64,
        conv_width_f: int = 1,
        conv_dec_mode: str = "transposed",
        conv_depthwise: bool = True,
        convt_depthwise: bool = True,
        emb_hidden_dim: int = 512,
        emb_num_layers: int = 3,
        df_hidden_dim: int = 512,
        df_num_layers: int = 2,
        gru_groups: int = 8,
        lin_groups: int = 8,
        group_shuffle: bool = True,
        dfop_method: str = "real_unfold",
        mask_pf: bool = False,
        conv_groups: int = 1,
    ):
        super().__init__()

        p = stft_config()
        self.sr = p.sr
        self.fft_size = p.fft_size
        self.hop_length = p.hop_length
        self.nb_erb = p.nb_erb
        self.nb_df = p.nb_df
        self.norm_tau = p.norm_tau
        self.lsnr_max = p.lsnr_max
        self.lsnr_min = p.lsnr_min
        self.min_nb_freqs = p.min_nb_freqs
        self.df_order = p.df_order
        self.df_lookahead = p.df_lookahead
        self.pad_mode = p.pad_mode

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
        self.conv_groups = conv_groups

        layer_width = self.conv_ch
        assert self.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        erb_inverse = erb_filterbank(inverse=True)
        self.freq_bins = self.fft_size // 2 + 1
        self.emb_dim = layer_width * self.nb_erb
        self.erb_bins = self.nb_erb

        self.enc = EncoderGRNN_DWS(self)
        self.erb_dec = ErbDecoder_DWS(self)
        self.mask = Mask(erb_inverse, post_filter=self.mask_pf)

        self.df_bins = self.nb_df
        self.df_dec = DfDecoderGRNN(self)
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
        feat_spec: Tensor,
        atten_lim: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feat_spec = feat_spec.transpose(1, 4).squeeze(4)
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)
        spec = self.mask(spec, m, atten_lim)
        df_coefs, df_alpha = self.df_dec(emb, c0)
        spec = self.df_op(spec, df_coefs, df_alpha)
        return spec, m, lsnr, df_alpha
