import math
import torch
from torch import nn

EPS = 1e-12


class DeepFilterNetSTFTLoss(nn.Module):
    """
    Loss from DeepFilterNet (arXiv:2110.05588), but computed directly on STFT tensors.

    Inputs:
      stft_pred   : predicted/enhanced complex STFT (Y)
      stft_target : clean target complex STFT (S)
      stft_noisy  : mixture/noisy complex STFT (X)  (optional; needed for L_alpha)
      df_alpha    : DF gate alpha in [0,1] per frame (optional; needed for L_alpha)

    The paper loss:
      L = L_spec + 0.05 * L_alpha
      L_spec = |||Y|^c - |S|^c||^2 + || |Y|^c e^{jφY} - |S|^c e^{jφS} ||^2
      c = 0.6
    """

    def __init__(
        self,
        sr: int = 16_000,
        n_fft: int = 960,
        compression: float = 0.6,
        f_df_hz: float = 5_000.0,
        lambda_spec: float = 1.0,
        lambda_alpha: float = 0.05,
        reduce: str = "mean",  # "mean" or "sum"
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.c = float(compression)
        self.f_df_hz = float(f_df_hz)
        self.lambda_spec = float(lambda_spec)
        self.lambda_alpha = float(lambda_alpha)
        assert reduce in ("mean", "sum")
        self.reduce = reduce

    # ---------- helpers ----------
    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
          - complex tensor [...], dtype complex
          - real/imag packed tensor [..., 2] (last dim = 2)
        Return complex tensor with same leading dims.
        """
        if torch.is_complex(x):
            return x
        if x.size(-1) != 2:
            raise ValueError("Expected complex tensor or real/imag packed tensor with last dim=2.")
        return torch.view_as_complex(x.contiguous())

    @staticmethod
    def _unit_phase(z: torch.Tensor) -> torch.Tensor:
        """z / |z| with eps for stability (keeps gradients nicer than angle())"""
        mag = torch.abs(z).clamp_min(EPS)
        return z / mag

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean() if self.reduce == "mean" else x.sum()

    # ---------- loss terms ----------
    def _L_spec(self, Y: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Implements Eq. (5) on complex STFT tensors.
        Y,S: complex [..., F, T] or [B,F,T] (any leading dims ok)
        """
        magY = torch.abs(Y).clamp_min(EPS)
        magS = torch.abs(S).clamp_min(EPS)

        Yc = magY.pow(self.c)
        Sc = magS.pow(self.c)

        # magnitude term: || |Y|^c - |S|^c ||^2
        term_mag = self._reduce((Yc - Sc).pow(2))

        # phase-aware term: || |Y|^c e^{jφY} - |S|^c e^{jφS} ||^2
        Ycplx = Yc * self._unit_phase(Y)
        Scplx = Sc * self._unit_phase(S)
        diff = Ycplx - Scplx
        term_phase = self._reduce(diff.real.pow(2) + diff.imag.pow(2))

        return term_mag + term_phase

    def _L_alpha(self, X: torch.Tensor, S: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Implements Eq. (7).
        X,S: complex [B,F,T] (mixture and clean)
        alpha: [B,T] or [B,T,1] (DF gate)
        LSNR computed using bins up to f_df_hz.
        """
        if alpha.dim() == 3 and alpha.size(-1) == 1:
            alpha = alpha.squeeze(-1)
        if alpha.dim() != 2:
            raise ValueError("alpha should have shape [B,T] or [B,T,1].")

        B, F, T = X.shape

        # number of bins up to f_df_hz
        # STFT bins correspond to frequencies f = bin * sr / n_fft
        f_df_bin = int(math.floor(self.f_df_hz * self.n_fft / self.sr)) + 1
        f_df_bin = max(1, min(F, f_df_bin))

        S_low = S[:, :f_df_bin, :]          # [B, fdf, T]
        N_low = (X - S)[:, :f_df_bin, :]    # noise estimate

        Es = (torch.abs(S_low).pow(2)).sum(dim=1)  # [B, T]
        En = (torch.abs(N_low).pow(2)).sum(dim=1).clamp_min(EPS)  # [B, T]

        lsnr = 10.0 * torch.log10((Es / En).clamp_min(EPS))  # [B, T]

        mask_low = (lsnr < -10.0).to(alpha.dtype)   # want alpha ~ 0
        mask_high = (lsnr > -5.0).to(alpha.dtype)   # want alpha ~ 1

        # Eq (7)
        return self._reduce((alpha * mask_low).pow(2) + ((1.0 - alpha) * mask_high).pow(2))

    # ---------- forward ----------
    def forward(
        self,
        stft_pred: torch.Tensor,      # Y
        stft_target: torch.Tensor,    # S
        stft_noisy: torch.Tensor | None = None,  # X
        df_alpha: torch.Tensor | None = None,
        **batch,
    ):
        # allow batch dict aliases
        if stft_noisy is None:
            stft_noisy = batch.get("stft_noisy", batch.get("stft_mix", batch.get("stft_input", None)))
        if df_alpha is None:
            df_alpha = batch.get("df_alpha", None)

        Y = self._to_complex(stft_pred)
        S = self._to_complex(stft_target)

        # accept either [B,T,F] complex OR [B,F,T] complex; normalize to [B,F,T]
        if Y.dim() == 3 and Y.shape[1] != S.shape[1]:
            # don't guess; better to be explicit
            pass

        # Common convention in audio: [B, F, T]. If you have [B, T, F], transpose before calling.
        if Y.dim() != 3 or S.dim() != 3:
            raise ValueError("Expected STFT tensors of shape [B,F,T] (complex) or [B,F,T,2] (re/im packed).")

        L_spec = self._L_spec(Y, S)
        loss = self.lambda_spec * L_spec
        out = {"loss": loss, "L_spec": L_spec}

        if stft_noisy is not None and df_alpha is not None:
            X = self._to_complex(stft_noisy)
            if X.dim() != 3:
                raise ValueError("stft_noisy should be [B,F,T] (complex) or [B,F,T,2].")

            # Align T if needed (crop to min)
            T = min(X.size(-1), S.size(-1), Y.size(-1))
            X = X[..., :T]
            S_ = S[..., :T]

            alpha = df_alpha
            if alpha.dim() == 3 and alpha.size(-1) == 1:
                alpha = alpha.squeeze(-1)
            alpha = alpha[..., :T].clamp(0.0, 1.0)  # [B,T]

            L_alpha = self._L_alpha(X, S_, alpha)
            loss = loss + self.lambda_alpha * L_alpha
            out["L_alpha"] = L_alpha
            out["loss"] = loss

        return out