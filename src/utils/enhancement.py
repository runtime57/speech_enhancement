from src.utils.config_utils import stft_config
from src.utils.df_utils import (audio_lengths_to_frame_lengths, exp_unit_norm,
                                istft_with_df_config, stft_with_df_config)
from src.utils.erb_utils import compute_erb_feats_from_stft


def prepare_enhancement_inputs(noisy, lengths):
    """Shared, length-aware features for training and evaluation."""
    p = stft_config()
    spectrum = stft_with_df_config(noisy, lengths)
    frames = audio_lengths_to_frame_lengths(lengths, p.fft_size, p.hop_length)
    packed = spectrum.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()
    normalized, _ = exp_unit_norm(packed[:, :, :, :p.nb_df], lengths=frames)
    erb = compute_erb_feats_from_stft(spectrum, lengths_frames=frames)
    return {"spec": packed, "feat_erb": erb, "feat_spec": normalized}, frames


def synthesize_enhancement(enhanced, lengths, output_length=None):
    packed = enhanced.squeeze(1).permute(0, 2, 1, 3).contiguous()
    return istft_with_df_config(packed, length=output_length, lengths=lengths)
