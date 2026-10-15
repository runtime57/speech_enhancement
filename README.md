# DFNet and FastGRNN listening demo

[Open the demo](https://runtime57.github.io/speech_enhancement/).

The three existing noisy/clean recording pairs are processed with **DFNet,
FastGRNN emb=128/256/512, and Rebalanced-1/2/3**. All seven models use the
complete checkpoint from **epoch 50** of the full-recording training protocol.
The weights contain inference state and model/STFT configuration, without optimizer state.
The previous convolution variants are excluded from this demo.

Audio is mono 16 kHz, PCM16. Enhancement uses FP32 on CPU, batch 1, with the
current length-aware feature extraction and reconstruction. There is no
per-model loudness normalization. All outputs have the same sample count as
their noisy input and clean reference.

Each example has nine spectrograms (noisy, clean and seven enhanced signals).
The FFT is 512 samples, Hann window, hop 128 samples. **All variants of an example
share a single peak reference and fixed −80…0 dB color limits**; the reference
is the maximum STFT magnitude across its nine signals. This makes changes in
noise level visually comparable. Time and frequency axes and a color scale are included.

`demo_manifest.json` records checkpoint/run identities, exact parameter counts,
SHA-256 hashes of the inference weights and original training checkpoints,
inference configuration, and hashes of every displayed audio/image asset.
FastGRNN-384 is not included because no trained checkpoint is available.

To rebuild from the included weights, in the repository's Python environment:

```bash
pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python refresh_demo.py
python -m http.server 8000
```

Open `http://localhost:8000`. To import fresh inference exports from the training
checkout, use `python refresh_demo.py --weights-root /path/to/outputs/paper_exports`.
The directory must contain `model_epoch50.pth` under each of `dfnet`, `grnn`,
`grnn_emb_128`, `grnn_emb_256`, `grnn_rebalanced`, `grnn_rebalanced2`, and `grnn_rebalanced3`.

[Full evaluation and CPU performance report](https://github.com/runtime57/speech_enhancement/blob/main/paper/results/cpu_benchmark_20261015.md).
