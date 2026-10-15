"""Shared experiment order for the full-recording DFNet/GRNN comparison."""

MODELS = [
    ("dfnet", "deepfilternet"),
    ("grnn", "deepfilternetgrnn"),
    ("dfnet_emb_128", "dfnet_emb_128"),
    ("dfnet_emb_256", "dfnet_emb_256"),
    ("dfnet_emb_384", "dfnet_emb_384"),
    ("grnn_emb_128", "grnn_emb_128"),
    ("grnn_emb_256", "grnn_emb_256"),
    ("grnn_emb_384", "grnn_emb_384"),
    ("grnn_rebalanced", "rebalancedgrnn"),
    ("grnn_rebalanced2", "rebalanced2grnn"),
    ("grnn_rebalanced3", "rebalanced3grnn"),
]

# Remaining report experiments; embedding-width ablations are intentionally omitted.
REPORT_MODELS = [
    ("grnn_rebalanced", "rebalancedgrnn"),
    ("grnn_rebalanced2", "rebalanced2grnn"),
    ("grnn_rebalanced3", "rebalanced3grnn"),
    ("dws", "dws"),
    ("mb", "mm"),
    ("conv", "conv"),
    ("xdws", "xdws"),
    ("xmb", "xmm"),
    ("xconv", "xconv"),
]
