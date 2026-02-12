import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    noisy = [elem['noisy'] for elem in dataset_items]
    clean = [elem['clean'] for elem in dataset_items]

    lengths = torch.tensor([x.shape[0] for x in noisy], dtype=torch.long)  # T_i
    T_max = int(lengths.max())

    noisy = pad_sequence(noisy, batch_first=True, padding_value=0.0)
    clean = pad_sequence(clean, batch_first=True, padding_value=0.0)

    return {
        "noisy": noisy,
        "clean": clean,
    }
