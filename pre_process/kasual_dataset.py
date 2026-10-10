import torch
from torch.utils.data import DataLoader, Dataset


class KasualDataset(Dataset):
    """Input/target windows over the processor's input_ids.

    Args:
        input_ids: one list of token IDs per document (KasualProcessor output).
        window_size: tokens per training example (the book's max_length).
        stride: how far the window moves between examples; window_size means
            no overlap, smaller values overlap consecutive windows.
        return_document_ids: also return which document each input token is from.
    """

    def __init__(self, input_ids, window_size, stride, return_document_ids=False):
        self.tokens = torch.tensor([t for doc in input_ids for t in doc])
        self.document_ids = torch.tensor([i for i, doc in enumerate(input_ids) for _ in doc])
        self.window_size = window_size
        self.return_document_ids = return_document_ids
        # A window needs window_size inputs plus one more token for its last target.
        self.starts = range(0, len(self.tokens) - window_size, stride)
        if not self.starts:
            raise ValueError(f"{len(self.tokens)} tokens can't fill one window of {window_size}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        inputs = self.tokens[start:start + self.window_size]
        targets = self.tokens[start + 1:start + self.window_size + 1]  # inputs shifted by one
        if self.return_document_ids:
            return inputs, targets, self.document_ids[start:start + self.window_size]
        return inputs, targets

def create_dataloader(input_ids, window_size=256, stride=128, batch_size=4, shuffle=True,
                      drop_last=True, num_workers=0, return_document_ids=False):
    dataset = KasualDataset(input_ids, window_size, stride, return_document_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last,  # avoids a short last batch causing loss spikes
                      num_workers=num_workers)
