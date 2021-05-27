from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        out_dir = Path(out_dir)
        self.signals = np.load(out_dir / "signals.npy")
        self.pitchs = np.load(out_dir / "pitchs.npy")
        self.loudness = np.load(out_dir / "loudness.npy")
        self.mfccs = np.load(out_dir / "mfccs.npy")

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx]).unsqueeze(-1)
        l = torch.from_numpy(self.loudness[idx]).unsqueeze(-1)
        m = torch.from_numpy(self.mfccs[idx])

        return {
            'sig': s,
            'pitch': p,
            'loudness': l,
            'mfcc': m
        }


class Datamodule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        out_dir = Path(self.config['preprocess']['out_dir'])
        self.train_data = Dataset(out_dir / 'train')
        self.val_data = Dataset(out_dir / 'validation')

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.config['train']['batch'],
                          shuffle=True,
                          drop_last=True,
                          collate_fn=dict_collate)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.config['train']['batch'],
                          shuffle=False,
                          collate_fn=dict_collate)


def dict_collate(records):
    batch = {}
    keys = list(records[0].keys())

    for k in keys:

        items = [r[k] for r in records]
        if isinstance(items[0], np.ndarray):
            items = np.stack(items)
            items = torch.from_numpy(items)
        if isinstance(items[0], torch.Tensor):
            items = torch.stack(items)

        batch[k] = items

    return batch