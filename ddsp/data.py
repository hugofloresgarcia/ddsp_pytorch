from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def rglob_np_array(pdir: Path, name: str):
    files = pdir.glob(f'**/{name}.npy')
    return np.concatenate([np.load(f) for f in files], axis=0)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        out_dir = Path(out_dir)
        self.signals = rglob_np_array(out_dir, 'signals')
        self.pitchs = rglob_np_array(out_dir, 'pitchs')
        self.loudness = rglob_np_array(out_dir, 'loudness')
        self.mfccs = rglob_np_array(out_dir, 'mfccs')

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx]).unsqueeze(-1)
        l = torch.from_numpy(self.loudness[idx]).unsqueeze(-1)
        m = torch.from_numpy(self.mfccs[idx])[:-1, :]

        return {
            'sig': s,
            'f0': p,
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
                          drop_last=True,
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