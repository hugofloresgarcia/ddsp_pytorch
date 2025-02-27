import yaml
import pathlib
from pathlib import Path
import librosa as li
import ddsp
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
import librosa


def get_files(data_location, extension):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sample_rate, block_size, signal_length, 
               oneshot, **kwargs):
    x, sr = li.load(str(f), sample_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = ddsp.extract_pitch(x, sample_rate, block_size)
    loudness = ddsp.extract_loudness(x, sample_rate, block_size)
    mfcc = librosa.feature.mfcc(x, sr=sample_rate,  n_mfcc=30,
                        n_fft=1024, hop_length=block_size, 
                        fmin=20, fmax=8000, n_mels=128,).T

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness, mfcc


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l

def preprocess_folder(root_dir, partition, config):
    assert (root_dir /
            partition).exists(), f'{root_dir / partition} does not exist'
    files = get_files(data_location=root_dir / partition,
                      extension=config['data']['extension'])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []
    mfccs = []

    for f in pb:
        pb.set_description(str(f))
        x, p, l, m = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)
        mfccs.append(m)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)
    mfccs = np.stack(mfccs).astype(np.float32)

    out_dir = Path(config["preprocess"]["out_dir"]) / partition
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)
    np.save(path.join(out_dir, "mfccs.npy"), mfccs)

def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    root_dir = Path(config['data']['data_location'])
    partitions = ['train', 'validation']

    for partition in partitions:
        preprocess_folder(root_dir, partition, config)


if __name__ == "__main__":
    main()