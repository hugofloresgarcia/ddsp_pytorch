import yaml
import pathlib
from pathlib import Path
import librosa as li
import ddsp
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import os
from os import makedirs, path
import torch
import random
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


class args(Config):
    data_dir = "./data"
    cache_dir = "./cache"
    config_dir = "./configs"
    instruments = "decoder_instruments.txt"
    prop_train = 0.95

args.parse_args()


if __name__ == "__main__":

    with open(args.instruments) as inst_file:
        instrument_list = list(filter(None, [l.strip() for l in inst_file.readlines()]))

    single_inst_models = ["single-inst-decoder"]

    config_default = {
        "data": {
            "data_location": "",
            "extension": "wav"
        },
        "preprocess": {
            "sample_rate": 48000,
            "signal_length": 192000,
            "block_size": 512,
            "oneshot": False,
            "out_dir": ""
        },
        "model": {
            "name": "",
            "kwargs": {
                "hidden_size": 512,
                "n_harmonic": 64,
                "n_bands": 65,
                "sample_rate": 48000,
                "block_size": 512,
                "has_reverb": True
            },
        },
        "train": {
            "scales": [4096, 2048, 1024, 512, 256, 128],
            "overlap": .75,
            "batch": 16,
            "lr": 1.0e-3,
            "steps": 25000
        }
    }

    cache_dirs = []
    file_counts = []

    # individual instrument datasets
    for inst in instrument_list:

        # data and cache directories for config
        inst_name = path.basename(path.normpath(inst))
        inst_data_dir = os.path.join(args.data_dir, inst_name)
        inst_cache_dir = os.path.join(args.cache_dir, inst_name)

        cache_dirs.append(inst_cache_dir)

        # data splits
        train_dir = os.path.join(inst_data_dir, "train")
        val_dir = os.path.join(inst_data_dir, "validation")

        # create single-instrument dataset
        files_list = get_files(inst, "wav")
        file_counts.append(len(files_list))

        # withold for validation
        n_train = int(args.prop_train * len(files_list))
        random.shuffle(files_list)

        train_files = files_list[:n_train]
        val_files = files_list[n_train:]

        # copy files to target directories
        for f in train_files:
            audio_path = f.absolute()

            # define symlink path
            link_path = Path(train_dir) / audio_path.name
            link_path.parent.mkdir(parents=True, exist_ok=True)

            # make the symbolic link
            os.symlink(audio_path, link_path)

        for f in val_files:
            audio_path = f.absolute()

            # define symlink path
            link_path = Path(val_dir) / audio_path.name
            link_path.parent.mkdir(parents=True, exist_ok=True)

            # make the symbolic link
            os.symlink(audio_path, link_path)

        # create instrument-specific config files
        config_inst = config_default.copy()
        config_inst["data"]["data_location"] = str(inst_data_dir)
        config_inst["preprocess"]["out_dir"] = str(inst_cache_dir)

        # preprocess data in cache directory
        for partition in ["train", "validation"]:
            preprocess_folder(Path(inst_data_dir), partition, config_inst)

        # create one config file per single-instrument model and name/save accordingly
        for model in single_inst_models:

            filename = os.path.join(args.config_dir, inst_name + "_" + model + ".yaml")

            # set data dir, out dir, model
            config_model = config_inst.copy()
            config_model["model"]["name"] = model

            # write instrument/model specifications to config file
            with open(filename, 'w') as config_file:
                yaml.dump(config_model, config_file, default_flow_style=False)


