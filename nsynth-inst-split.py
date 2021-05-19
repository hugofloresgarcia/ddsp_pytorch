""" 
creates a view of nsynth where audio files 
are grouped by instrument. 
"""

import os
from pathlib import Path
from typing import List

import torch
import json
from nussl import AudioSignal
from effortless_config import Config


class args(Config):
    root_dir = ""
    target_dir = ""


class NSynth(torch.utils.data.Dataset):
    def __init__(self,
                 root: Path,
                 sample_rate: int,
                 instrument_subset: List[str] = None):
        self.root = root
        self.audio_path = self.root / 'audio'
        self.sample_rate = sample_rate

        # grab the records
        with open(self.root / 'examples.json') as f:
            records = json.load(f)

        # convert records to list of dicts
        records = [dict(name=key, **value) for key, value in records.items()]
        self.records = records

        if instrument_subset is not None:
            self.records = self.filter_by_instruments(instrument_subset)

        self.classlist = list(set(r['instrument_str'] for r in self.records))

    def filter_by_instruments(self, instruments: List[str]):
        # filter by instrument subset
        return [r for r in self.records if r['instrument_str'] in instruments]

    def __len__(self):
        return len(self.records)

    def get_audio_path(self, data: dict):
        audio_path = Path(self.audio_path / data['name']).with_suffix('.wav')
        return audio_path

    def __getitem__(self, idx: int):
        data = self.records[idx]
        item = {}

        # load the audio to a nussl audiosignal
        audio_path = Path(self.audio_path / data['name']).with_suffix('.wav')
        item['audio'] = AudioSignal(path_to_input_file=audio_path, )
        item['audio_path'] = str(audio_path)
        item['instrument'] = data['instrument_str']
        item['data'] = data

        return item


def create_inst_view(root_dir: Path, target_dir: Path):
    # load the records
    dataset = NSynth(root_dir, 16000)

    # get the list of classes in the dataset
    classlist = dataset.classlist

    for inst in classlist:
        # get the subset of entries for this instrument
        records = dataset.filter_by_instruments([inst])

        for record in records:
            # get the audio path
            audio_path = dataset.get_audio_path(record)

            # define symlink path
            link_path = target_dir / inst / audio_path.name
            link_path.parent.mkdir(parents=True, exist_ok=True)

            # make the symbolic link
            os.symlink(audio_path, link_path)


if __name__ == "__main__":
    args.parse_args()
    create_inst_view(Path(args.root_dir), Path(args.target_dir))