import ddsp
import torch
import torch.nn as nn
import yaml
from effortless_config import Config
from os import path, makedirs, system
from ddsp.models.decoder import DDSPDecoder
import soundfile as sf
from preprocess import get_files

torch.set_grad_enabled(False)


class args(Config):
    RUN = None
    DATA = False
    OUT_DIR = "export"
    REALTIME = False


args.parse_args()
makedirs(args.OUT_DIR, exist_ok=True)


class ScriptDDSP(torch.nn.Module):
    def __init__(self, ddsp, mean_loudness, std_loudness, realtime):
        super().__init__()
        self.ddsp = ddsp
        self.ddsp.gru.flatten_parameters()

        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))
        self.realtime = realtime

    def forward(self, pitch, loudness):
        loudness = (loudness - self.mean_loudness) / self.std_loudness
        if self.realtime:
            pitch = pitch[:, ::self.ddsp.block_size]
            loudness = loudness[:, ::self.ddsp.block_size]
            return self.ddsp.realtime_forward(pitch, loudness)
        else:
            return self.ddsp(pitch, loudness)

class DDSPInterpolator(torch.nn.Module):
    def __init__(self, ddsp1, ddsp2, mean_loudness, std_loudness, realtime):
        super().__init__()
        for ddsp in (ddsp1, ddsp2):
            ddsp.gru.flatten_parameters()

        self.ddsp1 = ddsp1
        self.ddsp2 = ddsp2 

        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))
        self.realtime = realtime

    def forward(self, pitch, loudness, interp):
        if not self.realtime: raise ValueError
        
        loudness = (loudness - self.mean_loudness) / self.std_loudness

        pitch = pitch[:, ::self.ddsp.block_size]
        loudness = loudness[:, ::self.ddsp.block_size]

        batch = {'pitch': pitch, 'loudness': loudness}

        # get the controls for both models
        ctrls1 = self.ddsp1.get_controls(batch)
        ctrls2 = self.ddsp2.get_controls(batch)

        # interpolate between the controls
        interp_ctrls = NotImplemented
        
        # synthesize on just one model
        output = self.ddsp1.synthesize(batch, interp_ctrls)

        return output['signal']

def export_single_decoder():

    with open(path.join(args.RUN, "config.yaml"), "r") as config:
        config = yaml.safe_load(config)

    ddsp.train.load_mode

    ddsp = DDSPDecoder(**config["model"])

    state = ddsp.state_dict()
    pretrained = torch.load(path.join(args.RUN, "state.pth"), map_location="cpu")
    state.update(pretrained)
    ddsp.load_state_dict(state)

    name = path.basename(path.normpath(args.RUN))

    scripted_model = torch.jit.script(
        ScriptDDSP(
            ddsp,
            config["data"]["mean_loudness"],
            config["data"]["std_loudness"],
            args.REALTIME,
        ))
    torch.jit.save(
        scripted_model,
        path.join(args.OUT_DIR, f"ddsp_{name}_pretrained.ts"),
    )

    impulse = ddsp.reverb.build_impulse().reshape(-1).numpy()
    sf.write(
        path.join(args.OUT_DIR, f"ddsp_{name}_impulse.wav"),
        impulse,
        config["preprocess"]["sample_rate"],
    )

    with open(
            path.join(args.OUT_DIR, f"ddsp_{name}_config.yaml"),
            "w",
    ) as config_out:
        yaml.safe_dump(config, config_out)

    if args.DATA:
        makedirs(path.join(args.OUT_DIR, "data"), exist_ok=True)
        file_list = get_files(**config["data"])
        file_list = [str(f).replace(" ", "\\ ") for f in file_list]
        system(f"cp {' '.join(file_list)} {path.normpath(args.OUT_DIR)}/data/")