import ddsp
import torch
import torch.nn as nn
import yaml
from effortless_config import Config
from os import path, makedirs, system
from ddsp.models.decoder import DDSPDecoder
import soundfile as sf
from ddsp.preprocess import get_files
from pathlib import Path




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
    def __init__(self, ddsp1, ddsp2, mean_loudness, std_loudness):
        super().__init__()
        for ddsp in (ddsp1, ddsp2):
            ddsp.decoder.gru.flatten_parameters()

        self.ddsp1 = ddsp1
        self.ddsp2 = ddsp2

        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))

    def forward(self, pitch, loudness, alpha):
        if not self.realtime: raise ValueError

        loudness = (loudness - self.mean_loudness) / self.std_loudness

        pitch = pitch[:, ::self.ddsp.block_size]
        loudness = loudness[:, ::self.ddsp.block_size]

        batch = {'pitch': pitch, 'loudness': loudness}

        # get the controls for both models
        ctrls1 = self.ddsp1.get_controls(batch)
        ctrls2 = self.ddsp2.get_controls(batch)

        interp_ctrls = dict(ctrls1)
        # interpolate between the controls
        for ctrl_key in ('harmonic_ctrls', 'noise_ctrls'):
            interp_ctrls[ctrl_key] = ddsp.models.modules.interoplate_controls(
                ('harmonic_distribution', 'amplitudes'), ctrls1[ctrl_key],
                ctrls2[ctrl_key], alpha
            )

        # synthesize on just one model
        output = self.ddsp1.synthesize(batch, interp_ctrls)

        return output['signal']

def export_single_decoder(config):

    model = ddsp.train.load_model(config)

    state = model.state_dict()
    pretrained = torch.load(path.join(args.RUN, "state.pth"), map_location="cpu")
    state.update(pretrained)
    model.load_state_dict(state)

    name = path.basename(path.normpath(args.RUN))

    scripted_model = torch.jit.script(
        ScriptDDSP(
            model,
            config["data"]["mean_loudness"],
            config["data"]["std_loudness"],
            args.REALTIME,
        ))
    torch.jit.save(
        scripted_model,
        path.join(args.OUT_DIR, f"ddsp_{name}_pretrained.ts"),
    )

    impulse = model.reverb.build_impulse().reshape(-1).numpy()
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

def load_model_state_dict(model, config):
    state = model.state_dict()
    pretrained = torch.load(path.join(config['exp_dir'], "state.pth"),
                            map_location="cpu")
    state.update(pretrained)
    model.load_state_dict(state)
    return model

def export_multidecoder_interpolator(config1, config2, out_dir):
    torch.set_grad_enabled(False)

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    model1 = ddsp.train.load_model(config1)
    model2 = ddsp.train.load_model(config2)

    model1 = load_model_state_dict(model1, config1)
    model2 = load_model_state_dict(model2, config2)

    name = Path(config1['exp_dir']).stem

    scripted_model = torch.jit.script(
        DDSPInterpolator(
            model1,
            model2,
            config1["data"]["mean_loudness"],
            config1["data"]["std_loudness"],
        )
    )
    torch.jit.save(
        scripted_model,
        path.join(out_dir, f"ddsp_{name}_pretrained.ts"),
    )

    impulse = model1.reverb.build_impulse().reshape(-1).numpy()
    sf.write(
        path.join(out_dir, f"ddsp_{name}_impulse.wav"),
        impulse,
        config1["preprocess"]["sample_rate"],
    )

    with open(
            path.join(out_dir, f"ddsp_{name}_config.yaml"),
            "w",
    ) as config_out:
        yaml.safe_dump(config1, config_out)

def load_config(path):
    with open(path, "r") as config:
        config = yaml.safe_load(config)
    return config


def main():
    torch.set_grad_enabled(False)

    class args(Config):
        RUN = None
        DATA = False
        OUT_DIR = "export"
        REALTIME = False


    args.parse_args()
    makedirs(args.OUT_DIR, exist_ok=True)

    config = load_config(path.join(args.RUN, "config.yaml"))
    export_single_decoder(config)


if __name__ == "__main__":
    main()