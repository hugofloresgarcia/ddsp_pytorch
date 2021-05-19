import torch
import ddsp
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import path
from preprocess import Dataset
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from einops import rearrange
from ddsp.utils import get_scheduler
import numpy as np
import librosa as li

LOG_INTERVAL = 1 # in epochs


class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    STEPS = 500000
    BATCH = 16
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000
    DEVICE = 0 if torch.cuda.is_available() else None


args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

model = DDSP(**config["model"]).to(args.DEVICE)

dataset = Dataset(config["preprocess"]["out_dir"])

dataloader = torch.utils.data.DataLoader(
    dataset,
    args.BATCH,
    True,
    drop_last=True,
)

mean_loudness, std_loudness = mean_std_loudness(dataloader)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

schedule = get_scheduler(
    len(dataloader),
    args.START_LR,
    args.STOP_LR,
    args.DECAY_OVER,
)

# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

best_loss = float("inf")
mean_loss = 0
n_element = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(dataloader)))

def multiscale_spec_loss(ori_stft, rec_stft):
    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss
    return loss

pbar = tqdm(range(epochs))
for e in pbar:
    for sig, p, l in dataloader:
        sig = sig.to(args.DEVICE)
        p = p.unsqueeze(-1).to(args.DEVICE)
        l = l.unsqueeze(-1).to(args.DEVICE)

        l = (l - mean_loudness) / std_loudness

        output = model(p, l)
        rec = output['signal'].squeeze(-1)

        sig_stft = multiscale_fft(
            sig,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
        rec_stft = multiscale_fft(
            rec,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        loss = multiscale_spec_loss(sig_stft, rec_stft)

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss", loss.item(), step)

        step += 1
        pbar.set_description(desc=f'step {step%len(dataset)}/{len(dataset)}')

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element

    # LOGGING
    if not e % LOG_INTERVAL:
        writer.add_scalar("lr", schedule(e), e)
        writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
        writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
        # scheduler.step()
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                model.state_dict(),
                path.join(args.ROOT, args.NAME, "state.pth"),
            )

        mean_loss = 0
        n_element = 0

        # log original and recreated stfts
        # each of these spectrograms is multiscale (has channels for different window sizes)
        # so we'll just plot one of them

        ddsp.utils.log_sample_stft(writer, sig_stft, 'sig_stft', e,
                                   config)
        ddsp.utils.log_sample_stft(writer, rec_stft, 'rec_stft', e,
                                   config)

        # log the audio to tb (instead of writing to file)
        writer.add_audio('sig',
                         sig[ddsp.utils.IDX],
                         global_step=e,
                         sample_rate=config['preprocess']["sampling_rate"])
        writer.add_audio('rec',
                         rec[ddsp.utils.IDX],
                         global_step=e,
                         sample_rate=config['preprocess']["sampling_rate"])

        ddsp.utils.log_harmonic_amps(writer, output['harmonic_amps'], 'harmonic_amps', e)
        ddsp.utils.log_pitch_curve(writer, p, 'pitch (midi)', e)
        ddsp.utils.log_loudness_curve(writer, l, 'loudness', e)
