import torch
import ddsp
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSPDecoder
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
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000
    DEVICE = 0 if torch.cuda.is_available() else None


args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

model = DDSPDecoder(**config["model"]).to(args.DEVICE)

dm = ddsp.data.Datamodule(config)
dm.setup()

train_loader = dm.train_dataloader()

mean_loudness, std_loudness = mean_std_loudness(train_loader)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

best_loss = float("inf")
mean_loss = 0
n_element = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(train_loader)))


def multiscale_spec_loss(ori_stft, rec_stft):
    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss
    return loss

pbar = tqdm(range(epochs))
for e in pbar:
    for sig, p, l in train_loader:
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
        output.update({
            'sig_stft': sig_stft,
            'rec_stft': rec_stft,
            'sig': sig,
            'rec': rec,
            'loss': loss
        })

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss", loss.item(), step)

        step += 1
        pbar.set_description(
            desc=f'step {step%len(train_loader)}/{len(train_loader)}')

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element

    # LOGGING
    if not e % LOG_INTERVAL:

        # checkpoint model if needed
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                model.state_dict(),
                path.join(args.ROOT, args.NAME, "state.pth"),
            )
        # reset loss
        mean_loss = 0
        n_element = 0

        ddsp.utils.log_step(model, writer, output, stage='train',
                            step=step, config=config)