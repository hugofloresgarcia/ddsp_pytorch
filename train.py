from re import L
import torch
import ddsp
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSPDecoder
from effortless_config import Config
from os import path
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from einops import rearrange
from ddsp.utils import get_scheduler
import numpy as np
import librosa as li

LOG_INTERVAL = 1 # in epochs
VAL_INTERVAL = 10

class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    DEVICE = 0 if torch.cuda.is_available() else None

args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

model = DDSPDecoder(**config["model"]).to(args.DEVICE)

dm = ddsp.data.Datamodule(config)
dm.setup()

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

mean_loudness, std_loudness = mean_std_loudness(train_loader)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

best_loss = float("inf")
mean_loss = 0
n_element = 0
step = 0
epochs = int(np.ceil(config['train']['steps'] / len(train_loader)))

def multiscale_spec_loss(ori_stft, rec_stft):
    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss
    return loss

def batch2cuda(batch: dict, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def _main_step(model, batch):
    batch = batch2cuda(batch, args.DEVICE)

    batch['loudness'] = (batch['loudness'] - mean_loudness) / std_loudness

    output = model(batch)
    rec = output['signal'].squeeze(-1)

    sig_stft = multiscale_fft(
        batch['sig'],
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
        'sig': batch['sig'],
        'rec': rec,
        'loss': loss
    })

    return output

def val_loop(model, dataloader, config, step):
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for index, batch in pbar:
        output = _main_step(model, batch)

    ddsp.utils.log_step(model, writer, output,
                        'val', step, config)

pbar = tqdm(range(epochs))
for e in pbar:
    for batch in train_loader:
        output = _main_step(model, batch)
        loss = output['loss']

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss", loss.item(), step)

        step += 1
        pbar.set_description(
            desc=f'step {step%len(train_loader)}/{len(train_loader)}')

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element

    # VALIDATION
    if not e % VAL_INTERVAL:
        model.eval()
        with torch.no_grad():
            val_loop(model, val_loader, config, e)
        model.train()
    #

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