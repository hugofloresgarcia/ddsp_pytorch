import ddsp

from pathlib import Path

import yaml
from effortless_config import Config
from tqdm import tqdm

import torch
import numpy as np
import pytorch_lightning as pl

# pl.seed_everything(42.0)


def load_model(name: str, config: dict):
    """ load a ddsp model by name
    config["model"]["kwargs"] will be the kwargs 
    passed to the model. 
    """
    if name == "single-inst-decoder":
        model = ddsp.models.DDSPDecoder(**config["model"]["kwargs"])
    else:
        raise ValueError(f'invalid model name: {name}')
    return model


class DDSPTask(pl.LightningModule):
    """
    container for training, validation and logging
    corresponding to a DDSP model. 
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = load_model(name=config['model']['name'], config=config)
        self.model.to(self.config['device'])

    def normalize_loudness(self, l):
        mu = self.config['data']['mean_loudness']
        std = self.config['data']['std_loudness']
        return (l - mu) / std

    def _main_step(self, batch: dict, index: int):
        sig, p, l = batch
        sig = sig.to(self.config['device'])
        l = l.unsqueeze(-1).to(self.config['device'])
        p = p.unsqueeze(-1).to(self.config['device'])

        l = self.normalize_loudness(l)

        output = self.model(p, l)
        rec = output['signal'].squeeze(-1)

        sig_stft = ddsp.multiscale_fft(
            sig,
            self.config["train"]["scales"],
            self.config["train"]["overlap"],
        )
        rec_stft = ddsp.multiscale_fft(
            rec,
            self.config["train"]["scales"],
            self.config["train"]["overlap"],
        )

        loss = ddsp.multiscale_spec_loss(sig_stft, rec_stft)

        output.update(
            dict(rec=rec,
                 sig=sig,
                 sig_stft=sig_stft,
                 rec_stft=rec_stft,
                 loss=loss))

        return output

    def training_step(self, batch: dict, index: int):
        output = self._main_step(batch, index)
        # self.log_step(output, 'train')

        return output

    def validation_step(self, batch: dict, index: int):
        output = self._main_step(batch, index)
        self.log_step(output, 'validation')

    def log_step(self, output: dict, stage: str):
        IDX = 0
        writer = self.writer

        writer.add_scalar(f'loss/{stage}', output['loss'].item(), self.step)
        writer.add_scalar(f"reverb_decay/{stage}",
                          self.model.reverb.decay.item(), self.step)
        writer.add_scalar(f"reverb_wet/{stage}", self.model.reverb.wet.item(),
                          self.step)

        # log the audio to tb (instead of writing to file)
        writer.add_audio('sig',
                         output['sig'][IDX],
                         global_step=self.step,
                         sample_rate=self.config['preprocess']["sample_rate"])
        writer.add_audio('rec',
                         output['rec'][IDX],
                         global_step=self.step,
                         sample_rate=self.config['preprocess']["sample_rate"])

        fig = self.model.reconstruction_report(output['sig_stft'],
                                               output['rec_stft'],
                                               self.config,
                                               output,
                                               index=IDX)

        writer.add_figure(f'reconstruction/{stage}', fig, self.step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config['train']['start_lr'])
        return optimizer


def train(config):
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=args.ROOT, name=args.NAME)
    exp_dir = Path(logger.log_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    datamodule = ddsp.data.Datamodule(config)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    mean_loudness, std_loudness = ddsp.mean_std_loudness(datamodule.train_data)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    task = DDSPTask(config)
    task.writer = logger.experiment

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # CALLBACK FOR CHECKPOINTS
    from pytorch_lightning.callbacks import ModelCheckpoint
    ckpt_callback = ModelCheckpoint(dirpath=exp_dir / 'checkpoints',
                                    filename=None,
                                    monitor='loss/validation',
                                    save_top_k=1,
                                    mode='min')

    # OPTIMIZERS
    opt = task.configure_optimizers()

    best_loss = float("inf")
    mean_loss = 0
    n_element = 0
    step = 0
    epochs = int(np.ceil(config['train']['steps'] / len(train_loader)))

    pbar = tqdm(list(range(epochs)))
    for epoch in pbar:
        for index, batch in enumerate(train_loader):

            output = task.training_step(batch, index)
            loss = output['loss']

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            pbar.set_description(desc=f'epoch {epoch} // step {index}')

            task.writer.add_scalar("loss", loss.item(), step)
            task.step = step

            n_element += 1
            mean_loss += (loss.item() - mean_loss) / n_element

        if not epoch % config['train']['log_interval']:
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(
                    task.state_dict(),
                    exp_dir / 'state.pth',
                )

            mean_loss = 0
            n_element = 0

            task.log_step(output, 'train')



if __name__ == "__main__":

    class args(Config):
        CONFIG = "config.yaml"
        NAME = "debug"
        DEVICE = 0 if torch.cuda.is_available() else None
        ROOT = 'runs'

    args.parse_args()

    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    config['root'] = args.ROOT
    config['name'] = args.NAME
    config['device'] = args.DEVICE

    train(config)