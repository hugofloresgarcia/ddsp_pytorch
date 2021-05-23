import ddsp

from pathlib import Path
import yaml
from effortless_config import Config

import torch
import pytorch_lightning as pl

pl.seed_everything(42.0)

def load_model(name: str, config: dict):
    """ load a ddsp model by name
    config["model"]["kwargs"] will be the kwargs 
    passed to the model. 
    """
    if name == "single-inst-decoder":
        model = ddsp.models.DDSPDecoder(**config["model"]["kwargs"])
    elif name == "single-inst-fm":
        model = ddsp.models.DFMDecoder(**config["model"]["kwargs"])
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
        self.model = load_model(name=config['model']['name'],
                                config=config)

    def normalize_loudness(self, l):
        mu = self.config['data']['mean_loudness']
        std = self.config['data']['std_loudness']
        return (l - mu) / std

    def _main_step(self, batch: dict, index: int):
        sig, p, l = batch
        l = self.normalize_loudness(l).unsqueeze(-1)
        p = p.unsqueeze(-1)

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

        output.update(dict(
            rec=rec,
            sig=sig,
            sig_stft=sig_stft,
            rec_stft=rec_stft,
            loss=loss
        ))

        return output

    def training_step(self, batch: dict, index: int):
        output = self._main_step(batch, index)
        self.log_step(output, 'train')

        return output['loss']

    def validation_step(self, batch: dict, index: int):
        output = self._main_step(batch, index)
        self.log_step(output, 'validation')

    def log_step(self, output: dict, stage: str):
        IDX = 0
        writer = self.logger.experiment

        self.log(f'loss/{stage}', output['loss'].item())
        self.log(f"reverb_decay/{stage}", self.model.reverb.decay.item())
        self.log(f"reverb_wet/{stage}", self.model.reverb.wet.item())

        # log the audio to tb (instead of writing to file)
        writer.add_audio(f'sig/{stage}',
                         output['sig'][IDX],
                         global_step=self.global_step,
                         sample_rate=self.config['preprocess']["sample_rate"])
        writer.add_audio(f'rec/{stage}',
                         output['rec'][IDX],
                         global_step=self.global_step,
                         sample_rate=self.config['preprocess']["sample_rate"])

        fig = self.model.reconstruction_report(output['sig_stft'],
                                          output['rec_stft'],
                                          self.config,
                                          output,
                                          index=IDX)

        writer.add_figure(f'reconstruction/{stage}', fig, self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['train']['lr'])
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                          patience=10000, verbose=True,),
            'interval': 'step',
            'frequency': 1,
            'monitor': 'loss/train'
        }
        return [optimizer], [scheduler]

def train(config):
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=args.ROOT, name=args.NAME)
    exp_dir = Path(logger.log_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    datamodule = ddsp.data.Datamodule(config)
    datamodule.setup()

    mean_loudness, std_loudness = ddsp.mean_std_loudness(datamodule.train_data)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

    task = DDSPTask(config)

    with open(exp_dir / "config.yaml" , "w") as f:
        yaml.safe_dump(config, f)

    # CALLBACKS
    callbacks = []

    # log learning rate
    from pytorch_lightning.callbacks import LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # add checkpointing
    from pytorch_lightning.callbacks import ModelCheckpoint
    ckpt_callback = ModelCheckpoint(dirpath=exp_dir / 'checkpoints',
                                    filename=None,
                                    monitor='loss/validation',
                                    save_top_k=1,
                                    mode='min')
    callbacks.append(ckpt_callback)

    # add early stop
    from pytorch_lightning.callbacks import EarlyStopping
    callbacks.append(EarlyStopping(monitor='loss/validation', mode='min',
                                   patience=1000))

    # Setup trainer
    trainer = pl.Trainer(
        accelerator='dp',
        auto_lr_find=True,
        max_steps=config['train']['steps'],
        callbacks=callbacks,
        logger=logger,
        terminate_on_nan=True,
        # resume_from_checkpoint=best_model_path,
        log_gpu_memory=True,
        gpus=[config['device']],
        num_sanity_val_steps=0,
        move_metrics_to_cpu=True)

    trainer.fit(task, datamodule=datamodule)


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
