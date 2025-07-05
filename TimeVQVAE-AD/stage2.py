"""
Stage 2: Prior learning

run `python stage2.py`
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from argparse import ArgumentParser
import datetime

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.exp_stage2 import ExpStage2
from preprocessing.dataset import PreprocessedSamplesDataset
from utils import get_root_dir, load_yaml_param_settings


def load_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str,
        default=get_root_dir().joinpath('configs', 'config.yaml'),
        help="Path to the config file."
    )
    parser.add_argument(
        '--gpu_device_ind', nargs='+', default=[0], type=int,
        help="List of GPU device indices to use."
    )
    parser.add_argument(
        '--dataset_ind', default='1', nargs='+', required=True,
        help='e.g., 1 2 3. Indices of datasets to run experiments on.'
    )
    return parser.parse_args()


def train_stage2(config: dict,
                 dataset_idx: int,
                 window_size: int,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 gpu_device_ind: list):
    project_name = config['dataset']['names'] + "-stage2"
    input_length = window_size

    # initialize the Lightning module
    train_exp = ExpStage2(dataset_idx, input_length, config)

    # logging
    n_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    extra = {
        'dataset.idx': dataset_idx,
        'n_trainable_params': n_params,
        'gpu_device_ind': gpu_device_ind,
        'window_size': window_size
    }
    wandb_logger = WandbLogger(project=project_name, name=None,
                               config={**config, **extra})

    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        callbacks=[LearningRateMonitor(logging_interval='step')],
        max_steps=config['trainer_params']['max_steps']['stage2'],
        devices=gpu_device_ind,
        accelerator="gpu",
        strategy='ddp_find_unused_parameters_true' if len(gpu_device_ind) > 1 else "auto",
        val_check_interval=config['trainer_params']['val_check_interval']['stage2'],
        check_val_every_n_epoch=None,
        max_time=datetime.timedelta(hours=config['trainer_params']['max_hours']['stage2']),
    )

    # train
    trainer.fit(train_exp,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader)

    # save checkpoint
    print("saving the model...")
    trainer.save_checkpoint(
        os.path.join('saved_models', f'stage2.ckpt')
    )
    wandb.finish()


if __name__ == '__main__':
    args = load_args()
    config = load_yaml_param_settings(args.config)

    for idx in args.dataset_ind:
        dataset_idx = int(idx)

        # resolve data directory
        data_dir = config['dataset']['root_dir'] + config['dataset']['dataset_windowed']
        max_files = config['dataset']['max_files']

        # build datasets
        train_ds = PreprocessedSamplesDataset(
            data_dir=str(data_dir),
            max_loaded_files=max_files,
            kind='train',
            train_frac=0.8
        )
        test_ds = PreprocessedSamplesDataset(
            data_dir=str(data_dir),
            max_loaded_files=max_files,
            kind='test',
            train_frac=0.8
        )

        # infer window size from the first sample
        first = train_ds[0]
        x = first if not isinstance(first, tuple) else first[0]
        window_size = x.shape[-1]
        print(f"Using window_size = {window_size}")

        # wrap in DataLoaders
        batch_size = config['dataset']['batch_sizes']['stage2']
        num_workers = config['dataset']['num_workers']
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )

        # run Stage 2
        train_stage2(config, dataset_idx, window_size,
                     train_loader, test_loader,
                     args.gpu_device_ind)