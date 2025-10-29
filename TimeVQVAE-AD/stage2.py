"""
Stage 2: prior learning

run `python stage2.py`
"""
import os
import datetime
from argparse import ArgumentParser
from pathlib import Path

import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from experiments.exp_stage2 import ExpStage2
from preprocessing.data_pipeline import build_data_pipeline
from utils import get_root_dir, load_yaml_param_settings, set_window_size


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int)
    parser.add_argument('--dataset_ind', default=None, nargs='+',
                        help='e.g., 001 or all. Subject indices (as string or "all") to run experiments on.')
    return parser.parse_args()


def train_stage2(config: dict,
                 subject_id: str,
                 window_size: int,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind: list):
    """
    Train Stage 2 model.
    """
    project_name = config['dataset']['names'] + "-stage2"
    input_length = window_size

    # Initialize experiment
    train_exp = ExpStage2(subject_id, input_length, config)

    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    extra_config = {
        'dataset.sub': subject_id,
        'n_trainable_params': n_trainable_params,
        'gpu_device_ind': gpu_device_ind,
        'window_size': window_size
    }

    wandb_logger = WandbLogger(project=project_name, name=None, config={**config, **extra_config})

    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        callbacks=[LearningRateMonitor(logging_interval='step')],
        max_steps=config['trainer_params']['max_steps']['stage2'],
        devices=gpu_device_ind,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true' if len(gpu_device_ind) > 1 else "auto",
        val_check_interval=config['trainer_params']['val_check_interval']['stage2'],
        check_val_every_n_epoch=None,
        max_time=datetime.timedelta(hours=config['trainer_params']['max_hours']['stage2']),
    )

    trainer.fit(train_exp, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)

    print("Saving model...")
    trainer.save_checkpoint(os.path.join('saved_models', f'stage2-{subject_id}{"_window" if expand_labels else "_no_window"}.ckpt'))

    wandb.finish()


if __name__ == '__main__':
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # Dataset config
    sr = config['dataset']['downsample_freq']
    stride = config['dataset']['stride']
    data_dir = Path(config['dataset']['root_dir'] + f"/downsample_freq={sr},no_windows")
    batch_size = config['dataset']['batch_sizes']['stage2']
    num_workers = config['dataset']['num_workers']
    n_periods = config['dataset']['n_periods']
    bpm = config['dataset']['heartbeats_per_minute']
    expand_labels = config['dataset']['expand_labels']

    window_size = set_window_size(sr, n_periods, bpm=bpm)
    if not args.dataset_ind:
        args.dataset_ind = ["all"]
    for dataset_idx in [x for idx in args.dataset_ind for x in idx.split(',')]:
        sub = f"{int(dataset_idx):03d}" if str(dataset_idx) != "all" else "all"
        checkpoint_path = os.path.join('saved_models', f'stage2-{sub}{"_window" if expand_labels else "_no_window"}.ckpt')
        if os.path.exists(checkpoint_path):
            print(f"Skipping training for {sub}, checkpoint already exists at {checkpoint_path}")
            continue

        train_data_loader, test_data_loader = [build_data_pipeline(
                    batch_size,
                    data_dir,
                    sub,
                    kind,
                    window_size,
                    stride,
                    num_workers,
                    sampling_rate=sr,
                    expand_labels=expand_labels,
                    n_periods=n_periods,
                    bpm=bpm
                ) for kind in ['train', 'test']
            ]

        train_stage2(config, sub, window_size, train_data_loader, test_data_loader, args.gpu_device_ind)