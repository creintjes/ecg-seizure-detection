#!/usr/bin/env python3
"""
Two-Phase Training Script

Phase 1: Train Stage 1 models for each base config
Phase 2: Use trained Stage 1 models to run comprehensive Stage 2 grid search

This approach is more efficient because:
- Stage 1 is trained once per configuration
- Stage 2 experiments reuse the trained Stage 1 models
- No need to retrain Stage 1 for each Stage 2 experiment

Usage:
    python two_phase_training.py --base-configs config1.yaml config2.yaml [--phase 1|2|both]
"""

import os
import sys
import yaml
import shutil
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from typing import Dict, List, Any
import time
import copy

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--base-configs', type=str, nargs='+', required=True,
                       help="Base config files (e.g., config_graceful-frog-123.yaml config_faithful-butterfly.yaml)")
    parser.add_argument('--phase', type=str, choices=['1', '2', 'both'], default='both',
                       help="Which phase to run: 1=stage1 only, 2=stage2 only, both=stage1 then stage2")
    parser.add_argument('--dry-run', action='store_true',
                       help="Show what would be done without executing")
    parser.add_argument('--gpu-device', type=int, nargs='+', default=[0],
                       help="GPU device indices to use")
    parser.add_argument('--dataset-ind', type=str, default="001",
                       help="Dataset index to use for training")
    parser.add_argument('--stage2-hours', type=int, default=1,
                       help="Max hours for stage2 runs")
    parser.add_argument('--stage2-max-steps', type=int, default=8000,
                       help="Max steps for stage2 runs")
    return parser.parse_args()

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(cfg: Dict[str, Any], path: Path) -> None:
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2)

def backup_config(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, backup)
    print(f"[backup] {path} -> {backup}")
    return backup

def run_stage1_training(config_path: Path, gpu_devices: List[int], dataset_ind: str, dry_run: bool) -> bool:
    """Run stage1.py training script with the given config."""
    cmd = [
        sys.executable, "stage1.py",
        "--config", str(config_path),
        "--gpu_device_ind"] + [str(gpu) for gpu in gpu_devices] + [
        "--dataset_ind", dataset_ind
    ]
    
    print(f"[stage1] {' '.join(cmd)}")
    
    if dry_run:
        print("[dry-run] Would execute stage1.py here")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        print("[stage1] ✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[stage1] ✗ Training failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("[stage1] Training interrupted by user")
        return False

def run_stage2_training(config_path: Path, gpu_devices: List[int], dataset_ind: str, dry_run: bool) -> bool:
    """Run stage2.py training script with the given config."""
    cmd = [
        sys.executable, "stage2.py", 
        "--config", str(config_path),
        "--gpu_device_ind"] + [str(gpu) for gpu in gpu_devices] + [
        "--dataset_ind", dataset_ind
    ]
    
    print(f"[stage2] {' '.join(cmd)}")
    
    if dry_run:
        print("[dry-run] Would execute stage2.py here")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        print("[stage2] ✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[stage2] ✗ Training failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("[stage2] Training interrupted by user")
        return False

def clean_saved_models_for_stage1():
    """Clean saved models directory before stage1 training."""
    saved_models_dir = Path("saved_models")
    if saved_models_dir.exists() and any(saved_models_dir.iterdir()):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path(f"saved_models_backup_stage1_{timestamp}")
        print(f"[cleanup] Moving existing models to: {backup_dir}")
        shutil.move(str(saved_models_dir), str(backup_dir))
    saved_models_dir.mkdir(exist_ok=True)

def get_stage1_checkpoint_path(config_name: str, dataset_ind: str) -> Path:
    """Get the expected stage1 checkpoint path for a config."""
    # Use the config name as part of checkpoint name for uniqueness
    base_name = Path(config_name).stem
    checkpoint_name = f"stage1-{base_name}-{dataset_ind}_window.ckpt"
    return Path("saved_models") / checkpoint_name

def get_expected_stage2_checkpoint_path(dataset_ind: str) -> Path:
    """Get the checkpoint path that stage2/MaskGIT expects to find."""
    checkpoint_name = f"stage1-{dataset_ind}_window.ckpt"
    return Path("saved_models") / checkpoint_name

def update_config_with_stage2_params(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update config with stage2 parameters only.
    
    IMPORTANT: This function should ONLY modify:
    - MaskGIT.* parameters (prior_model, choice_temperature, T, etc.)
    - exp_params.lr.stage2
    - trainer_params.*stage2
    - dataset.batch_sizes.stage2
    
    NEVER modify Stage 1 parameters:
    - encoder.* (dim, n_resnet_blocks, downsampled_width)
    - decoder.*
    - VQ-VAE.* (n_fft, codebook_size)
    """
    config = copy.deepcopy(base_config)
    for dotted_key, value in updates.items():
        keys = dotted_key.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    return config

def latent_length(cfg: Dict[str, Any]) -> int:
    """Calculate latent sequence length."""
    nfft = int(cfg["VQ-VAE"]["n_fft"])
    W = int(cfg["encoder"]["downsampled_width"])
    F = (nfft // 2) + 1
    return F * W

def get_stage2_batch_sizes(L: int) -> List[int]:
    """Get appropriate batch sizes for stage2 based on latent length."""
    if L <= 144:
        return [256, 512]
    elif L <= 288:
        return [128, 256]
    elif L <= 528:
        return [64, 128]
    else:
        return [32, 48, 64]

def phase1_train_stage1_models(base_configs: List[str], args) -> Dict[str, bool]:
    """Phase 1: Train stage1 models for each base config."""
    print("\n" + "="*80)
    print("PHASE 1: TRAINING STAGE 1 MODELS")
    print("="*80)
    
    results = {}
    
    for config_path_str in base_configs:
        config_path = Path(config_path_str)
        if not config_path.exists():
            print(f"[ERROR] Config not found: {config_path}")
            results[config_path_str] = False
            continue
            
        print(f"\n[phase1] Processing: {config_path.name}")
        
        # Check if stage1 checkpoint already exists
        checkpoint_path = get_stage1_checkpoint_path(config_path.name, args.dataset_ind)
        if checkpoint_path.exists():
            print(f"[phase1] ✓ Stage1 checkpoint already exists: {checkpoint_path}")
            results[config_path_str] = True
            continue
        
        # Create a temporary working config for this training
        temp_config_path = Path("configs") / "temp_stage1_training.yaml"
        
        # Copy the base config to temp location
        if not args.dry_run:
            shutil.copy2(config_path, temp_config_path)
        
        try:
            # Train stage1 with the specific config
            print(f"[phase1] Training stage1 with config: {config_path}")
            success = run_stage1_training(temp_config_path, args.gpu_device, args.dataset_ind, args.dry_run)
            
            if success and not args.dry_run:
                # Move the generated checkpoint to the correct name
                default_checkpoint = Path("saved_models") / f"stage1-{args.dataset_ind}_window.ckpt"
                if default_checkpoint.exists():
                    shutil.move(str(default_checkpoint), str(checkpoint_path))
                    print(f"[phase1] ✓ Checkpoint saved as: {checkpoint_path}")
                else:
                    print(f"[phase1] ✗ Expected checkpoint not found: {default_checkpoint}")
                    success = False
            
            results[config_path_str] = success
            
            if success:
                print(f"[phase1] ✓ Stage1 training completed for {config_path.name}")
            else:
                print(f"[phase1] ✗ Stage1 training failed for {config_path.name}")
                
        finally:
            # Clean up temp config
            if temp_config_path.exists() and not args.dry_run:
                temp_config_path.unlink()
    
    print(f"\n[phase1] Summary:")
    for config, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {Path(config).name}")
    
    return results

def phase2_train_stage2_models(base_configs: List[str], stage1_results: Dict[str, bool], args) -> Dict[str, List[Dict]]:
    """Phase 2: Train stage2 models using trained stage1 checkpoints."""
    print("\n" + "="*80)
    print("PHASE 2: TRAINING STAGE 2 MODELS")
    print("="*80)
    
    # Stage2 hyperparameter grid - ONLY MaskGIT and training parameters
    # DO NOT modify Stage 1 parameters (encoder, decoder, VQ-VAE)
    stage2_grid = {
        "MaskGIT.prior_model.depth": [8, 10, 12],
        "MaskGIT.prior_model.heads": [6, 8, 12],
        "MaskGIT.prior_model.dropout": [0.15, 0.20],
        "exp_params.lr.stage2": [1e-3, 8e-4],
    }
    
    # Fixed stage2 parameters - ONLY MaskGIT and training parameters
    stage2_fixed = {
        "trainer_params.max_hours.stage2": args.stage2_hours,
        "trainer_params.max_steps.stage2": args.stage2_max_steps,
        "trainer_params.val_check_interval.stage2": 1000,
        "MaskGIT.choice_temperature": 2,
        "MaskGIT.T": 25,
        "MaskGIT.mask_scheduling_func": "cosine",
        "MaskGIT.prior_model.hidden_dim": 384,
        "MaskGIT.prior_model.use_rmsnorm": True,
        "MaskGIT.prior_model.ff_mult": 4,
    }
    
    all_results = {}
    
    for config_path_str in base_configs:
        config_path = Path(config_path_str)
        
        # Skip if stage1 training failed
        if not stage1_results.get(config_path_str, False):
            print(f"[phase2] Skipping {config_path.name} - stage1 training failed")
            all_results[config_path_str] = []
            continue
        
        print(f"\n[phase2] Processing: {config_path.name}")
        
        # Load base config
        base_config = load_config(config_path)
        
        # Calculate latent length and get appropriate batch sizes
        L = latent_length(base_config)
        batch_sizes = get_stage2_batch_sizes(L)
        
        print(f"[phase2] Latent length L = {L}")
        print(f"[phase2] Stage2 batch sizes: {batch_sizes}")
        
        # Get the stage1 checkpoint path for this config
        stage1_checkpoint = get_stage1_checkpoint_path(config_path.name, args.dataset_ind)
        expected_checkpoint = get_expected_stage2_checkpoint_path(args.dataset_ind)
        
        if not stage1_checkpoint.exists():
            print(f"[phase2] ✗ Stage1 checkpoint not found: {stage1_checkpoint}")
            all_results[config_path_str] = []
            continue
        
        # Create temporary config file for Stage 2 training
        temp_config_path = Path("configs") / "temp_stage2_training.yaml"
        backup_main_config = None
        backup_expected_checkpoint = None
        
        # Backup main config.yaml if it exists
        main_config = Path("configs") / "config.yaml"
        if main_config.exists():
            backup_main_config = backup_config(main_config)
        
        # Backup existing expected checkpoint if it exists
        if expected_checkpoint.exists():
            backup_expected_checkpoint = expected_checkpoint.with_suffix('.ckpt.backup.' + datetime.now().strftime('%Y%m%d_%H%M%S'))
            shutil.copy2(expected_checkpoint, backup_expected_checkpoint)
            print(f"[phase2] Backed up existing checkpoint: {backup_expected_checkpoint}")
        
        # Copy the stage1 checkpoint to the expected location
        if not args.dry_run:
            shutil.copy2(stage1_checkpoint, expected_checkpoint)
            print(f"[phase2] Copied checkpoint: {stage1_checkpoint} -> {expected_checkpoint}")
        
        config_results = []
        
        try:
            # Generate all parameter combinations
            param_names = list(stage2_grid.keys())
            param_values = list(stage2_grid.values())
            
            total_combinations = len(list(itertools.product(*param_values))) * len(batch_sizes)
            print(f"[phase2] Total stage2 combinations for {config_path.name}: {total_combinations}")
            
            combination_count = 0
            
            for batch_size in batch_sizes:
                for param_combo in itertools.product(*param_values):
                    combination_count += 1
                    
                    # Create parameter dictionary
                    params = dict(zip(param_names, param_combo))
                    
                    # Calculate attention dimension head
                    heads = params["MaskGIT.prior_model.heads"]
                    attn_dim_head = 384 // heads
                    
                    # Combine all stage2 updates
                    stage2_updates = {
                        **stage2_fixed,
                        **params,
                        "MaskGIT.prior_model.attn_dim_head": attn_dim_head,
                        "dataset.batch_sizes.stage2": batch_size,
                    }
                    
                    print(f"\n[phase2] Combination {combination_count}/{total_combinations}")
                    print(f"  Config: {config_path.name}")
                    print(f"  Batch size: {batch_size}")
                    print(f"  Depth: {params['MaskGIT.prior_model.depth']}")
                    print(f"  Heads: {heads}x{attn_dim_head}")
                    print(f"  Dropout: {params['MaskGIT.prior_model.dropout']}")
                    print(f"  LR: {params['exp_params.lr.stage2']}")
                    
                    # Update config with stage2 parameters
                    updated_config = update_config_with_stage2_params(base_config, stage2_updates)
                    
                    # Save updated config as the main config (stage2.py uses config.yaml by default)
                    if not args.dry_run:
                        save_config(updated_config, main_config)
                    
                    # Run stage2 training
                    start_time = time.time()
                    success = run_stage2_training(main_config, args.gpu_device, args.dataset_ind, args.dry_run)
                    end_time = time.time()
                    
                    # Record results
                    result = {
                        "combination": combination_count,
                        "batch_size": batch_size,
                        "depth": params["MaskGIT.prior_model.depth"],
                        "heads": heads,
                        "attn_dim_head": attn_dim_head,
                        "dropout": params["MaskGIT.prior_model.dropout"],
                        "lr": params["exp_params.lr.stage2"],
                        "success": success,
                        "duration": end_time - start_time,
                        "L": L,
                        "stage1_checkpoint": str(stage1_checkpoint)
                    }
                    config_results.append(result)
                    
                    status = "✓" if success else "✗"
                    print(f"  {status} Completed in {result['duration']:.1f}s")
        
        finally:
            # Restore original main config if it was backed up
            if backup_main_config and backup_main_config.exists():
                if not args.dry_run:
                    shutil.copy2(backup_main_config, main_config)
                    print(f"[phase2] Restored original config: {main_config}")
            
            # Restore original checkpoint if it was backed up
            if backup_expected_checkpoint and backup_expected_checkpoint.exists():
                if not args.dry_run:
                    shutil.copy2(backup_expected_checkpoint, expected_checkpoint)
                    print(f"[phase2] Restored original checkpoint: {expected_checkpoint}")
            elif expected_checkpoint.exists() and not args.dry_run:
                # Remove the checkpoint we copied if there was no backup
                expected_checkpoint.unlink()
                print(f"[phase2] Removed temporary checkpoint: {expected_checkpoint}")
        
        all_results[config_path_str] = config_results
        
        # Summary for this config
        successful = sum(1 for r in config_results if r["success"])
        total = len(config_results)
        print(f"[phase2] {config_path.name}: {successful}/{total} successful")
    
    return all_results

def print_final_summary(stage1_results: Dict[str, bool], stage2_results: Dict[str, List[Dict]]):
    """Print final summary of all results."""
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nStage 1 Results:")
    for config, success in stage1_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {Path(config).name}")
    
    print("\nStage 2 Results:")
    total_stage2_runs = 0
    total_stage2_success = 0
    
    for config, results in stage2_results.items():
        if results:
            successful = sum(1 for r in results if r["success"])
            total = len(results)
            total_stage2_runs += total
            total_stage2_success += successful
            print(f"  {Path(config).name}: {successful}/{total} successful")
        else:
            print(f"  {Path(config).name}: Skipped (stage1 failed)")
    
    print(f"\nOverall Stage 2: {total_stage2_success}/{total_stage2_runs} successful")
    print("\nCheck WandB for detailed metrics and model comparison!")

def main():
    args = load_args()
    
    print("TWO-PHASE TRAINING SCRIPT")
    print("="*50)
    print(f"Base configs: {[Path(c).name for c in args.base_configs]}")
    print(f"Phase: {args.phase}")
    print(f"Dataset: {args.dataset_ind}")
    print(f"GPU devices: {args.gpu_device}")
    print(f"Dry run: {args.dry_run}")
    
    stage1_results = {}
    stage2_results = {}
    
    try:
        # Phase 1: Train stage1 models
        if args.phase in ['1', 'both']:
            stage1_results = phase1_train_stage1_models(args.base_configs, args)
        else:
            # Assume stage1 models exist if we're only doing phase 2
            for config in args.base_configs:
                checkpoint_path = get_stage1_checkpoint_path(Path(config).name, args.dataset_ind)
                stage1_results[config] = checkpoint_path.exists()
        
        # Phase 2: Train stage2 models
        if args.phase in ['2', 'both']:
            stage2_results = phase2_train_stage2_models(args.base_configs, stage1_results, args)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    finally:
        # Print summary
        if stage1_results or stage2_results:
            print_final_summary(stage1_results, stage2_results)

if __name__ == "__main__":
    main()
