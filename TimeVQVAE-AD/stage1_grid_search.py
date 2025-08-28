#!/usr/bin/env python3
"""
Stage 1 Grid Search Automation Script

This script automates grid search for stage 1 model training by:
1. Iterating through different parameter combinations
2. Updating the config.yaml file for each combination
3. Running the stage1.py training script
4. Cleaning up saved models between runs
5. Logging all configurations and results via WandB

Usage:
    python stage1_grid_search.py [--config path/to/config.yaml] [--dry-run]
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

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, 
                       default='configs/config.yaml',
                       help="Path to the base config file")
    parser.add_argument('--dry-run', action='store_true',
                       help="Print configurations without running training")
    parser.add_argument('--dataset_ind', type=str, default="001",
                       help="Dataset index to use for training")
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

# def backup_original_config(config_path: str) -> str:
#     """Create a backup of the original config file."""
#     backup_path = f"{config_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     # shutil.copy2(config_path, backup_path)
#     print(f"Original config backed up to: {backup_path}")
#     return backup_path

def clean_saved_models() -> None:
    """Clean up saved model checkpoints before next training run."""
    stage2_ckpt = Path("saved_models", "stage2-001_window.ckpt")
    if stage2_ckpt.exists():
        stage2_ckpt.unlink()  # Delete the original checkpoint
        print(f"Deleted checkpoint: {stage2_ckpt}")
    else:
        print("No existing stage2 checkpoint to clean up")
    
    # Ensure saved_models directory exists
    Path("saved_models").mkdir(exist_ok=True)

def update_config_with_params(base_config: Dict[str, Any], param_combination: Dict[str, Any]) -> Dict[str, Any]:
    """Update base configuration with parameter combination."""
    config = base_config.copy()
    
    # Deep copy to avoid modifying the original
    import copy
    config = copy.deepcopy(base_config)
    
    for param_path, value in param_combination.items():
        # Parse nested parameter paths like "encoder.dim" or "exp_params.lr.stage1"
        keys = param_path.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
    
    return config

def run_stage2_training(gpu_devices: List[int], dataset_ind: str) -> bool:
    """Run stage2.py training script."""
    cmd = [sys.executable, "stage2.py",] + ["--dataset_ind", dataset_ind]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return False

def main():
    args = load_args()
    
    # Define parameter grid for stage 1 (54 configurations)
    # Fixed: max_hours = 1, learning rate kept at default, batch_size reduced for GPU safety
    # Focus on most impactful model architecture parameters
    param_grid = {
        "encoder.dim": [128, 256, 512],                    # 3 options
        "encoder.n_resnet_blocks": [8, 10, 12],           # 3 options  
        "VQ-VAE.codebook_size": [256, 512, 1024],         # 3 options
        "encoder.downsampled_width": [32],             # 1 option
        "trainer_params.max_hours.stage1": [1],           # 1 option (fixed)
        "dataset.batch_sizes.stage1": [256],              # 1 option (reduced from 1024 for GPU safety)
    }
    
    # Load base configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    base_config = load_config(config_path)
    
    # Create backup of original config
    # backup_path = backup_original_config(config_path)
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Create all combinations
    combinations = list(itertools.product(*param_values))
    
    print(f"Grid Search Configuration:")
    print(f"  Encoder dimensions: {param_grid['encoder.dim']}")
    print(f"  Encoder ResNet blocks: {param_grid['encoder.n_resnet_blocks']}")
    print(f"  Downsampled widths: {param_grid['encoder.downsampled_width']}")
    print(f"  Codebook sizes: {param_grid['VQ-VAE.codebook_size']}")
    print(f"  Batch size: {param_grid['dataset.batch_sizes.stage1'][0]} (reduced for GPU safety)")
    print(f"  Max hours per run: {param_grid['trainer_params.max_hours.stage1'][0]}")
    print(f"\nTotal combinations to test: {len(combinations)}")
    print(f"Estimated total time: {len(combinations)} hours (1 hour per combination)")
    print(f"Learning rate: kept at default value from config")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        for i, combination in enumerate(combinations[:5]):  # Show first 5
            param_dict = dict(zip(param_names, combination))
            print(f"\nCombination {i+1}:")
            for param, value in param_dict.items():
                print(f"  {param}: {value}")
        print(f"\n... and {len(combinations)-5} more combinations")
        return
    
    # Track successful and failed runs
    successful_runs = []
    failed_runs = []
    
    try:
        for i, combination in enumerate(combinations):
            print(f"\n{'='*60}")
            print(f"Running combination {i+1}/{len(combinations)}")
            print(f"{'='*60}")
            
            # Create parameter dictionary for this combination
            param_dict = dict(zip(param_names, combination))
            
            print("Parameter combination:")
            for param, value in param_dict.items():
                print(f"  {param}: {value}")
            
            # Update config with current parameter combination
            current_config = update_config_with_params(base_config, param_dict)
            
            # Save updated config
            save_config(current_config, config_path)
            print(f"Updated config saved to: {config_path}")
            
            # Clean up saved models from previous run
            clean_saved_models()
            
            # Run training
            print("Starting training...")
            start_time = time.time()
            
            success = run_stage1_training(args.gpu_device, args.dataset_ind)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if success:
                successful_runs.append({
                    'combination': i+1,
                    'params': param_dict,
                    'duration': duration
                })
                print(f"✓ Combination {i+1} completed successfully in {duration:.1f}s")
            else:
                failed_runs.append({
                    'combination': i+1,
                    'params': param_dict,
                    'duration': duration
                })
                print(f"✗ Combination {i+1} failed after {duration:.1f}s")
            
            print(f"Progress: {i+1}/{len(combinations)} combinations completed")
            
    except KeyboardInterrupt:
        print(f"\n\nGrid search interrupted by user after {len(successful_runs + failed_runs)} combinations")
    
    finally:
        # Restore original config
        print(f"\nRestoring original config from: {backup_path}")
        shutil.copy2(backup_path, config_path)
        
        # Print summary
        print(f"\n{'='*60}")
        print("GRID SEARCH SUMMARY")
        print(f"{'='*60}")
        print(f"Total combinations attempted: {len(successful_runs + failed_runs)}")
        print(f"Successful runs: {len(successful_runs)}")
        print(f"Failed runs: {len(failed_runs)}")
        
        if successful_runs:
            print(f"\n✓ SUCCESSFUL COMBINATIONS:")
            for run in successful_runs:
                print(f"  Combination {run['combination']} (duration: {run['duration']:.1f}s)")
        
        if failed_runs:
            print(f"\n✗ FAILED COMBINATIONS:")
            for run in failed_runs:
                print(f"  Combination {run['combination']} (duration: {run['duration']:.1f}s)")
        
        print(f"\nOriginal config restored to: {config_path}")
        print(f"Check WandB for detailed results and metrics comparison")

if __name__ == "__main__":
    main()
