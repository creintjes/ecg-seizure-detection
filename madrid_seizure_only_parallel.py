#!/usr/bin/env python3
"""
Parallel Madrid Seizure-Only Processor

High-performance parallel processing of seizure-only preprocessed data
with Madrid algorithm and JSON output.

Usage:
    python madrid_seizure_only_parallel.py --seizure-data-dir /path/to/data --n-workers 8 --max-files 20
    python madrid_seizure_only_parallel.py --seizure-data-dir /path/to/data --sampling-rate 32 --n-workers 4

Features:
    - Parallel processing with multiprocessing
    - Progress tracking with ETA
    - Thread-safe logging
    - Optimized for seizure-only data
    - GPU support in parallel workers
"""

import argparse
import json
import logging
import time
from pathlib import Path
from multiprocessing import cpu_count
from madrid_batch_processor_parallel import ParallelMadridBatchProcessor

def create_seizure_only_config():
    """Create optimized configuration for seizure-only data"""
    return {
        'use_gpu': True,
        'enable_output': False,  # Disabled for parallel processing
        'madrid_parameters': {
            'm_range': {
                'min_length': 80,
                'max_length': 800,
                'step_size': 80
            },
            'analysis_config': {
                'top_k': 5,  # More anomalies for seizure analysis
                'train_test_split_ratio': 0.3,  # Less training since seizure location known
                'threshold_percentile': 90  # Higher sensitivity
            },
            'algorithm_settings': {
                'use_gpu': True,
                'downsampling_factor': 1
            }
        }
    }

def estimate_processing_time(n_files: int, n_workers: int, sampling_rate: int = 32) -> dict:
    """
    Estimate processing time based on file count and hardware
    
    Args:
        n_files: Number of files to process
        n_workers: Number of parallel workers
        sampling_rate: Data sampling rate
        
    Returns:
        Dictionary with time estimates
    """
    # Base time estimates per file (seconds) based on sampling rate
    time_per_file = {
        8: 10,    # 8Hz: ~10s per file
        32: 30,   # 32Hz: ~30s per file  
        125: 120  # 125Hz: ~2min per file
    }
    
    base_time = time_per_file.get(sampling_rate, 30)
    
    # Serial time
    serial_time_sec = n_files * base_time
    
    # Parallel time (with some overhead)
    parallel_efficiency = 0.8  # 80% efficiency due to overhead
    parallel_time_sec = (serial_time_sec / n_workers) / parallel_efficiency
    
    # Add startup overhead
    startup_overhead = min(60, n_files * 2)  # Max 1min overhead
    parallel_time_sec += startup_overhead
    
    return {
        'serial_time_minutes': serial_time_sec / 60,
        'parallel_time_minutes': parallel_time_sec / 60,
        'speedup_factor': serial_time_sec / parallel_time_sec,
        'time_per_file_seconds': base_time,
        'estimated_completion': time.time() + parallel_time_sec
    }

def main():
    parser = argparse.ArgumentParser(description='Parallel Madrid analysis on seizure-only data')
    parser.add_argument('--seizure-data-dir', required=True, help='Directory with seizure-only .pkl files')
    parser.add_argument('--output-dir', default='madrid_seizure_results_parallel', help='Output directory')
    parser.add_argument('--max-files', type=int, help='Maximum files to process')
    parser.add_argument('--n-workers', type=int, help=f'Number of workers (default: {cpu_count()-1})')
    parser.add_argument('--config-file', help='Custom configuration file')
    parser.add_argument('--sampling-rate', type=int, choices=[8, 32, 125], help='Filter by sampling rate')
    parser.add_argument('--dry-run', action='store_true', help='Show processing plan without execution')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Validate data directory
    data_dir = Path(args.seizure_data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Find files
    pkl_files = list(data_dir.glob("*.pkl"))
    
    if args.sampling_rate:
        # Filter by sampling rate
        filtered_files = []
        for file in pkl_files:
            if f"{args.sampling_rate}hz" in file.name.lower():
                filtered_files.append(file)
        pkl_files = filtered_files
        logger.info(f"Filtered to {len(pkl_files)} files with {args.sampling_rate}Hz")
    
    if not pkl_files:
        logger.error(f"No .pkl files found in {data_dir}")
        return 1
    
    if args.max_files:
        pkl_files = pkl_files[:args.max_files]
    
    # Determine optimal worker count
    n_workers = args.n_workers or max(1, min(cpu_count() - 1, len(pkl_files)))
    
    # Estimate processing time
    estimated_rate = args.sampling_rate or 32  # Default assumption
    time_estimate = estimate_processing_time(len(pkl_files), n_workers, estimated_rate)
    
    logger.info(f"🚀 PARALLEL MADRID PROCESSING PLAN")
    logger.info(f"Files to process: {len(pkl_files)}")
    logger.info(f"Parallel workers: {n_workers}")
    logger.info(f"Estimated time: {time_estimate['parallel_time_minutes']:.1f} minutes")
    logger.info(f"Expected speedup: {time_estimate['speedup_factor']:.1f}x")
    logger.info(f"ETA: {time.strftime('%H:%M:%S', time.localtime(time_estimate['estimated_completion']))}")
    
    if args.dry_run:
        logger.info("🧪 DRY RUN - No actual processing performed")
        logger.info(f"Files that would be processed:")
        for i, file in enumerate(pkl_files):
            logger.info(f"  {i+1:3d}. {file.name}")
        return 0
    
    # Load or create configuration
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {args.config_file}")
    else:
        config = create_seizure_only_config()
        logger.info("Using default seizure-only configuration")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize parallel processor
    logger.info(f"🔧 Initializing parallel processor with {n_workers} workers...")
    processor = ParallelMadridBatchProcessor(config, n_workers=n_workers)
    
    # Start processing
    start_time = time.time()
    logger.info(f"🚀 Starting parallel processing...")
    
    try:
        processed_files = processor.process_files(
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            max_files=args.max_files
        )
        
        # Final summary
        total_time = time.time() - start_time
        actual_speedup = time_estimate['serial_time_minutes'] * 60 / total_time if total_time > 0 else 0
        
        logger.info(f"\n🎉 PARALLEL PROCESSING COMPLETED!")
        logger.info(f"✅ Successfully processed: {len(processed_files)}/{len(pkl_files)} files")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"⚡ Actual speedup: {actual_speedup:.1f}x")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        # Save processing summary
        summary = {
            'processing_summary': {
                'total_files': len(pkl_files),
                'processed_files': len(processed_files),
                'failed_files': len(pkl_files) - len(processed_files),
                'processing_time_minutes': total_time / 60,
                'n_workers': n_workers,
                'actual_speedup': actual_speedup,
                'files_per_minute': len(processed_files) / (total_time / 60) if total_time > 0 else 0
            },
            'config_used': config,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"📄 Processing summary saved to: {summary_path}")
        
        # Performance recommendations
        if actual_speedup < n_workers * 0.5:
            logger.warning(f"⚠️  Low speedup achieved ({actual_speedup:.1f}x vs {n_workers}x theoretical)")
            logger.warning(f"   Consider reducing workers or checking GPU availability")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("🛑 Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())