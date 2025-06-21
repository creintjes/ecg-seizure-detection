#!/usr/bin/env python3
"""
Example script for running Madrid Batch Processor on seizure-only preprocessed data

This script demonstrates how to use the Madrid Batch Processor with data
preprocessed using preprocess_seizure_only.py

Usage:
    python madrid_seizure_only_example.py --seizure-data-dir /path/to/seizure/data --max-files 5
"""

import argparse
import json
import logging
from pathlib import Path
from madrid_batch_processor import MadridBatchProcessor

def create_seizure_only_config():
    """
    Create configuration optimized for seizure-only data analysis
    """
    return {
        'use_gpu': True,
        'enable_output': False,
        'madrid_parameters': {
            'm_range': {
                'min_length': 80,
                'max_length': 800,
                'step_size': 80
            },
            'analysis_config': {
                'top_k': 5,  # More anomalies for seizure-focused analysis
                'train_test_split_ratio': 0.3,  # Less training data since we know seizure location
                'threshold_percentile': 90  # Lower threshold for higher sensitivity
            },
            'algorithm_settings': {
                'use_gpu': True,
                'downsampling_factor': 1
            }
        },
        'sampling_rate_adaptations': {
            '125hz': {
                'min_length': 80,
                'max_length': 800,
                'step_size': 80
            },
            '32hz': {
                'min_length': 20,
                'max_length': 200,
                'step_size': 20
            },
            '8hz': {
                'min_length': 5,
                'max_length': 50,
                'step_size': 5
            }
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Madrid analysis on seizure-only preprocessed data')
    parser.add_argument('--seizure-data-dir', required=True, help='Directory with seizure-only .pkl files')
    parser.add_argument('--output-dir', default='madrid_seizure_results', help='Output directory for JSON results')
    parser.add_argument('--max-files', type=int, help='Maximum files to process')
    parser.add_argument('--config-file', help='Custom configuration file')
    parser.add_argument('--sampling-rate', type=int, choices=[8, 32, 125], help='Filter by sampling rate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
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
    
    # Initialize processor
    processor = MadridBatchProcessor(config)
    
    # Find seizure-only preprocessed files
    data_dir = Path(args.seizure_data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    pkl_files = list(data_dir.glob("*.pkl"))
    
    if args.sampling_rate:
        # Filter files by sampling rate (if specified in filename)
        filtered_files = []
        for file in pkl_files:
            if f"{args.sampling_rate}hz" in file.name.lower():
                filtered_files.append(file)
        pkl_files = filtered_files
        logger.info(f"Filtered to {len(pkl_files)} files with {args.sampling_rate}Hz sampling rate")
    
    if not pkl_files:
        logger.error(f"No .pkl files found in {data_dir}")
        return
    
    if args.max_files:
        pkl_files = pkl_files[:args.max_files]
    
    logger.info(f"Found {len(pkl_files)} seizure-only files to process")
    
    # Process files
    try:
        processed_files = processor.process_files(
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            max_files=args.max_files
        )
        
        logger.info(f"✅ Successfully processed {len(processed_files)} files")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        # Print summary of what was processed
        logger.info("\n📊 PROCESSING SUMMARY:")
        for file_path in processed_files:
            file_name = Path(file_path).name
            logger.info(f"  ✓ {file_name}")
        
        # Save configuration used
        config_output_path = output_dir / "madrid_config_used.json"
        with open(config_output_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"📄 Configuration saved to: {config_output_path}")
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        raise

if __name__ == "__main__":
    main()