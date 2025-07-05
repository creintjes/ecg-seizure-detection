#!/usr/bin/env python3
"""
Test script to debug windowed format detection
"""

import os
import sys
import pickle
from pathlib import Path

# Add Madrid directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Madrid'))

from madrid_windowed_batch_processor_parallel import MadridWindowedBatchProcessorCore

def test_windowed_format_detection():
    """Test windowed format detection on sample files"""
    
    # Create processor core for testing
    config = {
        'use_gpu': False,
        'enable_output': False,
        'window_strategy': 'individual',
        'madrid_parameters': {
            'm_range': {
                'min_length': 80,
                'max_length': 800,
                'step_size': 80
            },
            'analysis_config': {
                'top_k': 3,
                'train_test_split_ratio': 0.5,
                'train_minutes': 30,
                'threshold_percentile': 95
            }
        }
    }
    
    processor = MadridWindowedBatchProcessorCore(config)
    
    # Test files
    test_files = [
        "/mnt/c/Users/Reintjes/Documents/aD/UoC/Semester2/ASIM/Projekt/ecg-seizure-detection/results/preprocessed/sub-001_run-01_preprocessed.pkl",
        "/mnt/c/Users/Reintjes/Documents/aD/UoC/Semester2/ASIM/Projekt/ecg-seizure-detection/results/preprocessed_all/sub-001_run-01_preprocessed.pkl"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"\n=== Testing: {Path(file_path).name} ===")
            
            try:
                # Load file
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Test windowed format detection
                is_windowed = processor.is_windowed_format(data)
                print(f"Is windowed format: {is_windowed}")
                
                # Get detailed analysis
                analysis = processor.analyze_file_format(data, file_path)
                print(f"Analysis: {analysis}")
                
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    test_windowed_format_detection()