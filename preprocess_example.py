#!/usr/bin/env python3
"""
Example script for ECG preprocessing using the ECGPreprocessor class.
"""

from preprocessing import ECGPreprocessor
from pathlib import Path
import sys

def main():
    # 1. Initialize preprocessor with custom parameters
    preprocessor = ECGPreprocessor(
        filter_params={
            'low_freq': 0.5,    # High-pass: remove baseline drift
            'high_freq': 40.0,  # Low-pass: remove noise  
            'order': 4          # Filter order
        },
        downsample_freq=125,    # Target sampling rate
        window_size=30.0,       # 30 second windows
        stride=15.0             # 50% overlap (15s stride)
    )
    
    # 2. Set data path (adjust to your SeizeIT2 dataset location)
    data_path = "ds005873-download"  # Update this path!
    
    # Check if path exists
    if not Path(data_path).exists():
        print(f"Error: Data path {data_path} does not exist!")
        print("Please update the data_path variable with correct SeizeIT2 dataset path")
        return
    

    
    # 4. Batch preprocessing example
    print("\nBatch processing multiple recordings...")
    
    # Nur diese zwei spezifischen Aufnahmen preprocessen
    recordings = [
        ("sub-071", "run-12"),
        ("sub-051", "run-03"),
    ]

    
    
    # Process batch with results saving
    results_path = "./results/preprocessed"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    batch_results = preprocessor.batch_preprocess(
        data_path=data_path,
        recordings=recordings,
        save_path=results_path
    )
    
    print(f"✓ Batch processing completed: {len(batch_results)} recordings")
    
    # 5. Access processed data
    if batch_results:
        first_result = batch_results[0]
        first_channel = first_result['channels'][0]
        
        print(f"\nExample: First window from {first_result['subject_id']}:")
        print(f"  - Window shape: {first_channel['windows'][0].shape}")
        print(f"  - Sampling rate: {first_channel['processed_fs']} Hz")
        print(f"  - Label: {'Seizure' if first_channel['labels'][0] else 'Normal'}")

if __name__ == "__main__":
    main()