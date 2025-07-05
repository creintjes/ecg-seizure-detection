#!/usr/bin/env python3
"""
Debug script to examine the structure of preprocessed files.
"""

import pickle
import json
from pathlib import Path

def analyze_preprocessed_file(file_path):
    """Analyze structure of a preprocessed file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n=== ANALYZING: {Path(file_path).name} ===")
        print(f"Top-level keys: {list(data.keys())}")
        
        if 'channels' in data:
            print(f"Number of channels: {len(data['channels'])}")
            
            for i, channel in enumerate(data['channels']):
                print(f"\nChannel {i}:")
                print(f"  Keys: {list(channel.keys())}")
                
                if 'windows' in channel:
                    windows = channel['windows']
                    print(f"  Number of windows: {len(windows)}")
                    if len(windows) > 0:
                        print(f"  First window type: {type(windows[0])}")
                        print(f"  First window shape: {windows[0].shape if hasattr(windows[0], 'shape') else 'No shape'}")
                        
                        # Check for windowed structure
                        if len(windows) > 1:
                            print(f"  MULTIPLE WINDOWS DETECTED - This is windowed format!")
                        else:
                            print(f"  Single window - This is NOT windowed format")
                
                if 'labels' in channel:
                    labels = channel['labels']
                    print(f"  Number of labels: {len(labels)}")
                    if len(labels) > 0:
                        print(f"  First label type: {type(labels[0])}")
                        if hasattr(labels[0], 'shape'):
                            print(f"  First label shape: {labels[0].shape}")
                        else:
                            print(f"  First label value: {labels[0]}")
                
                if 'metadata' in channel:
                    metadata = channel['metadata']
                    print(f"  Number of metadata entries: {len(metadata)}")
                    if len(metadata) > 0:
                        print(f"  First metadata keys: {list(metadata[0].keys())}")
        
        if 'preprocessing_params' in data:
            params = data['preprocessing_params']
            print(f"\nPreprocessing parameters:")
            print(f"  Window size: {params.get('window_size', 'Not specified')}")
            print(f"  Stride: {params.get('stride', 'Not specified')}")
            print(f"  Downsample freq: {params.get('downsample_freq', 'Not specified')}")
        
        return data
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def main():
    """Main function to analyze preprocessed files"""
    print("DEBUGGING PREPROCESSED FILE STRUCTURE")
    print("=" * 50)
    
    # Test files
    test_files = [
        "/mnt/c/Users/Reintjes/Documents/aD/UoC/Semester2/ASIM/Projekt/ecg-seizure-detection/results/preprocessed/sub-001_run-01_preprocessed.pkl",
        "/mnt/c/Users/Reintjes/Documents/aD/UoC/Semester2/ASIM/Projekt/ecg-seizure-detection/results/preprocessed_all/sub-001_run-01_preprocessed.pkl"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            analyze_preprocessed_file(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()