#!/usr/bin/env python3
"""
Analyze TimeVQVAE-AD pickle file structure and print key information.
"""

import pickle
import numpy as np
import torch
from pathlib import Path


def analyze_pkl_structure(pkl_path):
    """Load and analyze the structure of a TimeVQVAE-AD pickle file."""

    print("="*80)
    print(f"ANALYZING: {pkl_path}")
    print("="*80)

    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\n{'='*80}")
    print("1. TOP-LEVEL STRUCTURE")
    print(f"{'='*80}")
    print(f"Type: {type(data)}")

    if isinstance(data, dict):
        print(f"Number of keys: {len(data.keys())}")
        print(f"Keys: {list(data.keys())}\n")

    # Analyze each key
    print(f"{'='*80}")
    print("2. DETAILED KEY ANALYSIS")
    print(f"{'='*80}\n")

    for key, value in data.items():
        print(f"--- KEY: '{key}' ---")
        print(f"  Type: {type(value)}")

        if isinstance(value, (np.ndarray, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                value_np = value.cpu().numpy()
            else:
                value_np = value

            print(f"  Shape: {value_np.shape}")
            print(f"  Dtype: {value_np.dtype}")
            print(f"  Size (elements): {value_np.size:,}")
            print(f"  Min: {np.min(value_np):.4f}")
            print(f"  Max: {np.max(value_np):.4f}")
            print(f"  Mean: {np.mean(value_np):.4f}")
            print(f"  Std: {np.std(value_np):.4f}")

            # Count unique values for small arrays or boolean/integer arrays
            if value_np.size < 1000 or value_np.dtype in [bool, np.bool_, np.int32, np.int64]:
                unique_vals = np.unique(value_np)
                print(f"  Unique values: {len(unique_vals)}")
                if len(unique_vals) <= 10:
                    print(f"    Values: {unique_vals}")
                    # Count occurrences
                    for uv in unique_vals:
                        count = np.sum(value_np == uv)
                        pct = (count / value_np.size) * 100
                        print(f"      {uv}: {count:,} ({pct:.2f}%)")

            # Show first few elements for 1D arrays
            if len(value_np.shape) == 1 and value_np.size <= 20:
                print(f"  Values: {value_np}")
            elif len(value_np.shape) == 1:
                print(f"  First 5 values: {value_np[:5]}")
                print(f"  Last 5 values: {value_np[-5:]}")

        elif isinstance(value, (list, tuple)):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                if len(value) <= 5:
                    print(f"  Values: {value}")
                else:
                    print(f"  First 3: {value[:3]}")
                    print(f"  Last 3: {value[-3:]}")

        elif isinstance(value, (int, float, np.number)):
            print(f"  Value: {value}")

        elif isinstance(value, str):
            print(f"  Value: '{value}'")

        elif isinstance(value, range):
            print(f"  Range: start={value.start}, stop={value.stop}, step={value.step}")
            print(f"  Length: {len(value)}")

        else:
            print(f"  Value: {value}")

        print()

    # Calculate derived information
    print(f"{'='*80}")
    print("3. DERIVED INFORMATION")
    print(f"{'='*80}\n")

    # Check for common TimeVQVAE-AD keys
    if 'Y' in data:
        Y = data['Y']
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()
        print("GROUND TRUTH LABELS (Y):")
        total_samples = len(Y)
        anomaly_samples = np.sum(Y == 1)
        normal_samples = np.sum(Y == 0)
        print(f"  Total samples: {total_samples:,}")
        print(f"  Anomaly samples: {anomaly_samples:,} ({(anomaly_samples/total_samples)*100:.3f}%)")
        print(f"  Normal samples: {normal_samples:,} ({(normal_samples/total_samples)*100:.3f}%)")
        print()

    if 'a_final' in data:
        a_final = data['a_final']
        if isinstance(a_final, torch.Tensor):
            a_final = a_final.cpu().numpy()
        print("FINAL ANOMALY SCORES (a_final):")
        print(f"  Length: {len(a_final):,}")
        print(f"  Min score: {np.min(a_final):.4f}")
        print(f"  Max score: {np.max(a_final):.4f}")
        print(f"  Mean score: {np.mean(a_final):.4f}")
        print(f"  Median score: {np.median(a_final):.4f}")
        print(f"  Std score: {np.std(a_final):.4f}")
        print(f"  Percentiles:")
        for p in [50, 75, 90, 95, 99]:
            print(f"    {p}th: {np.percentile(a_final, p):.4f}")
        print()

    if 'final_threshold' in data:
        final_threshold = data['final_threshold']
        print("DETECTION THRESHOLD:")
        print(f"  Final threshold: {final_threshold:.4f}")

        if 'a_final' in data:
            a_final = data['a_final']
            if isinstance(a_final, torch.Tensor):
                a_final = a_final.cpu().numpy()
            detections = np.sum(a_final > final_threshold)
            print(f"  Samples above threshold: {detections:,} ({(detections/len(a_final))*100:.3f}%)")
        print()

    if 'joint_threshold' in data:
        joint_threshold = data['joint_threshold']
        if isinstance(joint_threshold, (list, np.ndarray)):
            print("JOINT THRESHOLDS (per frequency band):")
            for i, thresh in enumerate(joint_threshold):
                print(f"  Band {i}: {thresh:.4f}")
        print()

    if 'a_s^*' in data:
        a_s_star = data['a_s^*']
        if isinstance(a_s_star, torch.Tensor):
            a_s_star = a_s_star.cpu().numpy()
        print("FREQUENCY-SPECIFIC ANOMALY SCORES (a_s^*):")
        print(f"  Shape: {a_s_star.shape} (num_freq_bands x num_samples)")
        print(f"  Number of frequency bands: {a_s_star.shape[0]}")
        for i in range(a_s_star.shape[0]):
            print(f"  Band {i}: min={np.min(a_s_star[i]):.4f}, max={np.max(a_s_star[i]):.4f}, mean={np.mean(a_s_star[i]):.4f}")
        print()

    if 'X_test_unscaled' in data:
        X = data['X_test_unscaled']
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        print("ECG SIGNAL (X_test_unscaled):")
        print(f"  Length: {len(X):,}")

        # Estimate recording duration
        if 'dataset_index' in data:
            # Assuming 8 Hz sampling rate (from config)
            duration_seconds = len(X) / 8.0
            duration_minutes = duration_seconds / 60.0
            duration_hours = duration_minutes / 60.0
            print(f"  Duration (8 Hz): {duration_seconds:.1f}s = {duration_minutes:.1f}min = {duration_hours:.2f}h")
        print()

    # Memory usage estimation
    print(f"{'='*80}")
    print("4. MEMORY USAGE")
    print(f"{'='*80}\n")

    total_bytes = 0
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            size = value.nbytes
            total_bytes += size
            print(f"  {key}: {size / (1024**2):.2f} MB")
        elif isinstance(value, torch.Tensor):
            size = value.element_size() * value.nelement()
            total_bytes += size
            print(f"  {key}: {size / (1024**2):.2f} MB")

    print(f"\n  Total data size: {total_bytes / (1024**2):.2f} MB")
    print()

    # Summary
    print(f"{'='*80}")
    print("5. QUICK SUMMARY")
    print(f"{'='*80}\n")

    if 'dataset_index' in data:
        print(f"Dataset: {data['dataset_index']}")

    if 'X_test_unscaled' in data and 'Y' in data:
        X = data['X_test_unscaled']
        Y = data['Y']
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()

        print(f"Recording length: {len(X):,} samples")
        print(f"True anomalies: {np.sum(Y == 1):,} samples ({(np.sum(Y == 1)/len(Y))*100:.3f}%)")

    if 'a_final' in data and 'final_threshold' in data:
        a_final = data['a_final']
        if isinstance(a_final, torch.Tensor):
            a_final = a_final.cpu().numpy()
        final_threshold = data['final_threshold']

        predicted = a_final > final_threshold
        print(f"Predicted anomalies: {np.sum(predicted):,} samples ({(np.sum(predicted)/len(predicted))*100:.3f}%)")

    print("\n" + "="*80)


if __name__ == "__main__":
    pkl_path = Path(__file__).parent / "117_window-joint_anomaly_score.pkl" / "117_window-joint_anomaly_score.pkl"

    if not pkl_path.exists():
        # Try without nested folder
        pkl_path = Path(__file__).parent / "117_window-joint_anomaly_score.pkl"

    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        print("\nAvailable files in reverse folder:")
        reverse_dir = Path(__file__).parent
        for f in reverse_dir.iterdir():
            print(f"  {f.name}")
    else:
        analyze_pkl_structure(pkl_path)
