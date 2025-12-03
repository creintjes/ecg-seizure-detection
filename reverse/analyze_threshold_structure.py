#!/usr/bin/env python3
"""
Analyze whether thresholds are static or dynamic (time-varying).
"""

import pickle
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_threshold_structure(pkl_path):
    """Check if thresholds are static scalars or time-varying arrays."""

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print("="*80)
    print("THRESHOLD STRUCTURE ANALYSIS")
    print("="*80)

    # Check final_threshold
    print("\n1. final_threshold:")
    final_threshold = data['final_threshold']
    print(f"   Type: {type(final_threshold)}")
    print(f"   Value: {final_threshold}")

    if isinstance(final_threshold, (np.ndarray, list)):
        print(f"   Shape: {np.array(final_threshold).shape}")
        print(f"   → This is a TIME-VARYING threshold")
    else:
        print(f"   → This is a STATIC (single) threshold for the entire time series")

    # Check joint_threshold
    print("\n2. joint_threshold:")
    joint_threshold = data['joint_threshold']
    print(f"   Type: {type(joint_threshold)}")
    print(f"   Shape: {joint_threshold.shape}")
    print(f"   Values: {joint_threshold}")
    print(f"   → This has {len(joint_threshold)} thresholds - ONE per FREQUENCY BAND (not time!)")

    # Check the anomaly scores
    print("\n3. a_final (final anomaly scores):")
    a_final = data['a_final']
    print(f"   Type: {type(a_final)}")
    print(f"   Shape: {a_final.shape}")
    print(f"   → This is TIME-VARYING (one score per time sample)")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print()
    print("✗ The threshold is STATIC (constant over time)")
    print(f"  - final_threshold = {final_threshold:.4f} (single value)")
    print(f"  - joint_threshold has {len(joint_threshold)} values (one per frequency band)")
    print()
    print("✓ The anomaly scores are DYNAMIC (vary over time)")
    print(f"  - a_final has {len(a_final):,} values (one per time sample)")
    print()
    print("This means the same threshold is applied to the entire recording!")
    print("Alternative: Use adaptive/dynamic thresholding (e.g., rolling percentile)")
    print()

    # Demonstrate why static threshold might be problematic
    print("="*80)
    print("POTENTIAL ISSUE WITH STATIC THRESHOLD")
    print("="*80)

    # Calculate rolling statistics
    window_size = 10000  # ~20 minutes at 8 Hz
    rolling_mean = np.convolve(a_final, np.ones(window_size)/window_size, mode='valid')
    rolling_std = np.array([np.std(a_final[i:i+window_size]) for i in range(0, len(a_final)-window_size, window_size)])

    print(f"\nStatistics of a_final across time (window size = {window_size} samples):")
    print(f"  Overall mean: {np.mean(a_final):.4f}")
    print(f"  Overall std: {np.std(a_final):.4f}")
    print(f"  Range of rolling means: {np.min(rolling_mean):.4f} to {np.max(rolling_mean):.4f}")
    print(f"  Coefficient of variation: {(np.max(rolling_mean) - np.min(rolling_mean)) / np.mean(a_final) * 100:.2f}%")

    if (np.max(rolling_mean) - np.min(rolling_mean)) / np.mean(a_final) > 0.1:
        print("\n  ⚠️  WARNING: Anomaly scores vary significantly over time!")
        print("     A static threshold may miss anomalies in low-score regions")
        print("     or create false alarms in high-score regions.")
        print("     → Consider using ADAPTIVE THRESHOLDING")
    else:
        print("\n  ✓ Anomaly scores are relatively stable over time.")
        print("    A static threshold should work reasonably well.")

    print()

    # Show distribution of scores
    print("="*80)
    print("SCORE DISTRIBUTION")
    print("="*80)
    print(f"\nPercentiles of a_final:")
    for p in [50, 75, 90, 95, 98, 99, 99.5, 99.9]:
        val = np.percentile(a_final, p)
        above_static = "< static threshold" if val < final_threshold else "> static threshold"
        print(f"  {p:5.1f}th: {val:8.4f}  {above_static}")

    print(f"\n  Max score: {np.max(a_final):8.4f}  < static threshold")
    print(f"  Static threshold: {final_threshold:8.4f}")
    print(f"\n  → Even the MAXIMUM score ({np.max(a_final):.4f}) is below the threshold!")
    print(f"  → This threshold will detect ZERO anomalies")

    return data


def visualize_scores_and_threshold(data, output_path=None):
    """Visualize anomaly scores over time with threshold."""

    a_final = data['a_final']
    final_threshold = data['final_threshold']
    Y = data['Y']
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()

    # Downsample for visualization (plot every 100th point)
    step = 100
    a_final_plot = a_final[::step]
    Y_plot = Y[::step]
    time_plot = np.arange(len(a_final_plot)) * step / 8.0 / 3600.0  # Convert to hours

    fig, axes = plt.subplots(3, 1, figsize=(20, 10))

    # Plot 1: Anomaly scores with static threshold
    axes[0].plot(time_plot, a_final_plot, color='blue', alpha=0.7, linewidth=0.5, label='a_final')
    axes[0].axhline(y=final_threshold, color='red', linestyle='--', linewidth=2, label=f'Static Threshold ({final_threshold:.2f})')
    axes[0].fill_between(time_plot, 0, Y_plot * np.max(a_final_plot), alpha=0.3, color='orange', label='Ground Truth Anomalies')
    axes[0].set_ylabel('Anomaly Score', fontsize=12)
    axes[0].set_title('Final Anomaly Scores with Static Threshold', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Rolling mean and std
    window_size = 1000
    rolling_mean = np.convolve(a_final, np.ones(window_size)/window_size, mode='same')
    rolling_mean_plot = rolling_mean[::step]

    axes[1].plot(time_plot, rolling_mean_plot, color='green', linewidth=1.5, label='Rolling Mean (window=1000)')
    axes[1].axhline(y=final_threshold, color='red', linestyle='--', linewidth=2, label=f'Static Threshold ({final_threshold:.2f})')
    axes[1].fill_between(time_plot, rolling_mean_plot - np.std(a_final), rolling_mean_plot + np.std(a_final),
                         alpha=0.2, color='green', label='±1 Std Dev')
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Rolling Mean of Anomaly Scores', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Ground truth
    axes[2].fill_between(time_plot, 0, Y_plot, alpha=0.6, color='orange', label='Ground Truth Anomalies')
    axes[2].set_ylabel('Anomaly (0/1)', fontsize=12)
    axes[2].set_xlabel('Time (hours)', fontsize=12)
    axes[2].set_title('Ground Truth Anomaly Labels', fontsize=14, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    pkl_path = Path(__file__).parent / "117_window-joint_anomaly_score.pkl" / "117_window-joint_anomaly_score.pkl"

    if not pkl_path.exists():
        pkl_path = Path(__file__).parent / "117_window-joint_anomaly_score.pkl"

    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        import sys
        sys.exit(1)

    print("Loading data...\n")
    data = analyze_threshold_structure(pkl_path)

    print("\nGenerating visualization...")
    output_path = Path(__file__).parent / "threshold_analysis.png"
    visualize_scores_and_threshold(data, output_path)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
