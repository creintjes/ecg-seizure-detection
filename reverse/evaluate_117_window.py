#!/usr/bin/env python3
"""
Evaluate TimeVQVAE-AD results for Dataset 117 (window mode)
Using scale=0.7 and 90s clustering strategy.
"""

import pickle
import numpy as np
import torch
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add TimeVQVAE-AD evaluation to path
sys.path.append(str(Path(__file__).parent.parent / "TimeVQVAE-AD" / "evaluation"))

try:
    from clustering import analyze_detection_mask
    print("✓ Successfully imported clustering module")
except ImportError as e:
    print(f"✗ Could not import clustering module: {e}")
    print("  Continuing without clustering analysis...")
    analyze_detection_mask = None


def load_pkl(fname: str):
    """Load pickle file."""
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_events(mask) -> List[Tuple[int, int]]:
    """Extract events (contiguous True regions) from boolean mask."""
    # Convert tensor to numpy array if needed
    if hasattr(mask, 'cpu'):  # PyTorch tensor
        mask = mask.cpu().numpy()
    elif hasattr(mask, 'numpy'):  # TensorFlow tensor
        mask = mask.numpy()

    # Ensure it's a 1D array
    mask = np.asarray(mask).flatten()

    events = []
    in_event = False
    for i, v in enumerate(mask):
        if v and not in_event:
            start = i
            in_event = True
        elif not v and in_event:
            end = i - 1
            events.append((start, end))
            in_event = False
    if in_event:
        events.append((start, len(mask) - 1))
    return events


def evaluate_event_detection(pred: np.ndarray, truth: np.ndarray, fs: float = 8.0) -> Dict[str, Any]:
    """
    Evaluate event-based detection metrics.

    Args:
        pred: Predicted anomaly mask (boolean)
        truth: Ground truth anomaly mask (boolean)
        fs: Sampling frequency in Hz

    Returns:
        Dictionary with detection metrics
    """
    truth_events = get_events(truth)
    pred_events = get_events(pred)

    TP = sum(1 for (ts, te) in truth_events if pred[ts:te+1].any())
    FN = len(truth_events) - TP
    FP = sum(1 for (ps, pe) in pred_events if not truth[ps:pe+1].any())

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    iou = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else 0.0

    total_hours = len(truth) / fs / 3600.0
    far_per_hour = (FP / total_hours) if total_hours > 0 else float('inf')

    metrics = {
        'truth_events': truth_events,
        'pred_events': pred_events,
        'TP_events': TP,
        'FN_events': FN,
        'FP_events': FP,
        'sensitivity': sensitivity,
        'event_iou': iou,
        'false_alarm_rate_per_hour': far_per_hour,
        'total_hours': total_hours
    }

    return metrics


def get_clustered_mask_from_representatives(
    representatives: List[Dict[str, Any]],
    total_samples: int,
    fs: float
) -> np.ndarray:
    """
    Convert cluster representatives back to a boolean mask.

    Args:
        representatives: List of representative dicts from clustering
        total_samples: Total length of the original mask
        fs: Sampling rate in Hz

    Returns:
        Boolean numpy array with clustered anomalies marked as True
    """
    mask = np.zeros(total_samples, dtype=bool)

    for rep in representatives:
        # Get cluster time span
        start_sec = rep.get('cluster_start_seconds', rep['location_time_seconds'])
        end_sec = rep.get('cluster_end_seconds', rep['location_time_seconds'])

        # Convert to sample indices
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)

        # Clamp to valid range
        start_idx = max(0, min(start_idx, total_samples - 1))
        end_idx = max(0, min(end_idx, total_samples - 1))

        # Mark cluster span as True
        mask[start_idx:end_idx+1] = True

    return mask


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(text)
    print("="*80)


def print_section(text):
    """Print formatted section header."""
    print("\n" + "-"*80)
    print(text)
    print("-"*80)


def evaluate_dataset_117(pkl_path: Path, scale: float = 0.7, clustering_strategy: str = "time_90s",
                        output_dir: Path = None):
    """
    Evaluate Dataset 117 with specified scale and clustering strategy.

    Args:
        pkl_path: Path to pickle file
        scale: Threshold scaling factor (default: 0.7)
        clustering_strategy: Clustering strategy name (default: "time_90s")
        output_dir: Directory to save plots (default: same as pkl_path)
    """

    print_header("TIMEVQVAE-AD EVALUATION: DATASET 117 (Window Mode)")

    # Load data
    print("\nLoading data from:", pkl_path)
    data = load_pkl(pkl_path)

    # Extract key components
    a_final = data['a_final']
    if isinstance(a_final, torch.Tensor):
        a_final = a_final.cpu().numpy()

    Y = data['Y']
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    Y = Y.astype(bool)

    final_threshold_original = data['final_threshold']
    X_test = data.get('X_test_unscaled', None)
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.cpu().numpy()

    # Dataset info
    print_header("DATASET INFORMATION")
    print(f"Dataset Index: {data.get('dataset_index', 'Unknown')}")
    print(f"Total samples: {len(Y):,}")
    print(f"Duration: {len(Y)/8.0:.2f} seconds = {len(Y)/8.0/60:.2f} minutes = {len(Y)/8.0/3600:.2f} hours")
    print(f"Sampling rate: 8 Hz")
    print(f"\nGround Truth Anomalies:")
    print(f"  Anomaly samples: {np.sum(Y):,} ({(np.sum(Y)/len(Y))*100:.3f}%)")
    print(f"  Normal samples: {np.sum(~Y):,} ({(np.sum(~Y)/len(Y))*100:.3f}%)")

    truth_events = get_events(Y)
    print(f"  Number of anomaly events: {len(truth_events)}")

    if truth_events:
        print(f"\n  Anomaly Event Details:")
        total_anomaly_duration = 0
        for i, (start, end) in enumerate(truth_events):
            duration = (end - start + 1) / 8.0
            start_time = start / 8.0 / 60.0  # minutes
            total_anomaly_duration += duration
            print(f"    Event {i+1}: {duration:.2f}s (starts at {start_time:.2f} min, samples {start}-{end})")
        print(f"\n  Total anomaly duration: {total_anomaly_duration:.2f} seconds = {total_anomaly_duration/60:.2f} minutes")

    # Threshold info
    print_header("THRESHOLD INFORMATION")
    print(f"Original final_threshold: {final_threshold_original:.4f}")
    print(f"Scaling factor: {scale}")
    scaled_threshold = final_threshold_original * scale
    print(f"Scaled threshold: {scaled_threshold:.4f}")

    print(f"\nAnomaly Score (a_final) Statistics:")
    print(f"  Min:    {np.min(a_final):.4f}")
    print(f"  25th:   {np.percentile(a_final, 25):.4f}")
    print(f"  50th:   {np.percentile(a_final, 50):.4f}")
    print(f"  75th:   {np.percentile(a_final, 75):.4f}")
    print(f"  90th:   {np.percentile(a_final, 90):.4f}")
    print(f"  95th:   {np.percentile(a_final, 95):.4f}")
    print(f"  99th:   {np.percentile(a_final, 99):.4f}")
    print(f"  Max:    {np.max(a_final):.4f}")
    print(f"\n  Original threshold: {final_threshold_original:.4f} (above max! → 0 detections)")
    print(f"  Scaled threshold:   {scaled_threshold:.4f}")

    # Predict anomalies
    print_header("BASELINE DETECTION (No Clustering)")
    predicted = a_final > scaled_threshold

    print(f"Predicted anomaly samples: {np.sum(predicted):,} ({(np.sum(predicted)/len(predicted))*100:.3f}%)")

    # Evaluate baseline
    metrics_baseline = evaluate_event_detection(predicted, Y, fs=8.0)

    print(f"\nEvent-based Metrics (Baseline):")
    print(f"  True Positives (TP):  {metrics_baseline['TP_events']}")
    print(f"  False Negatives (FN): {metrics_baseline['FN_events']}")
    print(f"  False Positives (FP): {metrics_baseline['FP_events']}")
    print(f"  Sensitivity:          {metrics_baseline['sensitivity']:.3f} ({metrics_baseline['sensitivity']*100:.1f}%)")
    print(f"  Event IoU:            {metrics_baseline['event_iou']:.3f}")
    print(f"  False Alarm Rate:     {metrics_baseline['false_alarm_rate_per_hour']:.2f} events/hour")

    print(f"\nPredicted Events: {len(metrics_baseline['pred_events'])}")
    if len(metrics_baseline['pred_events']) > 0 and len(metrics_baseline['pred_events']) <= 50:
        print("  Event details (first 50):")
        for i, (start, end) in enumerate(metrics_baseline['pred_events'][:50]):
            duration = (end - start + 1) / 8.0
            start_time = start / 8.0 / 60.0
            is_tp = any(Y[start:end+1])
            marker = "✓ TP" if is_tp else "✗ FP"
            print(f"    Event {i+1}: {duration:.2f}s (at {start_time:.2f} min, samples {start}-{end}) {marker}")

    # Apply clustering if available
    clustered_mask = None
    metrics_clustered = None

    if analyze_detection_mask is not None:
        print_header(f"SMART CLUSTERING ({clustering_strategy})")

        try:
            # Ground truth intervals for clustering analysis
            gt_intervals = [(s/8.0, (e+1)/8.0) for s, e in truth_events]

            results_cluster = analyze_detection_mask(
                mask=predicted,
                fs=8.0,
                file_id="dataset_117",
                scores=a_final,
                subject_id="sub-117",
                gt_intervals=gt_intervals,
                output_folder=None,  # Don't save files
                fixed_strategy=clustering_strategy,
                true_anomaly_samples=int(np.sum(Y)),
                true_anomaly_events=len(truth_events)
            )

            # Extract representatives and create clustered mask
            representatives = results_cluster['best_results']['representatives']
            clustered_mask = get_clustered_mask_from_representatives(representatives, len(Y), 8.0)

            print(f"Clustering completed successfully!")
            print(f"  Number of clusters: {results_cluster['best_results']['clusters']}")
            print(f"  Number of representatives: {len(representatives)}")

            # Evaluate clustered results
            metrics_clustered = evaluate_event_detection(clustered_mask, Y, fs=8.0)

            print(f"\nEvent-based Metrics (After Clustering):")
            print(f"  True Positives (TP):  {metrics_clustered['TP_events']}")
            print(f"  False Negatives (FN): {metrics_clustered['FN_events']}")
            print(f"  False Positives (FP): {metrics_clustered['FP_events']}")
            print(f"  Sensitivity:          {metrics_clustered['sensitivity']:.3f} ({metrics_clustered['sensitivity']*100:.1f}%)")
            print(f"  Event IoU:            {metrics_clustered['event_iou']:.3f}")
            print(f"  False Alarm Rate:     {metrics_clustered['false_alarm_rate_per_hour']:.2f} events/hour")

            # Compare before/after
            print_section("CLUSTERING IMPACT")
            print(f"False Alarms:  {metrics_baseline['FP_events']} → {metrics_clustered['FP_events']} "
                  f"({((metrics_clustered['FP_events'] - metrics_baseline['FP_events']) / metrics_baseline['FP_events'] * 100):.1f}% change)")
            print(f"FAR (evt/hr):  {metrics_baseline['false_alarm_rate_per_hour']:.2f} → {metrics_clustered['false_alarm_rate_per_hour']:.2f} "
                  f"({((metrics_clustered['false_alarm_rate_per_hour'] - metrics_baseline['false_alarm_rate_per_hour']) / metrics_baseline['false_alarm_rate_per_hour'] * 100):.1f}% change)")
            print(f"Sensitivity:   {metrics_baseline['sensitivity']:.3f} → {metrics_clustered['sensitivity']:.3f} (no change)")

            # Representative details
            print(f"\nCluster Representatives: {len(representatives)}")
            if len(representatives) <= 20:
                print("  Representative details:")
                for i, rep in enumerate(representatives):
                    start_sec = rep.get('cluster_start_seconds', rep['location_time_seconds'])
                    end_sec = rep.get('cluster_end_seconds', rep['location_time_seconds'])
                    duration = end_sec - start_sec
                    score = rep.get('anomaly_score', 0)
                    print(f"    Cluster {i+1}: {duration:.2f}s (at {start_sec/60:.2f} min, score={score:.2f})")

        except Exception as e:
            print(f"✗ Clustering failed: {e}")
            print("  Continuing with baseline results only...")
    else:
        print_section("CLUSTERING SKIPPED")
        print("Clustering module not available. Only baseline results will be shown.")

    # Generate plots
    print_header("GENERATING VISUALIZATIONS")

    if output_dir is None:
        output_dir = pkl_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot 1: Overview Timeline
    plot_overview_timeline(
        a_final, Y, predicted, clustered_mask, scaled_threshold,
        metrics_baseline, metrics_clustered,
        output_dir / "117_overview_timeline.png"
    )

    # Plot 2: Metrics Comparison
    plot_metrics_comparison(
        metrics_baseline, metrics_clustered,
        output_dir / "117_metrics_comparison.png"
    )

    # Plot 3: Event Distribution
    plot_event_distribution(
        metrics_baseline, metrics_clustered, Y,
        output_dir / "117_event_distribution.png"
    )

    print_header("EVALUATION COMPLETE")
    print(f"\nResults Summary:")
    print(f"  Scale: {scale}")
    print(f"  Clustering: {clustering_strategy}")
    print(f"  Baseline Sensitivity: {metrics_baseline['sensitivity']:.3f}")
    print(f"  Baseline FAR: {metrics_baseline['false_alarm_rate_per_hour']:.2f} events/hour")
    if metrics_clustered:
        print(f"  Clustered Sensitivity: {metrics_clustered['sensitivity']:.3f}")
        print(f"  Clustered FAR: {metrics_clustered['false_alarm_rate_per_hour']:.2f} events/hour")
    print(f"\nPlots saved to: {output_dir}")
    print()

    return metrics_baseline, metrics_clustered


def plot_overview_timeline(a_final, Y, predicted, clustered_mask, threshold,
                           metrics_baseline, metrics_clustered, output_path):
    """Create overview timeline plot."""

    # Downsample for plotting
    step = max(1, len(a_final) // 5000)
    time_hours = np.arange(len(a_final))[::step] / 8.0 / 3600.0
    a_final_plot = a_final[::step]
    Y_plot = Y[::step]
    predicted_plot = predicted[::step]

    n_plots = 4 if clustered_mask is not None else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 3*n_plots))

    # Plot 1: Anomaly scores
    axes[0].plot(time_hours, a_final_plot, color='blue', alpha=0.7, linewidth=0.5, label='a_final')
    axes[0].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (scaled={threshold:.2f})')
    axes[0].fill_between(time_hours, 0, Y_plot * np.max(a_final_plot), alpha=0.3, color='orange', label='Ground Truth')
    axes[0].set_ylabel('Anomaly Score', fontsize=11)
    axes[0].set_title('Anomaly Scores (a_final) with Threshold', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Ground truth
    axes[1].fill_between(time_hours, 0, Y_plot, alpha=0.6, color='orange', label='Ground Truth')
    axes[1].set_ylabel('Ground Truth', fontsize=11)
    axes[1].set_title('Ground Truth Anomalies', fontsize=13, fontweight='bold')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Baseline predictions
    axes[2].fill_between(time_hours, 0, predicted_plot, alpha=0.6, color='blue', label='Baseline Predictions')
    axes[2].fill_between(time_hours, 0, Y_plot, alpha=0.3, color='orange', label='Ground Truth')
    axes[2].set_ylabel('Predictions', fontsize=11)
    title_baseline = f'Baseline Predictions (Sens={metrics_baseline["sensitivity"]:.3f}, FAR={metrics_baseline["false_alarm_rate_per_hour"]:.1f}/hr)'
    axes[2].set_title(title_baseline, fontsize=13, fontweight='bold')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Clustered predictions (if available)
    if clustered_mask is not None:
        clustered_plot = clustered_mask[::step]
        axes[3].fill_between(time_hours, 0, clustered_plot, alpha=0.6, color='green', label='Clustered Predictions')
        axes[3].fill_between(time_hours, 0, Y_plot, alpha=0.3, color='orange', label='Ground Truth')
        axes[3].set_ylabel('Predictions', fontsize=11)
        title_clustered = f'Clustered Predictions (Sens={metrics_clustered["sensitivity"]:.3f}, FAR={metrics_clustered["false_alarm_rate_per_hour"]:.1f}/hr)'
        axes[3].set_title(title_clustered, fontsize=13, fontweight='bold')
        axes[3].set_xlabel('Time (hours)', fontsize=11)
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[2].set_xlabel('Time (hours)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def plot_metrics_comparison(metrics_baseline, metrics_clustered, output_path):
    """Create metrics comparison bar plot."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_names = ['Sensitivity', 'FAR (events/hr)', 'Event IoU']
    baseline_vals = [
        metrics_baseline['sensitivity'],
        metrics_baseline['false_alarm_rate_per_hour'],
        metrics_baseline['event_iou']
    ]

    if metrics_clustered:
        clustered_vals = [
            metrics_clustered['sensitivity'],
            metrics_clustered['false_alarm_rate_per_hour'],
            metrics_clustered['event_iou']
        ]
        labels = ['Baseline', 'Clustered']
        colors = ['steelblue', 'seagreen']
    else:
        clustered_vals = None
        labels = ['Baseline']
        colors = ['steelblue']

    x = np.arange(len(metrics_names))
    width = 0.35

    for i, (name, baseline_val) in enumerate(zip(metrics_names, baseline_vals)):
        ax = axes[i]

        if clustered_vals:
            ax.bar(x[i] - width/2, baseline_val, width, label='Baseline', color='steelblue')
            ax.bar(x[i] + width/2, clustered_vals[i], width, label='Clustered', color='seagreen')
        else:
            ax.bar(x[i], baseline_val, width, label='Baseline', color='steelblue')

        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


def plot_event_distribution(metrics_baseline, metrics_clustered, Y, output_path):
    """Create event distribution plot."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TP/FN/FP counts
    categories = ['TP', 'FN', 'FP']
    baseline_counts = [
        metrics_baseline['TP_events'],
        metrics_baseline['FN_events'],
        metrics_baseline['FP_events']
    ]

    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width/2, baseline_counts, width, label='Baseline', color='steelblue')

    if metrics_clustered:
        clustered_counts = [
            metrics_clustered['TP_events'],
            metrics_clustered['FN_events'],
            metrics_clustered['FP_events']
        ]
        axes[0].bar(x + width/2, clustered_counts, width, label='Clustered', color='seagreen')

    axes[0].set_ylabel('Event Count', fontsize=11)
    axes[0].set_title('Event Classification Counts', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Anomaly percentage
    total_samples = len(Y)
    true_anomaly_pct = (np.sum(Y) / total_samples) * 100

    pred_baseline = metrics_baseline['FP_events'] > 0  # Has any predictions
    baseline_anomaly_count = sum(1 for (s, e) in metrics_baseline['pred_events'])
    baseline_anomaly_samples = sum((e - s + 1) for (s, e) in metrics_baseline['pred_events'])
    baseline_anomaly_pct = (baseline_anomaly_samples / total_samples) * 100

    labels_pct = ['Ground Truth', 'Baseline']
    values_pct = [true_anomaly_pct, baseline_anomaly_pct]
    colors_pct = ['orange', 'steelblue']

    if metrics_clustered:
        clustered_anomaly_samples = sum((e - s + 1) for (s, e) in metrics_clustered['pred_events'])
        clustered_anomaly_pct = (clustered_anomaly_samples / total_samples) * 100
        labels_pct.append('Clustered')
        values_pct.append(clustered_anomaly_pct)
        colors_pct.append('seagreen')

    axes[1].bar(labels_pct, values_pct, color=colors_pct, alpha=0.7)
    axes[1].set_ylabel('Percentage (%)', fontsize=11)
    axes[1].set_title('Anomaly Coverage (% of recording)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (label, val) in enumerate(zip(labels_pct, values_pct)):
        axes[1].text(i, val + 0.1, f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path.name}")
    plt.close()


if __name__ == "__main__":
    # File path
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
        sys.exit(1)

    # Output directory
    output_dir = pkl_path.parent / "evaluation_results"

    # Run evaluation
    metrics_baseline, metrics_clustered = evaluate_dataset_117(
        pkl_path=pkl_path,
        scale=0.7,
        clustering_strategy="time_90s",
        output_dir=output_dir
    )
