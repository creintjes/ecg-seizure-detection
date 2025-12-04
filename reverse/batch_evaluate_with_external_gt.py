#!/usr/bin/env python3
"""
Batch evaluation of TimeVQVAE-AD results with optional external ground truth.
Processes multiple joint_anomaly_score.pkl files and generates:
- CSV with all metrics
- Per-seizure detection CSV
- Individual plots per file
- Summary report (TXT)

This variant allows using ground truth labels (Y) from a different directory.
Useful when the same ground truth applies to different anomaly score configurations.
"""

import pickle
import numpy as np
import torch
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel processing
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import warnings

# Add TimeVQVAE-AD evaluation to path
sys.path.append(str(Path(__file__).parent.parent / "TimeVQVAE-AD" / "evaluation"))

try:
    from clustering import analyze_detection_mask
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    warnings.warn("Clustering module not available. Clustering will be skipped.")


def load_pkl(fname: str):
    """Load pickle file."""
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_events(mask) -> List[Tuple[int, int]]:
    """Extract events (contiguous True regions) from boolean mask."""
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()
    elif hasattr(mask, 'numpy'):
        mask = mask.numpy()
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


def evaluate_event_detection(pred: np.ndarray, truth: np.ndarray, fs: float = 8.0,
                            truth_events: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """
    Evaluate event-based detection metrics.

    Args:
        pred: Prediction mask
        truth: Ground truth mask (may be expanded)
        fs: Sampling rate
        truth_events: Optional pre-computed truth events (for expanded labels with original positions)

    Returns:
        Dictionary with evaluation metrics
    """
    # Use provided truth_events or extract from truth mask
    if truth_events is None:
        truth_events = get_events(truth)

    pred_events = get_events(pred)

    TP = sum(1 for (ts, te) in truth_events if pred[ts:te+1].any())
    FN = len(truth_events) - TP
    FP = sum(1 for (ps, pe) in pred_events if not truth[ps:pe+1].any())

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    iou = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else 0.0

    total_hours = len(truth) / fs / 3600.0
    far_per_hour = (FP / total_hours) if total_hours > 0 else float('inf')

    return {
        'truth_events': truth_events,
        'pred_events': pred_events,
        'TP_events': TP,
        'FN_events': FN,
        'FP_events': FP,
        'sensitivity': sensitivity,
        'event_iou': iou,
        'false_alarm_rate_per_hour': far_per_hour,
        'total_hours': total_hours,
        'n_truth_events': len(truth_events),
        'n_pred_events': len(pred_events)
    }


def get_clustered_mask_from_representatives(representatives: List[Dict[str, Any]],
                                            total_samples: int, fs: float) -> np.ndarray:
    """Convert cluster representatives back to a boolean mask."""
    mask = np.zeros(total_samples, dtype=bool)
    for rep in representatives:
        start_sec = rep.get('cluster_start_seconds', rep['location_time_seconds'])
        end_sec = rep.get('cluster_end_seconds', rep['location_time_seconds'])
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        start_idx = max(0, min(start_idx, total_samples - 1))
        end_idx = max(0, min(end_idx, total_samples - 1))
        mask[start_idx:end_idx+1] = True
    return mask


def extract_subject_from_filename(filename: str) -> str:
    """Extract subject ID from filename as fallback."""
    # Try patterns like "117_window..." or "sub-117_..."
    import re
    match = re.search(r'(?:sub-)?(\d{3})', filename)
    if match:
        return f"sub-{match.group(1)}"
    return "unknown"


def extract_dataset_index(pkl_path: Path) -> str:
    """Extract dataset_index from pkl file or filename."""
    try:
        data = load_pkl(pkl_path)
        dataset_index = data.get('dataset_index', None)
        if dataset_index:
            return str(dataset_index)
    except:
        pass

    # Fallback: extract from filename
    import re
    match = re.search(r'(\d{3})', pkl_path.stem)
    if match:
        return match.group(1)
    return None


def expand_labels_keep_seizure_events(labels: np.ndarray,
                                      fs: float = 8.0,
                                      pre_minutes: float = 5.0,
                                      post_minutes: float = 3.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Expand ground truth labels temporally while preserving original seizure event positions.

    Expands each seizure by pre_minutes before and post_minutes after.
    Overlapping expanded regions are allowed (merged in the output labels),
    but original seizure positions are tracked separately.

    Args:
        labels: Original boolean labels (1D array)
        fs: Sampling rate in Hz (default: 8.0)
        pre_minutes: Minutes to expand before seizure onset
        post_minutes: Minutes to expand after seizure end

    Returns:
        Tuple of (expanded_labels, original_seizure_events)
        - expanded_labels: Boolean array with temporally expanded labels
        - original_seizure_events: List of (start, end) tuples for original seizures
    """
    labels = np.asarray(labels).astype(bool)
    expanded = np.zeros_like(labels, dtype=bool)

    # Extract original seizure events
    original_events = get_events(labels)

    # Calculate expansion in samples
    pre_samples = int(pre_minutes * 60 * fs)
    post_samples = int(post_minutes * 60 * fs)

    n = len(labels)

    # Expand each seizure event
    for start, end in original_events:
        # Expand boundaries
        expanded_start = max(0, start - pre_samples)
        expanded_end = min(n - 1, end + post_samples)

        # Mark expanded region (overlaps are OK, will be merged)
        expanded[expanded_start:expanded_end + 1] = True

    return expanded, original_events


def load_ground_truth_from_external(gt_dir: Path, dataset_index: str, pattern: str = '*joint_anomaly_score.pkl') -> Optional[Tuple[np.ndarray, List[Tuple[int, int]]]]:
    """
    Load ground truth Y from external directory based on dataset_index matching.
    Applies temporal expansion (5 min pre, 3 min post) to the labels.

    Args:
        gt_dir: Directory containing ground truth pkl files
        dataset_index: Dataset index to match (e.g., "117")
        pattern: File pattern to search for

    Returns:
        Tuple of (expanded_Y, original_events) if found, None otherwise
        - expanded_Y: Temporally expanded boolean labels
        - original_events: List of (start, end) tuples for original seizures
    """
    # Search for files matching the dataset_index
    for gt_file in gt_dir.rglob(pattern):
        gt_dataset_index = extract_dataset_index(gt_file)

        if gt_dataset_index == dataset_index:
            # Found matching file
            try:
                gt_data = load_pkl(gt_file)
                Y = gt_data.get('Y', None)

                if Y is not None:
                    if isinstance(Y, torch.Tensor):
                        Y = Y.cpu().numpy()
                    Y = Y.astype(bool)

                    # Expand labels while keeping track of original events
                    expanded_Y, original_events = expand_labels_keep_seizure_events(
                        Y, fs=8.0, pre_minutes=5.0, post_minutes=3.0
                    )

                    return expanded_Y, original_events
                else:
                    warnings.warn(f"GT file {gt_file.name} has no 'Y' key")
                    return None

            except Exception as e:
                warnings.warn(f"Failed to load GT from {gt_file.name}: {e}")
                return None

    return None


def process_file_worker(args: Tuple) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Worker function for parallel processing of files.

    Args:
        args: Tuple of (pkl_path, scale, clustering_strategy, output_dir, save_plots, gt_dir)

    Returns:
        Tuple of (file_metrics_dict, seizure_detections_list)
    """
    pkl_path, scale, clustering_strategy, output_dir, save_plots, gt_dir = args
    return evaluate_single_file(pkl_path, scale, clustering_strategy, output_dir, save_plots, gt_dir)


def check_seizure_detections(truth_events: List[Tuple[int, int]],
                             clustered_mask: np.ndarray,
                             subject_id: str) -> List[Dict[str, Any]]:
    """
    Check which ground truth seizures were detected by clustered predictions.

    Returns:
        List of dicts with seizure-level detection results
    """
    seizure_detections = []

    for seizure_idx, (start, end) in enumerate(truth_events):
        # Check if any clustered prediction overlaps with this GT seizure
        detected = bool(clustered_mask[start:end+1].any())

        seizure_detections.append({
            'subject': subject_id,
            'seizure_idx': seizure_idx,
            'detected': detected
        })

    return seizure_detections


def evaluate_single_file(pkl_path: Path, scale: float, clustering_strategy: str,
                         output_dir: Path, save_plots: bool = True,
                         gt_dir: Optional[Path] = None) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluate a single pickle file.

    Args:
        pkl_path: Path to score pkl file
        scale: Threshold scaling factor
        clustering_strategy: Clustering strategy name
        output_dir: Output directory
        save_plots: Whether to save plots
        gt_dir: Optional directory to load ground truth Y from

    Returns:
        Tuple of (file_metrics_dict, seizure_detections_list)
        file_metrics_dict: Dictionary with evaluation results, or None if failed
        seizure_detections_list: List of per-seizure detection results
    """
    try:
        # Load data
        data = load_pkl(pkl_path)

        # Extract components
        a_final = data['a_final']
        if isinstance(a_final, torch.Tensor):
            a_final = a_final.cpu().numpy()

        # Load Y - either from external GT dir or from this file
        dataset_index = data.get('dataset_index', pkl_path.stem.split('_')[0])
        original_truth_events = None  # Will store original seizure positions

        if gt_dir is not None:
            # Try to load Y from external GT directory (with expansion)
            result = load_ground_truth_from_external(gt_dir, str(dataset_index))

            if result is not None:
                Y, original_truth_events = result
                # Verify length matches
                if len(Y) != len(a_final):
                    warnings.warn(f"GT length mismatch for {pkl_path.name}: "
                                f"Y={len(Y)}, a_final={len(a_final)}. Skipping.")
                    return None, []
            else:
                # No external GT found - skip this file
                warnings.warn(f"No external GT found for dataset_index={dataset_index} ({pkl_path.name}). Skipping.")
                return None, []
        else:
            # Use Y from this file (no expansion, might already be expanded)
            Y = data['Y']
            if isinstance(Y, torch.Tensor):
                Y = Y.cpu().numpy()
            Y = Y.astype(bool)
            # Extract events from Y as-is
            original_truth_events = None  # Will be extracted later

        final_threshold_original = data['final_threshold']

        # Get subject and run IDs (if available)
        # Try pkl first, fallback to filename parsing
        subject_id = data.get('subject_id', None)
        if not subject_id or subject_id == 'unknown':
            subject_id = extract_subject_from_filename(pkl_path.name)
        run_id = data.get('run_id', 'unknown')

        # Calculate scaled threshold
        scaled_threshold = final_threshold_original * scale

        # Baseline prediction
        predicted = a_final > scaled_threshold

        # Evaluate baseline (pass original_truth_events if available)
        metrics_baseline = evaluate_event_detection(
            predicted, Y, fs=8.0,
            truth_events=original_truth_events
        )

        # Apply clustering if available
        metrics_clustered = None
        n_clusters = 0
        n_representatives = 0

        if CLUSTERING_AVAILABLE:
            try:
                # Use original seizure events if available (from expanded external GT)
                # Otherwise extract from Y (which might already be expanded in the pkl)
                if original_truth_events is not None:
                    truth_events = original_truth_events
                else:
                    truth_events = get_events(Y)

                gt_intervals = [(s/8.0, (e+1)/8.0) for s, e in truth_events]

                results_cluster = analyze_detection_mask(
                    mask=predicted,
                    fs=8.0,
                    file_id=str(dataset_index),
                    scores=a_final,
                    subject_id=subject_id,
                    gt_intervals=gt_intervals,
                    output_folder=None,
                    fixed_strategy=clustering_strategy,
                    true_anomaly_samples=int(np.sum(Y)),
                    true_anomaly_events=len(truth_events)
                )

                representatives = results_cluster['best_results']['representatives']
                n_clusters = results_cluster['best_results']['clusters']
                n_representatives = len(representatives)

                clustered_mask = get_clustered_mask_from_representatives(representatives, len(Y), 8.0)
                metrics_clustered = evaluate_event_detection(
                    clustered_mask, Y, fs=8.0,
                    truth_events=original_truth_events
                )

            except Exception as e:
                warnings.warn(f"Clustering failed for {pkl_path.name}: {e}")

        # Create plots if requested
        
        if save_plots:
            try:
                file_output_dir = output_dir / "plots" / dataset_index
                file_output_dir.mkdir(parents=True, exist_ok=True)
                create_plots(a_final, Y, predicted,
                           clustered_mask if metrics_clustered else None,
                           scaled_threshold, metrics_baseline, metrics_clustered,
                           file_output_dir, dataset_index)
            except Exception as e:
                warnings.warn(f"Plot generation failed for {pkl_path.name}: {e}")

        # Compile results
        result = {
            'file': pkl_path.name,
            'dataset_index': dataset_index,
            'subject_id': subject_id,
            'run_id': run_id,
            'scale': scale,
            'clustering_strategy': clustering_strategy,
            'duration_seconds': len(Y) / 8.0,
            'duration_hours': len(Y) / 8.0 / 3600.0,
            'total_samples': len(Y),
            'true_anomaly_samples': int(np.sum(Y)),
            'true_anomaly_percentage': float(np.sum(Y) / len(Y) * 100),
            'n_truth_events': metrics_baseline['n_truth_events'],
            'original_threshold': float(final_threshold_original),
            'scaled_threshold': float(scaled_threshold),
            'max_score': float(np.max(a_final)),
            'min_score': float(np.min(a_final)),
            'mean_score': float(np.mean(a_final)),
            # Baseline metrics
            'baseline_pred_samples': int(np.sum(predicted)),
            'baseline_pred_percentage': float(np.sum(predicted) / len(predicted) * 100),
            'baseline_n_pred_events': metrics_baseline['n_pred_events'],
            'baseline_TP': metrics_baseline['TP_events'],
            'baseline_FN': metrics_baseline['FN_events'],
            'baseline_FP': metrics_baseline['FP_events'],
            'baseline_sensitivity': float(metrics_baseline['sensitivity']),
            'baseline_iou': float(metrics_baseline['event_iou']),
            'baseline_far_per_hour': float(metrics_baseline['false_alarm_rate_per_hour']),
        }

        # Add clustered metrics if available
        if metrics_clustered:
            clustered_mask_np = clustered_mask if isinstance(clustered_mask, np.ndarray) else np.array(clustered_mask)
            result.update({
                'clustered_n_clusters': n_clusters,
                'clustered_n_representatives': n_representatives,
                'clustered_pred_samples': int(np.sum(clustered_mask_np)),
                'clustered_pred_percentage': float(np.sum(clustered_mask_np) / len(clustered_mask_np) * 100),
                'clustered_n_pred_events': metrics_clustered['n_pred_events'],
                'clustered_TP': metrics_clustered['TP_events'],
                'clustered_FN': metrics_clustered['FN_events'],
                'clustered_FP': metrics_clustered['FP_events'],
                'clustered_sensitivity': float(metrics_clustered['sensitivity']),
                'clustered_iou': float(metrics_clustered['event_iou']),
                'clustered_far_per_hour': float(metrics_clustered['false_alarm_rate_per_hour']),
                'far_reduction_percentage': float((metrics_baseline['false_alarm_rate_per_hour'] -
                                                  metrics_clustered['false_alarm_rate_per_hour']) /
                                                 metrics_baseline['false_alarm_rate_per_hour'] * 100) if metrics_baseline['false_alarm_rate_per_hour'] > 0 else 0.0
            })

        # Generate per-seizure detection results
        seizure_detections = []

        # Use original seizure events if available (from expanded external GT)
        # Otherwise extract from Y
        if original_truth_events is not None:
            truth_events_for_detection = original_truth_events
        else:
            truth_events_for_detection = get_events(Y)

        if len(truth_events_for_detection) > 0 and metrics_clustered:
            # Only generate seizure detections if there are GT seizures and clustering succeeded
            seizure_detections = check_seizure_detections(
                truth_events_for_detection,
                clustered_mask,
                subject_id
            )

        return result, seizure_detections

    except Exception as e:
        warnings.warn(f"Failed to process {pkl_path.name}: {e}")
        return None, []


def create_plots(a_final, Y, predicted, clustered_mask, threshold,
                metrics_baseline, metrics_clustered, output_dir, dataset_index):
    """Create plots for a single file."""

    # Downsample for plotting
    step = max(1, len(a_final) // 5000)
    time_hours = np.arange(len(a_final))[::step] / 8.0 / 3600.0
    a_final_plot = a_final[::step]
    Y_plot = Y[::step]
    predicted_plot = predicted[::step]

    # Plot 1: Overview
    n_plots = 4 if clustered_mask is not None else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 2.5*n_plots))

    if n_plots == 1:
        axes = [axes]

    # Scores
    axes[0].plot(time_hours, a_final_plot, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].axhline(y=threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold={threshold:.1f}')
    axes[0].fill_between(time_hours, 0, Y_plot * np.max(a_final_plot), alpha=0.3, color='orange')
    axes[0].set_ylabel('Anomaly Score')
    axes[0].set_title(f'Dataset {dataset_index}: Anomaly Scores', fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Ground truth
    axes[1].fill_between(time_hours, 0, Y_plot, alpha=0.6, color='orange')
    axes[1].set_ylabel('Ground Truth')
    axes[1].set_title('Ground Truth Anomalies', fontweight='bold')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)

    # Baseline
    axes[2].fill_between(time_hours, 0, predicted_plot, alpha=0.6, color='blue')
    axes[2].fill_between(time_hours, 0, Y_plot, alpha=0.3, color='orange')
    axes[2].set_ylabel('Predictions')
    title = f'Baseline (Sens={metrics_baseline["sensitivity"]:.2f}, FAR={metrics_baseline["false_alarm_rate_per_hour"]:.1f}/h)'
    axes[2].set_title(title, fontweight='bold')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)

    # Clustered
    if clustered_mask is not None:
        clustered_plot = clustered_mask[::step]
        axes[3].fill_between(time_hours, 0, clustered_plot, alpha=0.6, color='green')
        axes[3].fill_between(time_hours, 0, Y_plot, alpha=0.3, color='orange')
        axes[3].set_xlabel('Time (hours)')
        axes[3].set_ylabel('Predictions')
        title = f'Clustered (Sens={metrics_clustered["sensitivity"]:.2f}, FAR={metrics_clustered["false_alarm_rate_per_hour"]:.1f}/h)'
        axes[3].set_title(title, fontweight='bold')
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].grid(True, alpha=0.3)
    else:
        axes[2].set_xlabel('Time (hours)')

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_index}_timeline.png', dpi=120, bbox_inches='tight')
    plt.close()


def generate_summary_report(results_df: pd.DataFrame, output_path: Path):
    """Generate a text summary report."""

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TIMEVQVAE-AD BATCH EVALUATION SUMMARY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files processed: {len(results_df)}\n\n")

        # Dataset info
        f.write("-"*80 + "\n")
        f.write("DATASET OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total recording duration: {results_df['duration_hours'].sum():.2f} hours\n")
        f.write(f"Mean recording duration: {results_df['duration_hours'].mean():.2f} hours\n")
        f.write(f"Total samples: {results_df['total_samples'].sum():,}\n")
        f.write(f"Total true anomaly samples: {results_df['true_anomaly_samples'].sum():,}\n")
        f.write(f"Mean true anomaly percentage: {results_df['true_anomaly_percentage'].mean():.3f}%\n")
        f.write(f"Total true events: {results_df['n_truth_events'].sum()}\n")
        f.write(f"Files with anomalies: {(results_df['n_truth_events'] > 0).sum()}/{len(results_df)}\n\n")

        # Threshold info
        f.write("-"*80 + "\n")
        f.write("THRESHOLD INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Scaling factor used: {results_df['scale'].iloc[0]}\n")
        f.write(f"Mean original threshold: {results_df['original_threshold'].mean():.2f}\n")
        f.write(f"Mean scaled threshold: {results_df['scaled_threshold'].mean():.2f}\n")
        f.write(f"Mean max score: {results_df['max_score'].mean():.2f}\n")
        above_threshold = (results_df['max_score'] > results_df['scaled_threshold']).sum()
        f.write(f"Files with max_score > threshold: {above_threshold}/{len(results_df)}\n\n")

        # Baseline performance
        f.write("-"*80 + "\n")
        f.write("BASELINE PERFORMANCE (No Clustering)\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Sensitivity: {results_df['baseline_sensitivity'].mean():.3f} (±{results_df['baseline_sensitivity'].std():.3f})\n")
        f.write(f"Median Sensitivity: {results_df['baseline_sensitivity'].median():.3f}\n")
        f.write(f"Mean FAR: {results_df['baseline_far_per_hour'].mean():.2f} events/hour (±{results_df['baseline_far_per_hour'].std():.2f})\n")
        f.write(f"Median FAR: {results_df['baseline_far_per_hour'].median():.2f} events/hour\n")
        f.write(f"Mean IoU: {results_df['baseline_iou'].mean():.3f}\n")
        f.write(f"Total TP: {results_df['baseline_TP'].sum()}\n")
        f.write(f"Total FN: {results_df['baseline_FN'].sum()}\n")
        f.write(f"Total FP: {results_df['baseline_FP'].sum()}\n")
        f.write(f"Mean predicted events per file: {results_df['baseline_n_pred_events'].mean():.1f}\n\n")

        # Clustered performance (if available)
        if 'clustered_sensitivity' in results_df.columns:
            f.write("-"*80 + "\n")
            f.write(f"CLUSTERED PERFORMANCE ({results_df['clustering_strategy'].iloc[0]})\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean Sensitivity: {results_df['clustered_sensitivity'].mean():.3f} (±{results_df['clustered_sensitivity'].std():.3f})\n")
            f.write(f"Median Sensitivity: {results_df['clustered_sensitivity'].median():.3f}\n")
            f.write(f"Mean FAR: {results_df['clustered_far_per_hour'].mean():.2f} events/hour (±{results_df['clustered_far_per_hour'].std():.2f})\n")
            f.write(f"Median FAR: {results_df['clustered_far_per_hour'].median():.2f} events/hour\n")
            f.write(f"Mean IoU: {results_df['clustered_iou'].mean():.3f}\n")
            f.write(f"Total TP: {results_df['clustered_TP'].sum()}\n")
            f.write(f"Total FN: {results_df['clustered_FN'].sum()}\n")
            f.write(f"Total FP: {results_df['clustered_FP'].sum()}\n")
            f.write(f"Mean clusters per file: {results_df['clustered_n_clusters'].mean():.1f}\n")
            f.write(f"Mean FAR reduction: {results_df['far_reduction_percentage'].mean():.1f}%\n\n")

            # Comparison
            f.write("-"*80 + "\n")
            f.write("CLUSTERING IMPACT\n")
            f.write("-"*80 + "\n")
            sens_change = results_df['clustered_sensitivity'].mean() - results_df['baseline_sensitivity'].mean()
            far_change = results_df['clustered_far_per_hour'].mean() - results_df['baseline_far_per_hour'].mean()
            far_change_pct = (far_change / results_df['baseline_far_per_hour'].mean()) * 100

            f.write(f"Sensitivity change: {sens_change:+.3f} ({sens_change/results_df['baseline_sensitivity'].mean()*100:+.1f}%)\n")
            f.write(f"FAR change: {far_change:+.2f} events/hour ({far_change_pct:+.1f}%)\n")
            f.write(f"Mean FP reduction: {(results_df['baseline_FP'] - results_df['clustered_FP']).mean():.1f} events\n\n")

        # Top/Bottom performers
        f.write("-"*80 + "\n")
        f.write("TOP 5 FILES BY SENSITIVITY (Baseline)\n")
        f.write("-"*80 + "\n")
        top5 = results_df.nlargest(5, 'baseline_sensitivity')[['dataset_index', 'baseline_sensitivity', 'baseline_far_per_hour']]
        for idx, row in top5.iterrows():
            f.write(f"  {row['dataset_index']}: Sens={row['baseline_sensitivity']:.3f}, FAR={row['baseline_far_per_hour']:.2f}/h\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("BOTTOM 5 FILES BY FAR (Baseline - Best)\n")
        f.write("-"*80 + "\n")
        bottom5 = results_df.nsmallest(5, 'baseline_far_per_hour')[['dataset_index', 'baseline_sensitivity', 'baseline_far_per_hour']]
        for idx, row in bottom5.iterrows():
            f.write(f"  {row['dataset_index']}: Sens={row['baseline_sensitivity']:.3f}, FAR={row['baseline_far_per_hour']:.2f}/h\n")

        f.write("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate TimeVQVAE-AD results with optional external ground truth',
        epilog='Use --gt_dir to load ground truth labels (Y) from a different directory. '
               'Files are matched by dataset_index (e.g., "117").'
    )
    parser.add_argument('input_dir', type=str, help='Directory containing pkl files with anomaly scores')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: input_dir/batch_results)')
    parser.add_argument('--gt_dir', type=str, default=None, help='Optional directory to load ground truth (Y) from. '
                                                                  'Files matched by dataset_index.')
    parser.add_argument('--scale', type=float, default=0.7, help='Threshold scaling factor (default: 0.7)')
    parser.add_argument('--clustering', type=str, default='time_90s', help='Clustering strategy (default: time_90s)')
    parser.add_argument('--pattern', type=str, default='*joint_anomaly_score.pkl', help='File pattern (default: *joint_anomaly_score.pkl)')
    parser.add_argument('--no_plots', action='store_true', help='Skip individual plot generation')
    parser.add_argument('--config', type=str, required=True, help='Configuration name (e.g., "baseline", "optimized")')
    parser.add_argument('--window_type', type=str, required=True, help='Window type (e.g., "window", "no_window")')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')

    args = parser.parse_args()

    # Setup GT directory if provided
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if gt_dir and not gt_dir.exists():
        print(f"ERROR: Ground truth directory not found: {gt_dir}")
        return

    # Setup paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "batch_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all matching files
    pkl_files = sorted(input_dir.rglob(args.pattern))

    if not pkl_files:
        print(f"ERROR: No files matching pattern '{args.pattern}' found in {input_dir}")
        return

    print("="*80)
    print("TIMEVQVAE-AD BATCH EVALUATION (WITH EXTERNAL GT + PARALLEL)")
    print("="*80)
    print(f"Input directory (scores): {input_dir}")
    print(f"Ground truth directory: {gt_dir if gt_dir else 'SAME (use scores Y)'}")
    print(f"Output directory: {output_dir}")
    print(f"Files found: {len(pkl_files)}")
    print(f"Config: {args.config}")
    print(f"Window type: {args.window_type}")
    print(f"Scale: {args.scale}")
    print(f"Clustering: {args.clustering}")
    print(f"Clustering available: {CLUSTERING_AVAILABLE}")
    print(f"Generate plots: {not args.no_plots}")
    print(f"Workers: {args.workers}")
    print()

    # Prepare arguments for parallel processing
    worker_args = [
        (pkl_file, args.scale, args.clustering, output_dir, not args.no_plots, gt_dir)
        for pkl_file in pkl_files
    ]

    # Process files in parallel
    results = []
    all_seizure_detections = []
    failed_files = []

    with Pool(processes=args.workers) as pool:
        # Use imap for progress tracking with tqdm
        for idx, (result, seizure_detections) in enumerate(tqdm(
            pool.imap(process_file_worker, worker_args),
            total=len(pkl_files),
            desc="Processing files"
        )):
            if result is not None:
                results.append(result)
                # Add config and window_type to each seizure detection
                for sd in seizure_detections:
                    sd['config'] = args.config
                    sd['window_type'] = args.window_type
                all_seizure_detections.extend(seizure_detections)
            else:
                # Track failed file using the index
                failed_files.append(worker_args[idx][0].name)

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'batch_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")

    # Save seizure-level detections CSV
    if all_seizure_detections:
        seizure_df = pd.DataFrame(all_seizure_detections)
        # Reorder columns: subject, seizure_idx, detected, config, window_type
        seizure_df = seizure_df[['subject', 'seizure_idx', 'detected', 'config', 'window_type']]
        seizure_csv_path = output_dir / f'seizure_detections_{timestamp}.csv'
        seizure_df.to_csv(seizure_csv_path, index=False)
        print(f"✓ Saved seizure detections CSV: {seizure_csv_path}")
        print(f"  Total seizures: {len(seizure_df)}")
        print(f"  Detected: {seizure_df['detected'].sum()} ({seizure_df['detected'].sum()/len(seizure_df)*100:.1f}%)")
    else:
        print("⚠ No seizure detections to save (no files with ground truth seizures)")


    # Generate summary report
    report_path = output_dir / f'summary_report_{timestamp}.txt'
    generate_summary_report(results_df, report_path)
    print(f"✓ Saved report: {report_path}")

    # Summary
    print("\n" + "="*80)
    print("BATCH EVALUATION COMPLETE")
    print("="*80)
    print(f"Successfully processed: {len(results)}/{len(pkl_files)} files")
    if failed_files:
        print(f"\n⚠ Failed files ({len(failed_files)}):")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files)-10} more")

    print(f"\nMean Baseline Sensitivity: {results_df['baseline_sensitivity'].mean():.3f}")
    print(f"Mean Baseline FAR: {results_df['baseline_far_per_hour'].mean():.2f} events/hour")

    if 'clustered_sensitivity' in results_df.columns:
        print(f"\nMean Clustered Sensitivity: {results_df['clustered_sensitivity'].mean():.3f}")
        print(f"Mean Clustered FAR: {results_df['clustered_far_per_hour'].mean():.2f} events/hour")
        print(f"Mean FAR Reduction: {results_df['far_reduction_percentage'].mean():.1f}%")

    print(f"\nResults saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
