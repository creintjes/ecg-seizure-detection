#!/usr/bin/env python3
"""
Evaluate adaptive (time-varying) thresholding strategies.
Compare static vs. adaptive thresholds for anomaly detection.
"""

import pickle
import numpy as np
import torch
from pathlib import Path
import sys
from scipy.ndimage import uniform_filter1d

# Add parent directory to import evaluation functions
sys.path.append(str(Path(__file__).parent.parent / "TimeVQVAE-AD" / "evaluation"))

try:
    from eval import evaluate_event_detection, get_events
except ImportError:
    def get_events(mask):
        """Extract events from boolean mask."""
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

    def evaluate_event_detection(pred, truth, fs=None):
        """Evaluate event-based detection."""
        truth_events = get_events(truth)
        pred_events = get_events(pred)
        TP = sum(1 for (ts, te) in truth_events if pred[ts:te+1].any())
        FN = len(truth_events) - TP
        FP = sum(1 for (ps, pe) in pred_events if not truth[ps:pe+1].any())
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        iou = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else 0.0
        metrics = {
            'truth_events': truth_events,
            'pred_events': pred_events,
            'TP_events': TP, 'FN_events': FN, 'FP_events': FP,
            'sensitivity': sensitivity,
            'event_iou': iou
        }
        if fs is not None:
            total_hours = len(truth) / fs / 3600.0
            metrics['false_alarm_rate_per_hour'] = (FP / total_hours) if total_hours > 0 else float('inf')
        return metrics


def rolling_percentile(arr, window_size, percentile):
    """
    Calculate rolling percentile using a sliding window.
    Note: This is approximate and uses a simpler method for speed.
    """
    result = np.zeros_like(arr)
    half_window = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        result[i] = np.percentile(arr[start:end], percentile)

    return result


def rolling_percentile_fast(arr, window_size, percentile):
    """
    Faster rolling percentile using block processing.
    """
    result = np.zeros_like(arr)
    half_window = window_size // 2

    # Process in blocks for speed
    block_size = 1000
    for block_start in range(0, len(arr), block_size):
        block_end = min(block_start + block_size, len(arr))

        for i in range(block_start, block_end):
            start = max(0, i - half_window)
            end = min(len(arr), i + half_window + 1)
            result[i] = np.percentile(arr[start:end], percentile)

    return result


def evaluate_adaptive_strategies(data, fs=8.0):
    """Test multiple adaptive thresholding strategies."""

    a_final = data['a_final']
    Y = data['Y']
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    Y = Y.astype(bool)

    final_threshold = data['final_threshold']

    print("="*80)
    print("ADAPTIVE THRESHOLDING EVALUATION")
    print("="*80)
    print(f"\nGround Truth: {np.sum(Y):,} anomaly samples ({(np.sum(Y)/len(Y))*100:.3f}%)")
    print(f"Total samples: {len(Y):,}")
    print(f"Duration: {len(Y)/fs/3600:.2f} hours\n")

    strategies = {}

    # Baseline: Static 95th percentile
    print(f"{'='*80}")
    print("BASELINE: Static 95th Percentile")
    print(f"{'='*80}")
    static_threshold = np.percentile(a_final, 95)
    pred = a_final > static_threshold
    metrics = evaluate_event_detection(pred, Y, fs)
    strategies['static_p95'] = {
        'threshold': static_threshold,
        'pred': pred,
        'metrics': metrics,
        'description': 'Static 95th percentile (baseline)'
    }
    print(f"Threshold: {static_threshold:.4f}")
    print(f"Predicted: {np.sum(pred):,} samples ({(np.sum(pred)/len(pred))*100:.3f}%)")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}, FAR: {metrics['false_alarm_rate_per_hour']:.2f}/hr, IoU: {metrics['event_iou']:.3f}")
    print(f"TP={metrics['TP_events']}, FN={metrics['FN_events']}, FP={metrics['FP_events']}")
    print()

    # Strategy 1: Rolling mean + k*std
    print(f"{'='*80}")
    print("STRATEGY 1: Rolling Mean + k×STD")
    print(f"{'='*80}")

    window_sizes = [500, 1000, 5000]  # Different window sizes
    k_values = [2.0, 2.5, 3.0]  # Different k multipliers

    best_metrics_mean_std = None
    best_sensitivity_mean_std = 0
    best_params_mean_std = None

    for window_size in window_sizes:
        for k in k_values:
            # Calculate rolling mean and std
            rolling_mean = uniform_filter1d(a_final, size=window_size, mode='nearest')
            rolling_std = np.sqrt(uniform_filter1d((a_final - rolling_mean)**2, size=window_size, mode='nearest'))

            # Adaptive threshold
            adaptive_threshold = rolling_mean + k * rolling_std

            pred = a_final > adaptive_threshold
            metrics = evaluate_event_detection(pred, Y, fs)

            if metrics['sensitivity'] > best_sensitivity_mean_std:
                best_sensitivity_mean_std = metrics['sensitivity']
                best_metrics_mean_std = metrics
                best_params_mean_std = (window_size, k)
                best_pred_mean_std = pred

    window_size, k = best_params_mean_std
    print(f"Best params: window={window_size}, k={k}")

    rolling_mean = uniform_filter1d(a_final, size=window_size, mode='nearest')
    rolling_std = np.sqrt(uniform_filter1d((a_final - rolling_mean)**2, size=window_size, mode='nearest'))
    adaptive_threshold_mean_std = rolling_mean + k * rolling_std

    strategies['adaptive_mean_std'] = {
        'threshold': adaptive_threshold_mean_std,  # time-varying!
        'pred': best_pred_mean_std,
        'metrics': best_metrics_mean_std,
        'params': best_params_mean_std,
        'description': f'Rolling mean + {k}×std (window={window_size})'
    }
    print(f"Predicted: {np.sum(best_pred_mean_std):,} samples ({(np.sum(best_pred_mean_std)/len(best_pred_mean_std))*100:.3f}%)")
    print(f"Sensitivity: {best_metrics_mean_std['sensitivity']:.3f}, FAR: {best_metrics_mean_std['false_alarm_rate_per_hour']:.2f}/hr, IoU: {best_metrics_mean_std['event_iou']:.3f}")
    print(f"TP={best_metrics_mean_std['TP_events']}, FN={best_metrics_mean_std['FN_events']}, FP={best_metrics_mean_std['FP_events']}")
    print()

    # Strategy 2: Rolling percentile
    print(f"{'='*80}")
    print("STRATEGY 2: Rolling Percentile")
    print(f"{'='*80}")

    window_sizes = [500, 1000, 2000]
    percentiles = [90, 92, 95]

    best_metrics_percentile = None
    best_sensitivity_percentile = 0
    best_params_percentile = None

    print("Testing rolling percentile (this may take a moment)...")

    for window_size in window_sizes:
        for percentile in percentiles:
            print(f"  Testing window={window_size}, percentile={percentile}...", end='\r')

            # Use fast block-based rolling percentile
            adaptive_threshold = rolling_percentile_fast(a_final, window_size, percentile)

            pred = a_final > adaptive_threshold
            metrics = evaluate_event_detection(pred, Y, fs)

            if metrics['sensitivity'] > best_sensitivity_percentile:
                best_sensitivity_percentile = metrics['sensitivity']
                best_metrics_percentile = metrics
                best_params_percentile = (window_size, percentile)
                best_pred_percentile = pred
                best_threshold_percentile = adaptive_threshold

    print(" " * 80)  # Clear the progress line

    window_size, percentile = best_params_percentile
    print(f"Best params: window={window_size}, percentile={percentile}th")

    strategies['adaptive_percentile'] = {
        'threshold': best_threshold_percentile,  # time-varying!
        'pred': best_pred_percentile,
        'metrics': best_metrics_percentile,
        'params': best_params_percentile,
        'description': f'Rolling {percentile}th percentile (window={window_size})'
    }
    print(f"Predicted: {np.sum(best_pred_percentile):,} samples ({(np.sum(best_pred_percentile)/len(best_pred_percentile))*100:.3f}%)")
    print(f"Sensitivity: {best_metrics_percentile['sensitivity']:.3f}, FAR: {best_metrics_percentile['false_alarm_rate_per_hour']:.2f}/hr, IoU: {best_metrics_percentile['event_iou']:.3f}")
    print(f"TP={best_metrics_percentile['TP_events']}, FN={best_metrics_percentile['FN_events']}, FP={best_metrics_percentile['FP_events']}")
    print()

    # Strategy 3: Hybrid (use different thresholds for different segments)
    print(f"{'='*80}")
    print("STRATEGY 3: Segment-based Adaptive Threshold")
    print(f"{'='*80}")

    # Divide recording into segments and use different thresholds
    num_segments = 10
    segment_size = len(a_final) // num_segments

    adaptive_threshold_segments = np.zeros_like(a_final)
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else len(a_final)

        # Use 95th percentile within each segment
        segment_threshold = np.percentile(a_final[start:end], 95)
        adaptive_threshold_segments[start:end] = segment_threshold

    pred = a_final > adaptive_threshold_segments
    metrics = evaluate_event_detection(pred, Y, fs)

    strategies['adaptive_segments'] = {
        'threshold': adaptive_threshold_segments,
        'pred': pred,
        'metrics': metrics,
        'params': num_segments,
        'description': f'Segment-based (n={num_segments} segments)'
    }
    print(f"Number of segments: {num_segments}")
    print(f"Predicted: {np.sum(pred):,} samples ({(np.sum(pred)/len(pred))*100:.3f}%)")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}, FAR: {metrics['false_alarm_rate_per_hour']:.2f}/hr, IoU: {metrics['event_iou']:.3f}")
    print(f"TP={metrics['TP_events']}, FN={metrics['FN_events']}, FP={metrics['FP_events']}")
    print()

    # Comparison table
    print(f"{'='*80}")
    print("COMPARISON TABLE: STATIC vs ADAPTIVE THRESHOLDING")
    print(f"{'='*80}\n")

    print(f"{'Strategy':<30} {'Pred %':<10} {'Sens':<8} {'FAR/hr':<8} {'IoU':<8} {'Params'}")
    print("-" * 100)

    for name, strat in strategies.items():
        pred_pct = (np.sum(strat['pred']) / len(strat['pred'])) * 100
        sens = strat['metrics']['sensitivity']
        far = strat['metrics']['false_alarm_rate_per_hour']
        iou = strat['metrics']['event_iou']

        # Format params
        if name == 'static_p95':
            params_str = f"threshold={strat['threshold']:.2f}"
        elif name == 'adaptive_mean_std':
            w, k = strat['params']
            params_str = f"window={w}, k={k}"
        elif name == 'adaptive_percentile':
            w, p = strat['params']
            params_str = f"window={w}, p={p}th"
        elif name == 'adaptive_segments':
            params_str = f"n_segments={strat['params']}"
        else:
            params_str = ""

        print(f"{name:<30} {pred_pct:<10.3f} {sens:<8.3f} {far:<8.2f} {iou:<8.3f} {params_str}")

    print("\n")

    # Determine best strategy
    best_strategy = max(strategies.items(), key=lambda x: x[1]['metrics']['sensitivity'])
    print(f"{'='*80}")
    print(f"BEST STRATEGY: {best_strategy[0]}")
    print(f"{'='*80}")
    print(f"Description: {best_strategy[1]['description']}")
    print(f"Sensitivity: {best_strategy[1]['metrics']['sensitivity']:.3f}")
    print(f"FAR: {best_strategy[1]['metrics']['false_alarm_rate_per_hour']:.2f} events/hour")
    print(f"IoU: {best_strategy[1]['metrics']['event_iou']:.3f}")
    print()

    return strategies


if __name__ == "__main__":
    pkl_path = Path(__file__).parent / "117_window-joint_anomaly_score.pkl" / "117_window-joint_anomaly_score.pkl"

    if not pkl_path.exists():
        pkl_path = Path(__file__).parent / "117_window-joint_anomaly_score.pkl"

    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        sys.exit(1)

    print("Loading data...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded successfully!\n")

    strategies = evaluate_adaptive_strategies(data, fs=8.0)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("- Static thresholds may not capture time-varying patterns")
    print("- Adaptive thresholds adjust to local score distributions")
    print("- Trade-off: Sensitivity vs. False Alarm Rate")
    print()
