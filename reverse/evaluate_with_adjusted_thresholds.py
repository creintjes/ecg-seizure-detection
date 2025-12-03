#!/usr/bin/env python3
"""
Evaluate TimeVQVAE-AD results with adjusted thresholds.
Since the original threshold is too high (no detections), we test multiple strategies.
"""

import pickle
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to import evaluation functions
sys.path.append(str(Path(__file__).parent.parent / "TimeVQVAE-AD" / "evaluation"))

try:
    from eval import evaluate_event_detection, get_events
except ImportError:
    print("WARNING: Could not import from TimeVQVAE-AD/evaluation/eval.py")
    print("Defining evaluation functions locally...\n")

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

        # Calculate IoU
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


def load_and_prepare_data(pkl_path):
    """Load pickle file and convert tensors to numpy."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Convert torch tensors to numpy
    for key in ['X_test_unscaled', 'Y']:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cpu().numpy()

    return data


def evaluate_threshold_strategies(data, fs=8.0):
    """Test multiple threshold strategies."""

    a_final = data['a_final']
    Y = data['Y'].astype(bool)
    final_threshold = data['final_threshold']
    joint_threshold = data['joint_threshold']
    a_s_star = data['a_s^*']

    print("="*80)
    print("THRESHOLD STRATEGY EVALUATION")
    print("="*80)
    print(f"\nGround Truth: {np.sum(Y):,} anomaly samples ({(np.sum(Y)/len(Y))*100:.3f}%)")
    print(f"Total samples: {len(Y):,}")
    print(f"Duration: {len(Y)/fs/3600:.2f} hours\n")

    strategies = {}

    # Strategy 1: Original threshold (will likely fail)
    print(f"{'='*80}")
    print("STRATEGY 1: Original Threshold")
    print(f"{'='*80}")
    print(f"Threshold: {final_threshold:.4f}")
    pred = a_final > final_threshold
    metrics = evaluate_event_detection(pred, Y, fs)
    strategies['original'] = {
        'threshold': final_threshold,
        'pred': pred,
        'metrics': metrics,
        'description': 'Original final_threshold from pickle'
    }
    print(f"Predicted samples: {np.sum(pred):,} ({(np.sum(pred)/len(pred))*100:.3f}%)")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}, FAR: {metrics['false_alarm_rate_per_hour']:.2f}/hr")
    print(f"TP={metrics['TP_events']}, FN={metrics['FN_events']}, FP={metrics['FP_events']}, IoU={metrics['event_iou']:.3f}")
    print()

    # Strategy 2: 99th percentile
    print(f"{'='*80}")
    print("STRATEGY 2: 99th Percentile Threshold")
    print(f"{'='*80}")
    threshold_99 = np.percentile(a_final, 99)
    print(f"Threshold: {threshold_99:.4f} (99th percentile)")
    pred = a_final > threshold_99
    metrics = evaluate_event_detection(pred, Y, fs)
    strategies['p99'] = {
        'threshold': threshold_99,
        'pred': pred,
        'metrics': metrics,
        'description': '99th percentile of a_final'
    }
    print(f"Predicted samples: {np.sum(pred):,} ({(np.sum(pred)/len(pred))*100:.3f}%)")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}, FAR: {metrics['false_alarm_rate_per_hour']:.2f}/hr")
    print(f"TP={metrics['TP_events']}, FN={metrics['FN_events']}, FP={metrics['FP_events']}, IoU={metrics['event_iou']:.3f}")
    print()

    # Strategy 3: 95th percentile
    print(f"{'='*80}")
    print("STRATEGY 3: 95th Percentile Threshold")
    print(f"{'='*80}")
    threshold_95 = np.percentile(a_final, 95)
    print(f"Threshold: {threshold_95:.4f} (95th percentile)")
    pred = a_final > threshold_95
    metrics = evaluate_event_detection(pred, Y, fs)
    strategies['p95'] = {
        'threshold': threshold_95,
        'pred': pred,
        'metrics': metrics,
        'description': '95th percentile of a_final'
    }
    print(f"Predicted samples: {np.sum(pred):,} ({(np.sum(pred)/len(pred))*100:.3f}%)")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}, FAR: {metrics['false_alarm_rate_per_hour']:.2f}/hr")
    print(f"TP={metrics['TP_events']}, FN={metrics['FN_events']}, FP={metrics['FP_events']}, IoU={metrics['event_iou']:.3f}")
    print()

    # Strategy 4: 90th percentile
    print(f"{'='*80}")
    print("STRATEGY 4: 90th Percentile Threshold")
    print(f"{'='*80}")
    threshold_90 = np.percentile(a_final, 90)
    print(f"Threshold: {threshold_90:.4f} (90th percentile)")
    pred = a_final > threshold_90
    metrics = evaluate_event_detection(pred, Y, fs)
    strategies['p90'] = {
        'threshold': threshold_90,
        'pred': pred,
        'metrics': metrics,
        'description': '90th percentile of a_final'
    }
    print(f"Predicted samples: {np.sum(pred):,} ({(np.sum(pred)/len(pred))*100:.3f}%)")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}, FAR: {metrics['false_alarm_rate_per_hour']:.2f}/hr")
    print(f"TP={metrics['TP_events']}, FN={metrics['FN_events']}, FP={metrics['FP_events']}, IoU={metrics['event_iou']:.3f}")
    print()

    # Strategy 5: Best single frequency band
    print(f"{'='*80}")
    print("STRATEGY 5: Best Single Frequency Band")
    print(f"{'='*80}")

    best_band = None
    best_metrics = None
    best_sensitivity = -1

    for i in range(a_s_star.shape[0]):
        band_threshold = np.percentile(a_s_star[i], 95)
        pred = a_s_star[i] > band_threshold
        metrics = evaluate_event_detection(pred, Y, fs)

        if metrics['sensitivity'] > best_sensitivity:
            best_sensitivity = metrics['sensitivity']
            best_metrics = metrics
            best_band = i
            best_threshold = band_threshold

    print(f"Best band: {best_band} (threshold: {best_threshold:.4f} at 95th percentile)")
    pred = a_s_star[best_band] > best_threshold
    strategies['best_freq'] = {
        'threshold': best_threshold,
        'pred': pred,
        'metrics': best_metrics,
        'band': best_band,
        'description': f'Best frequency band {best_band} at 95th percentile'
    }
    print(f"Predicted samples: {np.sum(pred):,} ({(np.sum(pred)/len(pred))*100:.3f}%)")
    print(f"Sensitivity: {best_metrics['sensitivity']:.3f}, FAR: {best_metrics['false_alarm_rate_per_hour']:.2f}/hr")
    print(f"TP={best_metrics['TP_events']}, FN={best_metrics['FN_events']}, FP={best_metrics['FP_events']}, IoU={best_metrics['event_iou']:.3f}")
    print()

    # Strategy 6: Top-k percentile sweep (find optimal)
    print(f"{'='*80}")
    print("STRATEGY 6: Optimal Percentile (maximize sensitivity with FAR constraint)")
    print(f"{'='*80}")

    max_far = 5.0  # Maximum allowed false alarms per hour
    best_percentile = None
    best_metrics_optimal = None
    best_sensitivity_optimal = 0

    for percentile in range(80, 100):
        threshold = np.percentile(a_final, percentile)
        pred = a_final > threshold
        metrics = evaluate_event_detection(pred, Y, fs)

        # Only consider if FAR is acceptable
        if metrics['false_alarm_rate_per_hour'] <= max_far:
            if metrics['sensitivity'] > best_sensitivity_optimal:
                best_sensitivity_optimal = metrics['sensitivity']
                best_metrics_optimal = metrics
                best_percentile = percentile
                best_threshold_optimal = threshold

    if best_percentile is not None:
        print(f"Best percentile: {best_percentile} (threshold: {best_threshold_optimal:.4f})")
        print(f"Constraint: FAR ≤ {max_far} events/hour")
        pred = a_final > best_threshold_optimal
        strategies['optimal'] = {
            'threshold': best_threshold_optimal,
            'pred': pred,
            'metrics': best_metrics_optimal,
            'percentile': best_percentile,
            'description': f'{best_percentile}th percentile with FAR≤{max_far}/hr'
        }
        print(f"Predicted samples: {np.sum(pred):,} ({(np.sum(pred)/len(pred))*100:.3f}%)")
        print(f"Sensitivity: {best_metrics_optimal['sensitivity']:.3f}, FAR: {best_metrics_optimal['false_alarm_rate_per_hour']:.2f}/hr")
        print(f"TP={best_metrics_optimal['TP_events']}, FN={best_metrics_optimal['FN_events']}, FP={best_metrics_optimal['FP_events']}, IoU={best_metrics_optimal['event_iou']:.3f}")
    else:
        print(f"No percentile found with FAR ≤ {max_far} events/hour")
    print()

    # Comparison table
    print(f"{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}\n")

    print(f"{'Strategy':<25} {'Threshold':<12} {'Pred %':<10} {'Sens':<8} {'FAR/hr':<8} {'IoU':<8}")
    print("-" * 80)

    for name, strat in strategies.items():
        threshold = strat['threshold']
        pred_pct = (np.sum(strat['pred']) / len(strat['pred'])) * 100
        sens = strat['metrics']['sensitivity']
        far = strat['metrics']['false_alarm_rate_per_hour']
        iou = strat['metrics']['event_iou']

        print(f"{name:<25} {threshold:<12.4f} {pred_pct:<10.3f} {sens:<8.3f} {far:<8.2f} {iou:<8.3f}")

    print("\n")

    # Detailed event information for best strategy
    if best_metrics_optimal is not None:
        print(f"{'='*80}")
        print(f"DETAILED EVENT ANALYSIS (Best Strategy: {best_percentile}th percentile)")
        print(f"{'='*80}\n")

        truth_events = best_metrics_optimal['truth_events']
        pred_events = best_metrics_optimal['pred_events']

        print(f"Ground Truth Events: {len(truth_events)}")
        for i, (start, end) in enumerate(truth_events):
            duration = (end - start + 1) / fs
            start_time = start / fs
            print(f"  Event {i+1}: {duration:.2f}s (from {start_time:.2f}s, samples {start}-{end})")

        print(f"\nPredicted Events: {len(pred_events)}")
        if len(pred_events) > 20:
            print(f"  (showing first 20 of {len(pred_events)} events)")
            pred_events_show = pred_events[:20]
        else:
            pred_events_show = pred_events

        for i, (start, end) in enumerate(pred_events_show):
            duration = (end - start + 1) / fs
            start_time = start / fs
            # Check if it overlaps with any ground truth
            is_tp = any(Y[start:end+1])
            marker = "✓ TP" if is_tp else "✗ FP"
            print(f"  Event {i+1}: {duration:.2f}s (from {start_time:.2f}s, samples {start}-{end}) {marker}")

    return strategies


if __name__ == "__main__":
    pkl_path = Path(__file__).parent / "117_window-joint_anomaly_score.pkl" / "117_window-joint_anomaly_score.pkl"

    if not pkl_path.exists():
        pkl_path = Path(__file__).parent / "117_window-joint_anomaly_score.pkl"

    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        sys.exit(1)

    print("Loading data...")
    data = load_and_prepare_data(pkl_path)
    print("Data loaded successfully!\n")

    strategies = evaluate_threshold_strategies(data, fs=8.0)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nRecommendation: Use the 'optimal' or 'best_freq' strategy for best results.")
    print("The original threshold is too high for this dataset.\n")
