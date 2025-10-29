import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Dict, Optional
from clustering import analyze_detection_mask
from metrics import event_metrics_for_clusters


def get_events(mask: np.ndarray) -> List[Tuple[int, int]]:
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


def evaluate_event_detection(pred: np.ndarray, truth: np.ndarray, fs: Optional[float] = None) -> Dict[str, object]:
    truth_events = get_events(truth)
    pred_events = get_events(pred)
    TP = sum(1 for (ts, te) in truth_events if pred[ts:te+1].any())
    FN = len(truth_events) - TP
    FP = sum(1 for (ps, pe) in pred_events if not truth[ps:pe+1].any())
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    metrics: Dict[str, object] = {
        'truth_events': truth_events,
        'pred_events': pred_events,
        'TP_events': TP, 'FN_events': FN, 'FP_events': FP,
        'sensitivity': sensitivity,
    }
    if fs is not None:
        total_hours = len(truth) / fs / 3600.0
        metrics['false_alarm_rate_per_hour'] = (FP / total_hours) if total_hours > 0 else float('inf')
    return metrics


def load_pkl(fname: str):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def intervals_from_mask(mask: np.ndarray, fs: float) -> List[Tuple[float, float]]:
    return [(s / fs, (e + 1) / fs) for (s, e) in get_events(mask)]


def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


# ---------- Evaluation with Best Freq strategy only ----------
def evaluate_joint_anomaly_score(dataset_index: int = 1, sampling_rate: float = 8.0,
                                 results_path: Optional[str] = None,
                                 best_freq_threshold_scale: float = 1.0,
                                 verbose: bool = True):
    """
    Evaluate joint anomaly score using only the Best Freq strategy.
    
    Args:
        dataset_index: Dataset identifier
        sampling_rate: Sampling rate in Hz
        results_path: Path to joint anomaly score pickle file
        best_freq_threshold_scale: Scale factor for best frequency threshold
        verbose: Whether to print detailed output
        
    Returns:
      results (dict), strategy_data (dict), truth (bool array)
    """
    joint_file = (results_path)
    if not os.path.exists(joint_file):
        raise FileNotFoundError(f"Joint anomaly score file not found: {joint_file}")

    if verbose:
        print(f"Loading joint anomaly score from: {joint_file}")
    R = load_pkl(joint_file)
    scores = np.array(R['a_final']).astype(float)
    final_threshold = R['final_threshold']
    Y = np.array(R['Y']).astype(bool)

    threshold = float(final_threshold) * best_freq_threshold_scale
    

    mask = (scores > threshold).astype(bool)
    
    if verbose:
        print(f"\nDataset {dataset_index} Evaluation Results (Best Freq Strategy):")
        print("=" * 60)
        print(f"Total samples: {len(Y)} | True anomaly samples: {int(Y.sum())} "
              f"({(Y.sum()/len(Y))*100:.2f}%)")
    
    # Evaluate event detection
    metrics = evaluate_event_detection(mask, Y, fs=sampling_rate)
    
    pred_events = metrics['pred_events']
    truth_events = metrics['truth_events']
    
    if verbose:
        print(f"\n" + "="*50)
        print(f"REAL ANOMALY EVENTS (Ground Truth)")
        print(f"="*50)
        print(f"Total real anomaly events: {len(truth_events)}")
        
        if truth_events:
            total_anomaly_duration = 0
            for i, (start, end) in enumerate(truth_events):
                duration = (end - start + 1) / sampling_rate
                start_time = start / sampling_rate
                end_time = (end + 1) / sampling_rate
                total_anomaly_duration += duration
                print(f"  Real Event {i+1}: {duration:.2f}s (from {start_time:.2f}s to {end_time:.2f}s, samples {start}-{end})")
            
            print(f"\nTotal anomaly duration: {total_anomaly_duration:.2f} seconds ({total_anomaly_duration/60:.2f} minutes)")
            total_recording_duration = len(Y) / sampling_rate
            anomaly_percentage = (total_anomaly_duration / total_recording_duration) * 100
            print(f"Total recording duration: {total_recording_duration:.2f} seconds ({total_recording_duration/3600:.2f} hours)")
            print(f"Anomaly percentage of recording: {anomaly_percentage:.3f}%")
        else:
            print("  No real anomaly events found in ground truth.")
        
        print(f"\nPredicted Events Summary: {len(pred_events)} events detected")
        
        print(f"\nDetection Performance:")
        print(f"  Predicted anomaly samples: {int(mask.sum())} ({(mask.sum()/len(mask))*100:.2f}% of recording)")
        print(f"  TP events: {metrics['TP_events']} (real events correctly detected)")
        print(f"  FN events: {metrics['FN_events']} (real events missed)")  
        print(f"  FP events: {metrics['FP_events']} (false alarm events)")
        print(f"  Event Sensitivity: {metrics['sensitivity']:.3f} (fraction of real events detected)")
        print(f"  False Alarm Rate: {metrics['false_alarm_rate_per_hour']:.2f} events/hour")

    strategy_data = {'best_freq': {'mask': mask, 'scores': scores}}
    
    return metrics, strategy_data, Y


# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Joint Anomaly Score using Best Freq Strategy')
    parser.add_argument('--dataset_index', type=int, default=1, 
                        help='Dataset identifier (default: 1)')
    parser.add_argument('--windowed', action='store_true', 
                        help='Use windowed model results (default: no_window)')
    parser.add_argument('--train', action='store_true', 
                        help='Use train set results (default: test set)')
    parser.add_argument('--best_freq_threshold_scale', type=float, default=1.0,
                        help='Scale factor for best frequency threshold (default: 1.0)')
    parser.add_argument('--sampling_rate', type=float, default=8.0,
                        help='Sampling rate in Hz (default: 8.0)')
    
    args = parser.parse_args()
    
    # Construct the file path based on arguments
    window_suffix = "window" if args.windowed else "no-window"
    dataset_suffix = "train" if args.train else "test"
    window_file_suffix = "window" if args.windowed else "no_window"  # Use underscore for filename
    results_file = f"{args.dataset_index:03d}_{window_file_suffix}-joint_anomaly_score.pkl"

    results_path = os.path.join("results", "final", window_suffix, dataset_suffix, results_file)
    
    print(f"Evaluating Joint Anomaly Score for Dataset {args.dataset_index:03d} ({'Windowed' if args.windowed else 'No Window'}) - Best Freq Strategy Only")
    print("=" * 70)
    print(f"Using results file: {results_path}")
    
    results, strategy_data, truth = evaluate_joint_anomaly_score(
        dataset_index=args.dataset_index,
        sampling_rate=args.sampling_rate,
        results_path=results_path,
        best_freq_threshold_scale=args.best_freq_threshold_scale
    )

    # Use the best_freq strategy results
    mask = strategy_data['best_freq']['mask']
    scores = strategy_data['best_freq']['scores']

    if np.any(mask):
        print(f"\nRunning clustering using 'best_freq' ({mask.sum()} positive samples).")
        gt_intervals = intervals_from_mask(truth, args.sampling_rate)

        # Calculate ground truth statistics for clustering
        true_anomaly_samples = int(truth.sum())
        truth_events = get_events(truth)
        true_anomaly_events = len(truth_events)

        results_cluster = analyze_detection_mask(
            mask=mask,
            fs=args.sampling_rate,
            file_id=f"rec_{args.dataset_index:03d}",
            scores=scores,
            subject_id=f"subj_{args.dataset_index:03d}",
            gt_intervals=gt_intervals,
            output_folder=f"rec_{args.dataset_index:03d}_smart_clustered",
            true_anomaly_samples=true_anomaly_samples,  # NEW: Pass ground truth sample count
            true_anomaly_events=true_anomaly_events     # NEW: Pass ground truth event count
        )

        # Show clustering results
        best = results_cluster["best_results"]
        file_level = best["metrics"]
        print("\n=== SMART CLUSTERING (file-level) ===")
        print("Best strategy:", results_cluster["best_strategy"])
        print("Clusters:", best["clusters"], "Representatives:", len(best["representatives"]))
        
        # Calculate anomaly percentages before and after clustering
        total_recording_duration = len(truth) / args.sampling_rate
        
        # Before clustering: predicted anomaly samples
        pred_anomaly_samples_before = mask.sum()
        pred_anomaly_percentage_before = (pred_anomaly_samples_before / len(mask)) * 100
        
        # After clustering: calculate total duration covered by cluster representatives
        clustered_anomaly_duration = 0.0
        for rep in best["representatives"]:
            cluster_start = rep.get('cluster_start_seconds', rep['location_time_seconds'])
            cluster_end = rep.get('cluster_end_seconds', rep['location_time_seconds'])
            clustered_anomaly_duration += (cluster_end - cluster_start)
        
        clustered_anomaly_samples = clustered_anomaly_duration * args.sampling_rate
        clustered_anomaly_percentage = (clustered_anomaly_duration / total_recording_duration) * 100
        
        print(f"\n--- Anomaly Coverage Comparison ---")
        print(f"BEFORE clustering: {pred_anomaly_samples_before:,} samples ({pred_anomaly_percentage_before:.3f}% of recording)")
        print(f"AFTER clustering:  {clustered_anomaly_samples:.0f} samples ({clustered_anomaly_percentage:.3f}% of recording)")
        print(f"Coverage reduction: {((pred_anomaly_percentage_before - clustered_anomaly_percentage) / pred_anomaly_percentage_before * 100):.1f}%")

        ev = event_metrics_for_clusters(best["representatives"], gt_intervals, len(truth), args.sampling_rate)
        print("\n--- Event-level metrics on clustered output ---")
        print(f"Sensitivity: {ev['event_sensitivity']:.3f}  FAR/hr: {ev['event_far_per_hour']:.2f}  "
              f"(TP={ev['TP_events']}, FN={ev['FN_events']}, FP={ev['FP_events']}, N_pred={ev['n_pred_events']})")

    else:
        print("No positive predictions; skipping clustering.")