import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from clustering import analyze_detection_mask


# ---------- Helpers ----------
def smooth_labels(arr: np.ndarray, window_size_samples: int, threshold_count: int) -> np.ndarray:
    x = arr.astype(int)
    kernel = np.ones(window_size_samples, dtype=int)
    counts = np.convolve(x, kernel, mode='same')
    return (counts > threshold_count).astype(int)


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
    event_iou = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else 0.0
    metrics: Dict[str, object] = {
        'truth_events': truth_events,
        'pred_events': pred_events,
        'TP_events': TP, 'FN_events': FN, 'FP_events': FP,
        'sensitivity': sensitivity, 'event_iou': event_iou, 'recall': sensitivity,
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


def event_metrics_for_clusters(representatives: List[Dict], gt_intervals: List[Tuple[float, float]],
                               total_samples: int, fs: float) -> Dict[str, float]:
    # Prefer full cluster span if present; fall back to representative segment span
    pred_events = []
    for r in representatives:
        start = float(r.get('cluster_start_seconds', r.get('segment_start_seconds', r['location_time_seconds'])))
        end = float(r.get('cluster_end_seconds', r.get('segment_end_seconds', r['location_time_seconds'])))
        if end < start:
            start, end = end, start
        pred_events.append((start, end))
    TP = sum(1 for gt in gt_intervals if any(intervals_overlap(gt, pe) for pe in pred_events))
    FN = len(gt_intervals) - TP
    FP = sum(1 for pe in pred_events if not any(intervals_overlap(pe, gt) for gt in gt_intervals))
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    iou = TP / (TP + FN + FP) if (TP + FN + FP) else 0.0
    far_hr = FP / (total_samples / fs / 3600.0) if total_samples > 0 else float('inf')
    return {'event_recall': recall, 'event_iou': iou, 'event_far_per_hour': far_hr,
            'TP_events': TP, 'FN_events': FN, 'FP_events': FP, 'n_pred_events': len(pred_events)}


# ---------- Evaluation with compact strategy loop ----------
def evaluate_joint_anomaly_score(dataset_index: int = 1, sampling_rate: float = 8.0,
                                 results_path: Optional[str] = None,
                                 smoothing_window_s: float = 60.0,
                                 smoothing_threshold_s: float = 50.0,
                                 include_smoothing: bool = True):
    """
    Evaluate joint anomaly score with multiple strategies.
    
    Args:
        dataset_index: Dataset identifier
        sampling_rate: Sampling rate in Hz
        results_path: Path to joint anomaly score pickle file
        best_freq_threshold_scale: Scale factor for best frequency threshold
        smoothing_window_s: Smoothing window size in seconds
        smoothing_threshold_s: Smoothing threshold in seconds
        include_smoothing: Whether to include smoothed versions of strategies
        
    Returns:
      results (dict name->metrics), strategy_data (name->{'mask','scores'}),
      truth (bool array), include_smoothing (bool)
    """
    joint_file = (results_path or
                  "/home/mballo_sw/Repositories/ecg-seizure-detection/TimeVQVAE-AD/evaluation/results/Test2-1/Try 2/001-joint_anomaly_score.pkl")
    if not os.path.exists(joint_file):
        raise FileNotFoundError(f"Joint anomaly score file not found: {joint_file}")

    print(f"Loading joint anomaly score from: {joint_file}")
    R = load_pkl(joint_file)

    a_s_star = R['a_s^*']              # shape [H, T]
    a_bar_s_star = R['bar{a}_s^*']     # [T]
    a_final = R['a_final']             # [T]
    joint_threshold = R['joint_threshold']  # [H]
    final_threshold = R['final_threshold']
    Y = np.array(R['Y']).astype(bool)

    # Build strategies in a single loop
    strategies = []
    w = int(round(smoothing_window_s * sampling_rate))
    th = int(round(smoothing_threshold_s * sampling_rate))
    
    # final
    strategies.append({
        'name': 'final_score',
        'scores': a_final,
        'mask': (a_final > final_threshold)
    })
    # best_freq
    best_freq_idx = int(np.argmin(joint_threshold))
    min_thr = float(np.min(joint_threshold))
    strategies.append({
        'name': 'best_freq',
        'scores': a_s_star[best_freq_idx],
        'mask': (a_s_star[best_freq_idx] > min_thr)
    })
    # bar_score
    strategies.append({
        'name': 'bar_score',
        'scores': a_bar_s_star,
        'mask': (a_bar_s_star > final_threshold)
    })
    
    # Add smoothed versions only if requested
    if include_smoothing:
        # final_smoothed
        strategies.append({
            'name': 'final_smoothed',
            'scores': a_final,
            'mask': smooth_labels((a_final > final_threshold).astype(int), w, th).astype(bool)
        })
        # best_freq_smoothed
        strategies.append({
            'name': 'best_freq_smoothed',
            'scores': a_s_star[best_freq_idx],
            'mask': smooth_labels((a_s_star[best_freq_idx] > min_thr).astype(int), w, th).astype(bool)
        })
        # bar_score_smoothed
        strategies.append({
            'name': 'bar_score_smoothed',
            'scores': a_bar_s_star,
            'mask': smooth_labels((a_bar_s_star > final_threshold).astype(int), w, th).astype(bool)
        })

    results: Dict[str, Dict] = {}
    strategy_data: Dict[str, Dict[str, np.ndarray]] = {}

    print(f"\nDataset {dataset_index} Evaluation Results:")
    print("=" * 50)
    print(f"Total samples: {len(Y)} | True anomaly samples: {int(Y.sum())} "
          f"({(Y.sum()/len(Y))*100:.2f}%)")

    for s in strategies:
        name, mask, scores = s['name'], s['mask'].astype(bool), s['scores']
        metrics = evaluate_event_detection(mask, Y, fs=sampling_rate)
        results[name] = metrics
        strategy_data[name] = {'mask': mask, 'scores': scores}
        # print(f"\n{name}:")
        # print(f"   Predicted anomaly samples: {int(mask.sum())} "
        #       f"({(mask.sum()/len(mask))*100:.2f}%)")
        # print(f"   True events: {len(metrics['truth_events'])} | Predicted events: {len(metrics['pred_events'])}")
        # print(f"   TP={metrics['TP_events']}, FN={metrics['FN_events']}, FP={metrics['FP_events']}")
        # print(f"   Recall: {metrics['recall']:.3f} | IoU: {metrics['event_iou']:.3f} "
        #       f"| FAR/hr: {metrics['false_alarm_rate_per_hour']:.2f}")

    # Summary table
    print(f"\n{'='*50}\nSUMMARY COMPARISON:\n{'='*50}")
    print(f"{'Strategy':<20} {'Recall':<8} {'FAR/hr':<8} {'Event-IoU':<10} {'TP':<4} {'FN':<4} {'FP':<4}")
    print("-" * 80)
    
    # Dynamic strategy list based on what was actually created
    strategy_names = ['final_score', 'best_freq', 'bar_score']
    if include_smoothing:
        strategy_names.extend(['final_smoothed', 'best_freq_smoothed', 'bar_score_smoothed'])
    
    for name in strategy_names:
        if name in results:  # Safety check
            m = results[name]
            print(f"{name.replace('_',' ').title():<20} {m['recall']:<8.3f} {m['false_alarm_rate_per_hour']:<8.2f} "
                  f"{m['event_iou']:<10.3f} {m['TP_events']:<4} {m['FN_events']:<4} {m['FP_events']:<4}")

    return results, strategy_data, Y, include_smoothing


# ---------- Plot ----------
def plot_evaluation_results(dataset_index: int = 1, results_dir: str = 'results'):
    joint_file = os.path.join(results_dir, f'{dataset_index:03d}-joint_anomaly_score.pkl')
    if not os.path.exists(joint_file):
        raise FileNotFoundError(f"Joint anomaly score file not found: {joint_file}")
    R = load_pkl(joint_file)
    a_final, final_threshold = R['a_final'], R['final_threshold']
    X_test_unscaled, Y = R['X_test_unscaled'], R['Y']
    pred_final = a_final > final_threshold

    fig, axes = plt.subplots(4, 1, figsize=(20, 12))
    axes[0].plot(X_test_unscaled, color='black', label='ECG Signal')
    axes[0].set_title(f'Dataset {dataset_index}: ECG Signal and Ground Truth', fontsize=14)
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(Y, alpha=0.7, color='red', label='Ground Truth')
    ax0_twin.set_ylabel('Ground Truth', color='red')
    axes[0].set_ylabel('ECG Amplitude')
    axes[0].legend(loc='upper left')
    ax0_twin.legend(loc='upper right')

    axes[1].plot(a_final, color='blue', label='Final Anomaly Score')
    axes[1].axhline(y=final_threshold, color='red', linestyle='--', label='Threshold')
    axes[1].set_title('Final Anomaly Score'); axes[1].set_ylabel('Anomaly Score'); axes[1].legend()

    axes[2].plot(Y, alpha=0.7, color='red', label='Ground Truth', linewidth=2)
    axes[2].plot(pred_final.astype(int), alpha=0.7, color='blue', label='Predictions', linewidth=1)
    axes[2].set_title('Predictions vs Ground Truth'); axes[2].set_ylabel('Binary Labels'); axes[2].legend()

    axes[3].plot(X_test_unscaled, color='black', alpha=0.7, label='ECG Signal')
    ax3 = axes[3].twinx()
    ax3.fill_between(range(len(Y)), 0, Y, alpha=0.3, color='red', label='True Anomalies')
    ax3.fill_between(range(len(pred_final)), 0, pred_final.astype(int), alpha=0.3, color='blue', label='Predicted Anomalies')
    axes[3].set_title('Combined View: Signal with Anomaly Regions'); axes[3].set_ylabel('ECG Amplitude')
    axes[3].set_xlabel('Sample Index'); ax3.set_ylabel('Anomaly Regions')
    axes[3].legend(loc='upper left'); ax3.legend(loc='upper right')
    plt.tight_layout(); plt.show()
    return fig


# ---------- Main ----------
if __name__ == "__main__":
    print("Evaluating Joint Anomaly Score for Dataset 001")
    print("=" * 60)
    sampling_rate = 8.0
    dataset_index = 1

    results, strategy_data, truth, include_smoothing = evaluate_joint_anomaly_score(
        dataset_index=dataset_index,
        sampling_rate=sampling_rate,
        # Optional knobs:
        # #best_freq_threshold_scale=1
        smoothing_threshold_s=15.0,
        include_smoothing=True  # Set to False to disable smoothing strategies
    )

    # Choose the best *pred* source with improved strategy selection
    # Prioritize strategies with good recall and reasonable FAR
    best_strategy = None
    best_score = -1e9
    
    for name, m in results.items():
        # Skip strategies with zero recall (no detections)
        if m['recall'] == 0:
            continue
            
        # Improved scoring: balance recall and FAR with IoU consideration
        # Penalize high FAR more heavily and reward higher IoU
        far_penalty = min(m['false_alarm_rate_per_hour'] / 5.0, 1.0)  # Cap penalty at 1.0
        
        # Score formula: prioritize recall, penalize FAR, bonus for IoU
        score = m['recall'] - far_penalty

        if score > best_score:
            best_score = score
            best_strategy = name

    print("\nBest performing strategy (event-level):", best_strategy)
    
    if best_strategy is None:
        print("No strategy with positive recall found; using fallback selection.")
        # Fallback: choose strategy with lowest FAR among those with any detections
        candidates = [(name, m) for name, m in results.items() if np.any(strategy_data[name]['mask'])]
        if candidates:
            best_strategy = min(candidates, key=lambda x: x[1]['false_alarm_rate_per_hour'])[0]
            print("Fallback strategy (lowest FAR with detections):", best_strategy)
        else:
            # Last resort: just pick the first strategy
            best_strategy = list(results.keys())[0]
            print("Last resort fallback strategy:", best_strategy)
    
    mask = strategy_data[best_strategy]['mask']
    scores = strategy_data[best_strategy]['scores']

    if not np.any(mask):
        print("No positives in chosen strategy; falling back to first with positives.")
        # Dynamic fallback list based on whether smoothing is enabled
        fallback_order = (['bar_score_smoothed', 'best_freq_smoothed', 'final_smoothed'] if include_smoothing else []) + \
                        ['bar_score', 'best_freq', 'final_score']
        
        for name in fallback_order:
            if name in strategy_data and np.any(strategy_data[name]['mask']):
                best_strategy = name
                mask = strategy_data[name]['mask']
                scores = strategy_data[name]['scores']
                print("Fallback strategy:", best_strategy)
                break

    if np.any(mask):
        print(f"\nRunning clustering using '{best_strategy}' ({mask.sum()} positive samples).")
        gt_intervals = intervals_from_mask(truth, sampling_rate)

        results_cluster = analyze_detection_mask(
            mask=mask,
            fs=sampling_rate,
            file_id=f"rec_{dataset_index:03d}",
            scores=scores,
            subject_id=f"subj_{dataset_index:03d}",
            gt_intervals=gt_intervals,
            output_folder=f"rec_{dataset_index:03d}_smart_clustered"
        )

        # ---- Show BOTH: file-level (from clustering) and event-level (ours) ----
        best = results_cluster["best_results"]
        file_level = best["metrics"]  # sensitivity/precision/FAR (proportions), file-based
        print("\n=== SMART CLUSTERING (file-level) ===")
        print("Best strategy:", results_cluster["best_strategy"])
        print("Clusters:", best["clusters"], "Representatives:", len(best["representatives"]))
        print(f"File-level Sensitivity: {file_level['sensitivity']:.3f}  "
              f"Precision: {file_level['precision']:.3f}  "
              f"FAR (proportion): {file_level['false_alarm_rate']:.3f}")

        ev = event_metrics_for_clusters(best["representatives"], gt_intervals, len(truth), sampling_rate)
        print("\n--- Event-level metrics on clustered output ---")
        print(f"Recall: {ev['event_recall']:.3f}  IoU: {ev['event_iou']:.3f}  FAR/hr: {ev['event_far_per_hour']:.2f}  "
              f"(TP={ev['TP_events']}, FN={ev['FN_events']}, FP={ev['FP_events']}, N_pred={ev['n_pred_events']})")

        # Quick visualization
        t = np.arange(len(scores)) / sampling_rate
        plt.figure(figsize=(14, 4))
        plt.plot(t, scores, label=f'Score ({best_strategy})')
        plt.plot(t, mask.astype(int), alpha=0.5, label='Mask')
        first = True
        for r in best["representatives"]:
            # use cluster span if present
            s = r.get('cluster_start_seconds', r.get('segment_start_seconds', r['location_time_seconds']))
            e = r.get('cluster_end_seconds', r.get('segment_end_seconds', r['location_time_seconds']))
            plt.axvspan(s, e, alpha=0.2, label='Cluster' if first else None)
            first = False
        plt.xlabel("Time (s)"); plt.legend(); plt.title("Score, mask, and clustered representatives")
        plt.tight_layout(); plt.show()
    else:
        print("No positive predictions in any strategy; skipping clustering.")