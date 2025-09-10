"""
Event metrics calculation for clustering analysis.
Separated from eval.py to avoid circular imports.
"""

from typing import List, Tuple, Dict


def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Check if two intervals overlap."""
    return max(a[0], b[0]) < min(a[1], b[1])


def event_metrics_for_clusters(representatives: List[Dict], gt_intervals: List[Tuple[float, float]],
                               total_samples: int, fs: float) -> Dict[str, float]:
    """
    Calculate event-based metrics for clustered representatives.
    
    Args:
        representatives: List of cluster representatives with timing information
        gt_intervals: Ground truth seizure intervals [(start_sec, end_sec), ...]
        total_samples: Total number of samples in the recording
        fs: Sampling rate in Hz
        
    Returns:
        Dictionary with event-based metrics including sensitivity, IoU, FAR, etc.
    """
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
    
    Sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
    iou = TP / (TP + FN + FP) if (TP + FN + FP) else 0.0
    far_hr = FP / (total_samples / fs / 3600.0) if total_samples > 0 else float('inf')
    
    return {
        'event_sensitivity': Sensitivity, 
        'event_recall': Sensitivity,  # Add alias for backward compatibility
        'event_iou': iou, 
        'event_far_per_hour': far_hr,
        'TP_events': TP, 
        'FN_events': FN, 
        'FP_events': FP, 
        'n_pred_events': len(pred_events)
    }
