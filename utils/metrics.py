import numpy as np
from typing import List, Tuple

def compute_sensitivity_false_alarm_rate_classic(
    label_sequences: List[np.ndarray],
    detection_indices: List[List[int]]
) -> Tuple[float, float]:
    """
    Computes the sensitivity and false alarm rate based on ground truth label sequences and detection indices.
    
    Args:
        label_sequences (List[np.ndarray]): List of binary numpy arrays (0 or 1) indicating ground truth events.
        detection_indices (List[List[int]]): List of lists with detection indices corresponding to each label array.
    
    Returns:
        Tuple[float, float]: A tuple containing (sensitivity, false alarm rate)
        
        - Sensitivity: TP / (TP + FN)
        - False Alarm Rate: FP / (TP + FP)
    """
    true_positives = 0
    false_positives = 0
    total_positives = 0

    for labels, indices in zip(label_sequences, detection_indices):
        labels = np.asarray(labels)
        total_positives += np.count_nonzero(labels)

        for idx in indices:
            if 0 <= idx < len(labels):
                if labels[idx] == 1:
                    true_positives += 1
                else:
                    false_positives += 1

    false_negatives = total_positives - true_positives

    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    false_alarm_rate = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    return sensitivity, false_alarm_rate

import numpy as np
from typing import List, Tuple


def compute_sensitivity_false_alarm_rate_seizures(
    label_sequences: List[np.ndarray],
    detection_indices: List[List[int]]
) -> Tuple[float, float]:
    """
    Computes event-based sensitivity and false alarm rate using efficient indexing.
    
    An event is defined as a contiguous block of 1's in the label array.
    A detection is a true positive if it falls within such a block.
    
    Args:
        label_sequences (List[np.ndarray]): Binary label arrays (0 or 1) indicating event regions.
        detection_indices (List[List[int]]): Lists of detection indices corresponding to each label sequence.
    
    Returns:
        Tuple[float, float]: (sensitivity, false alarm rate)
        
        - Sensitivity = true_positives / total_events
        - False Alarm Rate = false_positives / total_detections
    """
    total_events = 0
    true_positives = 0
    false_positives = 0
    total_detections = 0

    for labels, indices in zip(label_sequences, detection_indices):
        labels = np.asarray(labels)
        indices_set = set(indices)
        total_detections += len(indices)

        # Efficient extraction of event regions
        event_indices = np.where(labels == 1)[0]
        if event_indices.size == 0:
            continue

        # Identify gaps to separate contiguous event blocks
        gaps = np.where(np.diff(event_indices) > 1)[0]
        split_points = np.split(event_indices, gaps + 1)
        event_regions = [(event[0], event[-1]) for event in split_points]

        total_events += len(event_regions)

        # Count true positive events
        for start, end in event_regions:
            if any(start <= idx <= end for idx in indices_set):
                true_positives += 1

        # Count false positives: detections outside all event regions
        for idx in indices:
            if not any(start <= idx <= end for start, end in event_regions):
                false_positives += 1

    sensitivity = true_positives / total_events if total_events > 0 else 0.0
    false_alarm_rate = false_positives / total_detections if total_detections > 0 else 0.0

    return sensitivity, false_alarm_rate

def compute_sensitivity_false_alarm_rate_timing_tolerance(
    label_sequences: List[np.ndarray],
    detection_indices: List[List[int]],
    lower: int,
    upper: int,
    frequency: float
) -> Tuple[float, float]:
    """
    Computes event-based sensitivity and false alarm rate with timing tolerance.
    
    An event is defined as a contiguous block of 1s in the label array.
    A detection is a true positive if it falls within the event block extended by a tolerance range.
    
    Args:
        label_sequences (List[np.ndarray]): Binary label arrays (0 or 1) indicating event regions.
        detection_indices (List[List[int]]): Lists of detection indices corresponding to each label sequence.
        lower (int): Time in seconds to extend before the start of an event.
        upper (int): Time in seconds to extend after the end of an event.
        frequency (float): Sampling frequency in Hz (used to convert seconds to samples).
    
    Returns:
        Tuple[float, float]: (sensitivity, false alarm rate)
        
        - Sensitivity = true_positives / total_events
        - False Alarm Rate = false_positives / total_detections
    """
    total_events = 0
    true_positives = 0
    false_positives = 0
    total_detections = 0
    hours = 0
    tolerance_left = int(lower * frequency)
    tolerance_right = int(upper * frequency)

    for labels, indices in zip(label_sequences, detection_indices):
        hours += (labels.__len__()/frequency)/3600
        labels = np.asarray(labels)
        indices_set = set(indices)
        total_detections += len(indices)

        # Efficient extraction of event regions
        event_indices = np.where(labels == 1)[0]
        if event_indices.size == 0:
            false_positives += len(indices)
            continue

        # Identify contiguous event regions
        gaps = np.where(np.diff(event_indices) > 1)[0]
        split_points = np.split(event_indices, gaps + 1)
        event_regions = [(max(0, event[0] - tolerance_left), min(len(labels) - 1, event[-1] + tolerance_right))
                         for event in split_points]

        total_events += len(event_regions)

        # Count true positives (detections within extended regions)
        for start, end in event_regions:
            if any(start <= idx <= end for idx in indices_set):
                true_positives += 1

        # Count false positives (detections outside all extended regions)
        for idx in indices:
            if not any(start <= idx <= end for start, end in event_regions):
                false_positives += 1

    # sensitivity = true_positives / total_events if total_events > 0 else 0.0
    # false_alarm_rate = false_positives / total_detections if total_detections > 0 else 0.0
    false_alarms_per_hour = false_positives/hours
    # print(f"False alarms per hour {false_positives/hours}")
    # return true_positives, false_positives, hours, total_events
    return true_positives, false_positives, hours, total_events

    # return sensitivity, false_alarm_rate, true_positives, false_positives

