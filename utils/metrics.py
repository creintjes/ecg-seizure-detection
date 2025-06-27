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
