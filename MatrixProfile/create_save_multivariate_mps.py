import sys
from pathlib import Path
import re
import pickle
import numpy as np
from datetime import datetime
from matrix_profile import MatrixProfile
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Determine project root as the parent of 'pa' folder (adjust according to your structure)
project_root = Path(__file__).resolve().parent.parent  # If g.py is in pa/test, parent.parent ist pa/

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from utils.metrics import compute_sensitivity_false_alarm_rate_timing_tolerance

import os
seizeit2_main_path = Path(__file__).resolve().parent / '..' / 'Information' / 'Data' / 'seizeit2-main'
seizeit2_main_path = seizeit2_main_path.resolve()

# Add seizeit2-main to sys.path if not already present
if str(seizeit2_main_path) not in sys.path:
    sys.path.insert(0, str(seizeit2_main_path))

# Now the imports should work
from classes.data import Data
from classes.annotation import Annotation
def load_data(
        data_path: str, 
        subject_id: str, 
        run_id: str
    ):
        """
        Load ECG data and annotations for a specific recording.
        
        Args:
            data_path: Path to SeizeIT2 dataset
            subject_id: Subject identifier (e.g., 'sub-001')
            run_id: Run identifier (e.g., 'run-01')
            
        Returns:
            Tuple of (Data object, Annotation object)
        """
        recording = [subject_id, run_id]
        
        # Load data with ECG modality
        data = Data.loadData(data_path, recording, modalities=['ecg'])
        
        # Load annotations
        annotations = Annotation.loadAnnotation(data_path, recording)
        
        return data, annotations


def load_one_preprocessed_sample(filepath: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load preprocessed ECG window samples from pickle files and collect them into a list.

    Args:
        data_dir: Path to directory containing the preprocessed .pkl files.

    Returns:
        List of windowed ECG samples as numpy arrays and a List with their according labels [0 or 1].
    """
    # Skip empty files
    if os.path.getsize(filepath) == 0:
        print(50*"-")
        print(f"Skipped empty file: {filepath}")
        return None

    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

            if not data or "channels" not in data:
                print(50*"-")
                print(f"Missig channels or data in file: {filepath}")
                return None
                

            # Since data only cover 1 channel we can use chanell[0]
            channel_data = data["channels"][0]
            windows = channel_data.get("windows", [])
            labels = channel_data.get("labels", [])

    except (EOFError, pickle.UnpicklingError) as e:
        # print(f"Warning: {filename} is empty or corrupted.")
        print(f"Corrupted pickle file: {filepath} ({e})")
        return None
    return windows, labels

def save_numpy_array_list(array_list: list[np.ndarray], name:str, path:str) -> None:
    """
    Saves a list of NumPy arrays to a compressed .npz file with a timestamped filename.
    
    Parameters:
    ----------
    array_list : list[np.ndarray]
        List of NumPy arrays to save.
    """
    # timestamp = datetime.now().strftime("%H-%M-%S")
    timestamp = datetime.now().strftime("%d.%m.%Y, %H:%M")
    filename = f"/home/swolf/asim_shared/results/MP/{name}_{timestamp}.pkl"
    
    with open(filename, "wb") as f:
        pickle.dump(array_list, f)

def save_one_mp(mp:np.ndarray, folder:str, run_name:str):
    filename = f"{folder}/mp_{run_name}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(mp, f)

def _build_downsampled_labels(length_samples: int,
                              fs: int,
                              gt_intervals: list[list[int]],
                              downsample_freq: int) -> np.ndarray:
    """
    Build a downsampled binary label sequence from ground-truth sample intervals.
    """
    duration_sec = length_samples / float(fs)
    L = int(round(duration_sec * downsample_freq))
    labels = np.zeros(L, dtype=np.int8)

    scale = downsample_freq / float(fs)
    for start_samp, end_samp in gt_intervals:
        ds_start = max(0, int(np.floor(start_samp * scale)))
        ds_end   = min(L - 1, int(np.ceil(end_samp * scale)))
        if ds_end >= ds_start:
            labels[ds_start:ds_end + 1] = 1
    return labels


def _map_anomaly_seconds_to_downsample_indices(times_sec: list[float],
                                               downsample_freq: int) -> list[int]:
    """
    Map anomaly timestamps (seconds) to integer indices on a downsampled grid.
    """
    return [int(round(t * downsample_freq)) for t in times_sec]


def MatProfDemo()-> None:
    print(f'Started MP calc at {datetime.now().strftime("%d.%m.%Y, %H:%M")}')
    
    data_path = "/home/swolf/asim_shared/raw_data/ds005873-1.1.0" 

    downsample_freq: int=8
    window_size_sec:int = 25
    subsequence_length:int = downsample_freq*window_size_sec # Assuming seizure of max. N sec
    amount_samples : int = 2
    approx_matrix_profile: bool = False
    multi_variate_matrix_profile: bool = True
    printer_int = 100
    # Add parent directory (../) to sys.path
    project_root = Path().resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    results_path = Path(
        "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/mv_mp"
    )
    results_path.mkdir(parents=True, exist_ok=True)

    data, annotations = load_data(data_path=data_path, subject_id="sub-012", run_id="run-03")
    # data = data[0]
    # print(data.shape)
    # print(len(data))
    print(data)
    print(annotations.events)
    gt_intervals = annotations.events
    if not gt_intervals:
        return
    # annomaly_indices = [int(idx) for idx in annomaly_indices]
    # print(annomaly_indices)
    # return
    data_ecg = None
    for i, (channel_data, channel_name, fs) in enumerate(
                zip(data.data, data.channels, data.fs)
            ):
                if 'ecg' not in channel_name.lower():
                    continue
                
                print(f"Processing channel: {channel_name}")
                data_ecg=channel_data
                break
    print(int(data.fs[0]))
    frequency = int(data.fs[0])
    print(data_ecg.shape)
    # return
    # features, timestamps = MatrixProfile.process_ecg_to_hrv_features(data_ecg, sampling_rate=frequency)
    # print("Feature shape:", features.shape)
    # anomalies, tp, fp, mp = MatrixProfile.detect_anomalies_from_hrv_features(
        # features,
        # timestamps,
        # subsequence_length=20,
        # ground_truth_intervals=gt_intervals,
        # sampling_rate=256,
        # top_k_percent=10.0
    # )


    # Toggle here: HRV features (slow) vs. lightweight FFT features (fast)
    use_lightweight_features: bool = True  

    if use_lightweight_features:
        feature_list = [
            "std", "min", "max", "line_length", "rms",
            "total_power", "mean_delta_power", "mean_theta_power",
            "mean_alpha_power", "mean_beta_power"
        ]

        features, timestamps = MatrixProfile.compute_ecg_window_features(
            ecg=data_ecg,
            sampling_rate=frequency,
            window_size_sec=5.0,
            step_size_sec=2.5,
            feature_names=feature_list,
            min_freq_hz=0.5,
            max_freq_hz=40.0
        )
        print("Lightweight feature shape:", features.shape)

        mp, nn_idx = MatrixProfile.compute_multivariate_matrix_profile(
            features=features,
            subsequence_length=20
        )

        # Use mean MP across feature dimensions as anomaly score
        mp_mean: np.ndarray = mp.mean(axis=0)

        # Pick top-k (10%) anomalies in feature-index space
        k: int = max(1, int(0.10 * len(mp_mean)))
        anomaly_feature_indices: list[int] = MatrixProfile.get_top_k_anomaly_indices(mp_mean, k=k)

        # Map feature-index → time (sec) using subsequence center
        m: int = 20  # must match subsequence_length used in mstump
        anomaly_times_sec: list[float] = []
        for i in anomaly_feature_indices:
            j = min(i + m // 2, len(timestamps) - 1)  # center of the subsequence
            anomaly_times_sec.append(float(timestamps[j]))

        # Map seconds → downsampled detection indices
        detection_indices_down: list[int] = _map_anomaly_seconds_to_downsample_indices(
            times_sec=anomaly_times_sec,
            downsample_freq=downsample_freq
        )

        # Optional: de-duplicate very close hits
        detection_indices_down = sorted(set(detection_indices_down))

        # Build downsampled label sequence from GT intervals
        label_ds: np.ndarray = _build_downsampled_labels(
            length_samples=len(data_ecg),
            fs=frequency,
            gt_intervals=gt_intervals,
            downsample_freq=downsample_freq
        )

        # Prepare inputs for the metrics function
        label_list: list[np.ndarray] = [label_ds]
        anomaly_indices_cons: list[list[int]] = [detection_indices_down]

        # Timing tolerance (in seconds)
        pre_thresh_sec: int = 10
        post_thresh_sec: int = 10

        # Compute TP/FP (plus hours, total_events) with timing tolerance
        true_positives, false_positives, hours, total_events = compute_sensitivity_false_alarm_rate_timing_tolerance(
            label_sequences=label_list,
            detection_indices=anomaly_indices_cons,
            lower=pre_thresh_sec,
            upper=post_thresh_sec,
            frequency=downsample_freq
        )

    else:
        features, timestamps = MatrixProfile.process_ecg_to_hrv_features(
            data_ecg,
            sampling_rate=frequency
        )
        print("HRV feature shape:", features.shape)

        anomalies, tp, fp, mp = MatrixProfile.detect_anomalies_from_hrv_features(
            features,
            timestamps,
            subsequence_length=20,
            ground_truth_intervals=gt_intervals,
            sampling_rate=256,
            top_k_percent=10.0
        )

    print(f"Anomalien (Feature-Idx): {anomaly_feature_indices if use_lightweight_features else anomalies}")
    if use_lightweight_features:
        print(f"Detections (downsample idx): {anomaly_indices_cons[0]}")
        print(f"TP: {true_positives}, FP: {false_positives}, hours: {hours:.2f}, total_events: {total_events}")
    else:
        print(f"TP: {tp}, FP: {fp}")
    print(f'Ended MP calc at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == "__main__":
    MatProfDemo()    

