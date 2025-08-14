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
    features, timestamps = MatrixProfile.process_ecg_to_hrv_features(data_ecg, sampling_rate=frequency)
    print("Feature shape:", features.shape)
    anomalies, tp, fp, mp = MatrixProfile.detect_anomalies_from_hrv_features(
        features,
        timestamps,
        subsequence_length=20,
        ground_truth_intervals=gt_intervals,
        sampling_rate=256,
        top_k_percent=10.0
    )

    print(f"Anomalien (Samples): {anomalies}")
    print(f" Treffer: {tp}, Falsch: {fp}")

    print(f"Top 10% Anomalien (Samples): {anomalies}")

    # mp, indices = MatrixProfile.compute_multivariate_matrix_profile(features, subsequence_length=subsequence_length)

    # save_one_mp(mp=mp, folder=results_path, run_name="sub-089_run-97")
    print(f'Ended MP calc at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    


if __name__ == "__main__":
    MatProfDemo()    

