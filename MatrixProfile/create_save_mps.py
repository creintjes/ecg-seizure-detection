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
# Add parent directory (../) to sys.path
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# from data_helpers import load_one_preprocessed_sample
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

    DATA_DIRECTORY = f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"
    results_path = Path(
        "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/mv_mp"
        # f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"
    )
    results_path.mkdir(parents=True, exist_ok=True)

    filenames = [filename for filename in os.listdir(DATA_DIRECTORY) if filename.endswith("_preprocessed.pkl")]

    counter:int = 0
    for filename in filenames[:amount_samples]:
        data, label = load_one_preprocessed_sample(filepath=os.path.join(DATA_DIRECTORY, filename))
        data = data[0]
        run_name = filename[:-17]
        # print(data)
        if multi_variate_matrix_profile:
            print()
            print(len(data))
            features = MatrixProfile.process_ecg_to_hrv_features(data, sampling_rate=downsample_freq)
            print("Feature shape:", features.shape)
            mp, indices = MatrixProfile.compute_multivariate_matrix_profile(features, subsequence_length=subsequence_length)
    
        else:

            if approx_matrix_profile:
                mp = MatrixProfile.compute_approx_matrix_profile(time_series=data, subsequence_length=subsequence_length, percentage=0.1)
            else:
                mp = MatrixProfile.calculate_matrix_profile_for_sample(sample=data, subsequence_length=subsequence_length)
        save_one_mp(mp=mp, folder=results_path, run_name=run_name)
        counter +=1
        if counter % printer_int == 0:
            print(counter)
    print(f'Ended MP calc at {datetime.now().strftime("%d.%m.%Y, %H:%M")}')
    


if __name__ == "__main__":
    MatProfDemo()    

