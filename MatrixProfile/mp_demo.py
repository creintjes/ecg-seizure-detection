import sys
from pathlib import Path
import re
import pickle
import numpy as np
from datetime import datetime

def save_numpy_array_list(array_list: list[np.ndarray], name:str) -> None:
    """
    Saves a list of NumPy arrays to a compressed .npz file with a timestamped filename.
    
    Parameters:
    ----------
    array_list : list[np.ndarray]
        List of NumPy arrays to save.
    """
    timestamp = datetime.now().strftime("%H-%M-%S")
    filename = f"{name}_{timestamp}.pkl"
    
    with open(filename, "wb") as f:
        pickle.dump(array_list, f)


def MatProfDemo()-> None:
    # Add parent directory (../) to sys.path
    project_root = Path().resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data_helpers import load_preprocessed_samples

    # DATA_DIRECTORY = "../results/preprocessed_all"

    
    DATA_DIRECTORY = "/home/swolf/asim_shared/preprocessed_data/downsample_freq=32,window_size=3600_0,stride=1800_0"
    match = re.search(r"downsample_freq=(\d+)", DATA_DIRECTORY)
    downsample_freq: int = int(match.group(1))
    samples, labels = load_preprocessed_samples(data_dir=DATA_DIRECTORY, max_loaded_files=350)
    samples.__len__()
    amount_samples : int = 3000
    example_samples = samples[:amount_samples]
    from matrix_profile import MatrixProfile
    matrix_profiles = []
    counter:int = 0
    subsequence_length:int = downsample_freq*70 # Assuming seizure of max. 70 sec
    for s in example_samples:
        matrix_profiles.append(MatrixProfile.calculate_matrix_profile_for_sample(sample=s, subsequence_length=subsequence_length))
        counter +=1
        if counter % 100 == 0:
            print(counter)

    save_numpy_array_list(array_list=matrix_profiles, name=list(DATA_DIRECTORY.split("/"))[-1])

if __name__ == "__main__":
    MatProfDemo()    
