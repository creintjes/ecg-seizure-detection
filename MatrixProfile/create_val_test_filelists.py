import os
import random
import pandas as pd
from pathlib import Path
from typing import Tuple

import os
import random
import pandas as pd
from pathlib import Path
from typing import Tuple, List

def extract_subject_run_pairs(mps_path: str) -> List[str]:
    """
    Extract subject_run strings from MP filenames using same logic as produce_mp_results.

    Args:
        mps_path (str): Path where MP files are stored.

    Returns:
        List[str]: List of subject_run identifiers, e.g., "sub-119_run-34"
    """
    mps_filenames = [f for f in os.listdir(mps_path) if f.endswith(".pkl") and f.startswith("mp_")]
    # remove prefix "mp_" and suffix ".pkl", e.g. mp_sub-119_run-34.pkl -> sub-119_run-34
    subject_run_pairs = [filename[3:-4] for filename in mps_filenames]
    return subject_run_pairs

def create_val_test_split(
    mps_path: str,
    output_dir: str,
    total_records: int,
    test_ratio: float = 0.6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a random split of subject_run filenames into test and val Excel files.

    Args:
        mps_path (str): Path to directory with MP files.
        output_dir (str): Directory where the Excel files will be saved.
        total_records (int): Number of total subject_run entries to sample.
        test_ratio (float): Ratio of test set vs val set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for test and val.
    """
    all_subject_runs = extract_subject_run_pairs(mps_path)

    if total_records > len(all_subject_runs):
        raise ValueError(f"Requested {total_records} records, but only {len(all_subject_runs)} available.")

    random.shuffle(all_subject_runs)
    selected = all_subject_runs[:total_records]

    n_test = int(test_ratio * total_records)
    test_list = selected[:n_test]
    val_list = selected[n_test:]

    test_df = pd.DataFrame(test_list, columns=["subject_run"])
    val_df = pd.DataFrame(val_list, columns=["subject_run"])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    test_path = Path(output_dir) / f"test_files_{len(test_df)}.xlsx"
    val_path = Path(output_dir) / f"val_files_{len(val_df)}.xlsx"

    test_df.to_excel(test_path, index=False)
    val_df.to_excel(val_path, index=False)

    print(f"Test list saved to: {test_path}")
    print(f"Val list saved to: {val_path}")

    return test_df, val_df


if __name__ == "__main__":
    create_val_test_split(
        mps_path="/home/swolf/asim_shared/results/MP/downsample_freq=8,no_windows/seq_len25sec",
        output_dir="/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/configs/splits",
        total_records=500,
        test_ratio=0.6
    )
