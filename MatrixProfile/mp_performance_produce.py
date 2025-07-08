from pathlib import Path
from typing import List
import pickle
import re
import os
import itertools
import csv
import sys
from typing import Any, Callable, Dict, List
import pandas as pd

import sys
from pathlib import Path

# Determine project root as the parent of 'pa' folder (adjust according to your structure)
project_root = Path(__file__).resolve().parent.parent  # If g.py is in pa/test, parent.parent ist pa/

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now you can import utils.metrics as if project_root is root of your package
from utils.metrics import compute_sensitivity_false_alarm_rate_timing_tolerance

from matrix_profile import MatrixProfile

def find_files_with_prefix(directory: str, prefix: str) -> List[Path]:
    """
    Finds all files in the given directory that start with the specified prefix.

    Args:
        directory (str): Path to the directory to search.
        prefix (str): The prefix that filenames should start with.

    Returns:
        List[Path]: List of Path objects for matching files.
    """
    dir_path = Path(directory)
    return [file for file in dir_path.iterdir() if file.is_file() and file.name.startswith(prefix)]

def produce_mp_results(amount_of_annomalies_per_record: int, amount_of_records:int, batch_size_load :int, downsample_freq:int, max_gap_annos_in_sec:int, n_cons:int, window_size_sec:int, pre_thresh_sec:int, post_thresh_sec:int, DIR_preprocessed:str, MPs_path:str, verbose:bool):
    mps_filenames = [filename for filename in os.listdir(MPs_path) if filename.endswith(".pkl")]
    # preprocessed_filenames = [filename for filename in os.listdir(DIR_preprocessed) if filename.endswith("_preprocessed.pkl")]
    mps_list = []
    label_list = []
    tp_list = []
    fp_list = []
    hours_list = []
    total_events_list = []
    for i, mp_filename in enumerate(mps_filenames[:amount_of_records], 1):
        mp_filename=mp_filename.__str__()
        # Load MP
        with open(os.path.join(MPs_path, mp_filename), "rb") as f:
            mps_list.append(pickle.load(f)[:, 0].reshape(-1, 1))
        with open(os.path.join(DIR_preprocessed, mp_filename[3:-4]+"_preprocessed.pkl"), "rb") as g:
            label_list.append(pickle.load(g)["channels"][0]["labels"][0])
        if i%batch_size_load ==0:
            if verbose:
                print(i)
            if not len(label_list)==len(mps_list):
                print(f"len(label_list) not ==len(mps_list)")
            annomaly_indices = [MatrixProfile.get_top_k_anomaly_indices(matrix_profile=mp.flatten(), k=amount_of_annomalies_per_record)
                        for mp in mps_list]
            annomaly_indices_cons = [MatrixProfile.mean_of_all_consecutive_anomalies(indices=annos, n=n_cons, max_gap = downsample_freq*max_gap_annos_in_sec)
                            for annos in annomaly_indices]
            true_positives, false_positives, hours, total_events = compute_sensitivity_false_alarm_rate_timing_tolerance(
                label_sequences=label_list, detection_indices=annomaly_indices_cons, lower=pre_thresh_sec, upper=post_thresh_sec, frequency=downsample_freq
                )
            tp_list.append(true_positives)
            fp_list.append(false_positives)
            hours_list.append(hours)
            total_events_list.append(total_events)
            mps_list = []
            label_list = []
    if sum(total_events_list) > 0:
        sensitivity = sum(tp_list)/sum(total_events_list)
    else:
        sensitivity = 0.0

    if sum(hours_list) > 0:
        false_alarms_per_hour = sum(fp_list)/sum(hours_list)
    else:
        false_alarms_per_hour = 0.0
    # false_alarms_per_hour = sum(fp_list)/sum(hours_list) if sum(hours_list) > 0 else 0.0
    overview = {"# TP":sum(tp_list), "# FP": sum(fp_list), "# Total seizures":sum(total_events_list)}
    return sensitivity, false_alarms_per_hour, overview

def run_grid_search(param_grid: Dict[str, List[Any]],
                    target_function: Callable[..., Dict[str, Any]],
                    save_results: bool = False) -> List[Dict[str, Any]]:
    """
    Executes a grid search over all combinations of parameter values and optionally saves results to CSV.

    Args:
        param_grid (Dict[str, List[Any]]): Parameter grid with possible values for each parameter.
        target_function (Callable): The function to evaluate.
        save_results (bool): Whether to save results to CSV.

    Returns:
        List[Dict[str, Any]]: List of parameter combinations and their results.
    """
    # csv_path = "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/hp_tuning_mp_results.csv"
    excel_path = "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/hp_tuning_mp_results.xlsx"

    # os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    # file_exists = os.path.isfile(excel_path)

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))

    results = []

    for values in combinations:
        params = dict(zip(keys, values))
        print(f"Testing combination: {params}")
        try:
            # result = target_function(**params)
            sensitivity, false_alarms_per_hour, overview = target_function(**params)
            combined = {
                **params,
                "sensitivity": sensitivity,
                "false_alarms_per_hour": false_alarms_per_hour,
                "overview": overview
            }

            results.append(combined)

            # if save_results:
            #     write_mode = 'a' if file_exists else 'w'
            #     with open(csv_path, write_mode, newline='') as csvfile:
            #         writer = csv.DictWriter(csvfile, fieldnames=list(combined.keys()))
            #         if not file_exists:
            #             writer.writeheader()
            #             file_exists = True
            #         writer.writerow(combined)
            if save_results:
                df_row = pd.DataFrame([combined])

                if os.path.isfile(excel_path):
                    # Load existing Excel file and append new row
                    existing_df = pd.read_excel(excel_path)
                    df_combined = pd.concat([existing_df, df_row], ignore_index=True)
                else:
                    df_combined = df_row
                # Write the full DataFrame back to the file
                df_combined.to_excel(excel_path, index=False)


        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            continue

    return results

if __name__ == "__main__":
    # Example parameter grid

    downsample_freq = 8
    window_size_sec = 25
    parameter_grid: Dict[str, List[Any]] = {
        "amount_of_annomalies_per_record": [10, 20, 40, 80, 120, 160, 200, 240, 300, 500, 700, 1000],
        "amount_of_records": [559], # 2795 * 0.2 => 20% of samples
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [0, 1, 2, 3, 4, 6, 8, 10],
        "n_cons": [1, 3, 5, 10, 12, 16, 24, 32, 40, 50, 60, 80],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [60 * 5],
        "post_thresh_sec": [60 * 3],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"]
    }

        # Run grid search with saving enabled
    grid_search_results = run_grid_search(parameter_grid, produce_mp_results, save_results=True)
