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
from collections import defaultdict

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

def produce_mp_results(amount_of_annomalies_per_record: int, 
                       batch_size_load: int,
                       downsample_freq: int, 
                       max_gap_annos_in_sec: int, 
                       n_cons: int, 
                       window_size_sec: int, 
                       pre_thresh_sec: int, 
                       post_thresh_sec: int, 
                       DIR_preprocessed: str, 
                       MPs_path: str, 
                       recording_list_excel: str,
                       verbose: bool):
    if not recording_list_excel:
        return 0, 0, 0.0, 0.0, 0.0, 0.0, {}

    df = pd.read_excel(recording_list_excel)
    recs = [tuple(sr.split("_")) for sr in df["subject_run"].tolist()]

    mps_list = []
    label_list = []

    # True positives for responder
    resp_tp_list = []
    resp_fp_list = []
    resp_total_events_list = []
    resp_hours_list = []

    # True positives for both
    tp_list = []
    fp_list = []

    hours_list = []
    total_events_list = []

    # Create a dictionary where each subject is a key and values are lists of runs
    grouped_recs = defaultdict(list)

    # Populate the dictionary
    for subject, run in recs:
        grouped_recs[subject].append(run)

    grouped_recs = dict(grouped_recs)

    # Iterate over grouped_recs in insertion order
    loaded_recs = 0
    loaded_recs_resp = 0

    for subject, runs in grouped_recs.items():
        true_positives_sub = []
        false_positives_sub = []
        hours_sub = []
        total_events_sub = []
        loaded_recs_resp_sub = 0

        for run in runs:
            mp_filename = f"mp_{subject}_{run}.pkl"
            with open(os.path.join(MPs_path, mp_filename), "rb") as f:
                mps_list.append(pickle.load(f)[:, 0].reshape(-1, 1))
            with open(os.path.join(DIR_preprocessed, mp_filename[3:-4]+"_preprocessed.pkl"), "rb") as g:
                label_list.append(pickle.load(g)["channels"][0]["labels"][0])
            if len(mps_list[0]) < amount_of_annomalies_per_record:
                print(f"{subject=}, {run=}, {len(mps_list[0])=}, {amount_of_annomalies_per_record}")
                mps_list = []
                label_list = []
                continue
            
            annomaly_indices = [MatrixProfile.get_top_k_anomaly_indices(matrix_profile=mp.flatten(), k=amount_of_annomalies_per_record)
                        for mp in mps_list]
            annomaly_indices_cons = [MatrixProfile.mean_of_all_consecutive_anomalies(indices=annos, n=n_cons, max_gap = downsample_freq*max_gap_annos_in_sec)
                            for annos in annomaly_indices]
            true_positives, false_positives, hours, total_events = compute_sensitivity_false_alarm_rate_timing_tolerance(
                label_sequences=label_list, detection_indices=annomaly_indices_cons, lower=pre_thresh_sec, upper=post_thresh_sec, frequency=downsample_freq
                )
            true_positives_sub.append(true_positives)
            false_positives_sub.append(false_positives)
            hours_sub.append(hours)
            total_events_sub.append(total_events)
            loaded_recs += 1
            loaded_recs_resp_sub += 1
            if loaded_recs%batch_size_load ==0:
                if verbose: 
                    print(loaded_recs)
                if not len(label_list)==len(mps_list):
                    print(f"len(label_list) not ==len(mps_list)")
            mps_list = []
            label_list = []
            
        responder_rate = 0.0 if sum(total_events_sub) == 0 else sum(true_positives_sub) / sum(total_events_sub)
        if responder_rate >= 0.66:
            resp_tp_list.append(sum(true_positives_sub))
            resp_fp_list.append(sum(false_positives_sub))
            resp_total_events_list.append(sum(total_events_sub))
            resp_hours_list.append(sum(hours_sub))
            loaded_recs_resp += loaded_recs_resp_sub
        
        tp_list.append(sum(true_positives_sub))
        fp_list.append(sum(false_positives_sub))
        hours_list.append(sum(hours_sub))
        total_events_list.append(sum(total_events_sub))

    sensitivity = 0.0 if sum(total_events_list) == 0 else sum(tp_list) / sum(total_events_list)
    resp_sensitivity = 0.0 if sum(resp_total_events_list) == 0 else sum(resp_tp_list) / sum(resp_total_events_list)
    resp_false_alarms_per_hour = 0.0 if sum(resp_hours_list) == 0 else sum(resp_fp_list) / sum(resp_hours_list)
    false_alarms_per_hour = 0.0 if sum(hours_list) == 0 else sum(fp_list) / sum(hours_list)

    false_alarms_per_hour = sum(fp_list)/sum(hours_list) if sum(hours_list) > 0 else 0.0
    overview = {"# TP":sum(tp_list), "# FP": sum(fp_list), "# Total seizures":sum(total_events_list)}
    return loaded_recs, loaded_recs_resp, sensitivity, false_alarms_per_hour, resp_sensitivity, resp_false_alarms_per_hour, overview

def run_grid_search(param_grid: Dict[str, List[Any]],
                    target_function: Callable[..., Dict[str, Any]],
                    val_excel_path: str,
                    test_excel_path: str,
                    save_results: bool = False) -> None:

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))

    for values in combinations:
        params = dict(zip(keys, values))
        print(f"Testing combination: {params}")
        try:
            test_results = target_function(**params, recording_list_excel=test_excel_path)

            val_results = target_function(**params, recording_list_excel=val_excel_path)

            combined = {
                **params,
                # Test-metrics
                "test_loaded_recs": test_results[0],
                "test_loaded_recs_resp": test_results[1],
                "test_sensitivity": test_results[2],
                "test_false_alarms_per_hour": test_results[3],
                "test_resp_sensitivity": test_results[4],
                "test_resp_false_alarms_per_hour": test_results[5],
                "test_overview": test_results[6],
                # Val-metrics
                "val_loaded_recs": val_results[0],
                "val_loaded_recs_resp": val_results[1],
                "val_sensitivity": val_results[2],
                "val_false_alarms_per_hour": val_results[3],
                "val_resp_sensitivity": val_results[4],
                "val_resp_false_alarms_per_hour": val_results[5],
                "val_overview": val_results[6]
            }

            if save_results:
                # Dynamischer Pfad
                detection_window_used = (
                    "pre_thresh_sec" in params and params["pre_thresh_sec"] > 0
                ) or (
                    "post_thresh_sec" in params and params["post_thresh_sec"] > 0
                )
                excel_suffix = "_detection_window" if detection_window_used else ""
                excel_path = f"/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/results/hp_tuning_mp_results_resp{excel_suffix}.xlsx"

                df_row = pd.DataFrame([combined])
                if os.path.isfile(excel_path):
                    existing_df = pd.read_excel(excel_path)
                    df_combined = pd.concat([existing_df, df_row], ignore_index=True)
                else:
                    df_combined = df_row

                df_combined.to_excel(excel_path, index=False)

        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            continue



if __name__ == "__main__":
    # Example parameter grid

    downsample_freq = 8
    window_size_sec = 25

    val_excel_path = "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/configs/splits/val_filelist_4.xlsx"
    test_excel_path = "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/configs/splits/test_filelist_6.xlsx"
    parameter_grid: Dict[str, List[Any]] = {
        "amount_of_annomalies_per_record": [1500, ],
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [1,],
        "n_cons": [1,],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [0],
        "post_thresh_sec": [0],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"]
    }
    parameter_grid_detection_window: Dict[str, List[Any]] = {
        "amount_of_annomalies_per_record": [1500, ],
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [1,],
        "n_cons": [1, ],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [60 * 5],
        "post_thresh_sec": [60 * 3],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"]
    }

    # Run grid search with saving enabled
    run_grid_search(
        param_grid=parameter_grid,
        target_function=produce_mp_results,
        val_excel_path=val_excel_path,
        test_excel_path=test_excel_path,
        save_results=True
    )
    run_grid_search(
        param_grid=parameter_grid_detection_window,
        target_function=produce_mp_results,
        val_excel_path=val_excel_path,
        test_excel_path=test_excel_path,
        save_results=True
    )
