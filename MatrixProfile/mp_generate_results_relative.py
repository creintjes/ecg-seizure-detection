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
from collections import defaultdict
from matrix_profile import MatrixProfile
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*numpy.core.numeric is deprecated.*")

# Determine project root as the parent of 'pa' folder (adjust according to your structure)
project_root = Path(__file__).resolve().parent.parent  # If g.py is in pa/test, parent.parent ist pa/

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from utils.metrics import compute_sensitivity_false_alarm_rate_timing_tolerance

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

def produce_mp_results(
    amount_of_annomalies_per_record: int,
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
    verbose: bool,
    anomaly_ratio: float = None
) -> tuple:
    """
    Processes matrix profiles and computes evaluation metrics.

    Args:
        amount_of_annomalies_per_record (int): Default number of anomalies per record if anomaly_ratio is not used.
        batch_size_load (int): Number of files to process before reporting.
        downsample_freq (int): Downsampling frequency.
        max_gap_annos_in_sec (int): Maximum allowed gap in seconds for anomalies to be grouped.
        n_cons (int): Minimum consecutive anomalies to be grouped.
        window_size_sec (int): Window size in seconds.
        pre_thresh_sec (int): Pre threshold in seconds.
        post_thresh_sec (int): Post threshold in seconds.
        DIR_preprocessed (str): Directory with preprocessed files.
        MPs_path (str): Path to MatrixProfile files.
        recording_list_excel (str): Excel file with list of recordings.
        verbose (bool): Whether to print verbose output.
        anomaly_ratio (float, optional): Ratio of anomalies to find based on MP length (e.g. 0.01 finds 1% anomalies). If None, use amount_of_annomalies_per_record.

    Returns:
        tuple: (loaded_recs, loaded_recs_resp, sensitivity, false_alarms_per_hour, resp_sensitivity, resp_false_alarms_per_hour, overview)
    """
    if not recording_list_excel:
        return 0, 0, 0.0, 0.0, 0.0, 0.0, {}

    df = pd.read_excel(recording_list_excel)
    recs = [tuple(sr.split("_")) for sr in df["subject_run"].tolist()]

    mps_list = []
    label_list = []

    resp_tp_list = []
    resp_fp_list = []
    resp_total_events_list = []
    resp_hours_list = []

    tp_list = []
    fp_list = []

    hours_list = []
    total_events_list = []

    grouped_recs = defaultdict(list)
    for subject, run in recs:
        grouped_recs[subject].append(run)
    grouped_recs = dict(grouped_recs)

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
            mp_path = os.path.join(MPs_path, mp_filename)
            preprocessed_path = os.path.join(DIR_preprocessed, mp_filename[3:-4] + "_preprocessed.pkl")
            try:
                with open(mp_path, "rb") as f:
                    mp_loaded = pickle.load(f)[:, 0].reshape(-1, 1)
                    mps_list.append(mp_loaded)
                with open(preprocessed_path, "rb") as g:
                    label_list.append(pickle.load(g)["channels"][0]["labels"][0])
            except Exception as e:
                if verbose:
                    print(f"Error loading {mp_filename}: {e}")
                mps_list = []
                label_list = []
                continue

            mp_length = len(mp_loaded)
            if anomaly_ratio is not None:
                k = max(1, int(mp_length * anomaly_ratio))
            else:
                k = amount_of_annomalies_per_record

            if mp_length < k:
                if verbose:
                    print(f"{subject=}, {run=}, {mp_length=}, {k=}")
                mps_list = []
                label_list = []
                continue

            anomaly_indices = [
                MatrixProfile.get_top_k_anomaly_indices(matrix_profile=mp.flatten(), k=k)
                for mp in mps_list
            ]
            anomaly_indices_cons = [
                MatrixProfile.mean_of_all_consecutive_anomalies(
                    indices=annos, n=n_cons, max_gap=downsample_freq * max_gap_annos_in_sec
                )
                for annos in anomaly_indices
            ]
            suffix = ""
            if (pre_thresh_sec and pre_thresh_sec > 0) or (post_thresh_sec and post_thresh_sec > 0):
                suffix = "_detection_window"
            rf_xlsx_path = f"/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/rf_train_data/rf_train_samples{suffix}.xlsx"

            if False: #Set to True to create RF samples. NOTE: This can create a very large file - do not use in grid search
                append_rf_samples_to_xlsx(
                    mp=mp_loaded,
                    labels=label_list[0],
                    anomaly_indices=anomaly_indices_cons[0],
                    subject=subject,
                    run=run,
                    downsample_freq=downsample_freq,
                    out_path=rf_xlsx_path
                )


            true_positives, false_positives, hours, total_events = compute_sensitivity_false_alarm_rate_timing_tolerance(
                label_sequences=label_list, detection_indices=anomaly_indices_cons,
                lower=pre_thresh_sec, upper=post_thresh_sec, frequency=downsample_freq
            )
            true_positives_sub.append(true_positives)
            false_positives_sub.append(false_positives)
            hours_sub.append(hours)
            total_events_sub.append(total_events)
            loaded_recs += 1
            loaded_recs_resp_sub += 1

            if loaded_recs % batch_size_load == 0 and verbose:
                print(loaded_recs)
                if not len(label_list) == len(mps_list):
                    print(f"len(label_list) not == len(mps_list)")

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

    overview = {"# TP": sum(tp_list), "# FP": sum(fp_list), "# Total seizures": sum(total_events_list)}
    return loaded_recs, loaded_recs_resp, sensitivity, false_alarms_per_hour, resp_sensitivity, resp_false_alarms_per_hour, overview

import pickle
import numpy as np
from typing import List

import pandas as pd
import numpy as np
import os
from typing import List

def append_rf_samples_to_xlsx(
    mp: np.ndarray,
    labels: np.ndarray,
    anomaly_indices: List[int],
    subject: str,
    run: str,
    downsample_freq: int,
    out_path: str
) -> None:
    """
    Appends all context window samples for RF to a shared Excel file (row-wise, each sample one row).

    Args:
        mp (np.ndarray): Univariate MatrixProfile (1D or 2D).
        labels (np.ndarray): Label sequence (1D).
        anomaly_indices (List[int]): List of anomaly indices for this record.
        subject (str): Subject identifier.
        run (str): Run identifier.
        downsample_freq (int): Sampling rate in Hz.
        out_path (str): Output .xlsx path.
    """
    window_sec = 30  # Hardcoded window in seconds!
    half_window = window_sec * downsample_freq
    mp = mp.flatten()  # ensure 1D

    records = []
    for idx in anomaly_indices:
        start = idx - half_window
        end = idx + half_window + 1
        if start < 0 or end > len(mp):
            continue
        window = mp[start:end]
        if len(window) != (2 * half_window + 1):
            continue
        label = int(labels[idx] == 1) if idx < len(labels) else 0
        record = {
            'subject': subject,
            'run': run,
            'label': label
        }
        for j, v in enumerate(window):
            record[f'mp_{j}'] = float(v)
        records.append(record)

    if not records:
        print(f"No valid RF samples for {subject}_{run}")
        return

    df = pd.DataFrame(records)

    # Append to existing Excel or create new
    if os.path.isfile(out_path):
        # Read existing, append, and overwrite (sicher, weil append bei .xlsx nicht nativ wie bei CSV)
        df_existing = pd.read_excel(out_path)
        df_out = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_out = df

    df_out.to_excel(out_path, index=False)
    print(f"Appended {len(df)} samples from {subject}_{run} to {out_path}.")

def run_grid_search(
    param_grid: Dict[str, List[Any]],
    target_function: Callable[..., Any],
    val_excel_path: str,
    test_excel_path: str,
    save_results: bool = False
) -> None:
    """
    Runs a grid search on the target function.

    Args:
        param_grid (Dict[str, List[Any]]): Dictionary of parameters to try.
        target_function (Callable): Function to evaluate.
        val_excel_path (str): Path to validation file list.
        test_excel_path (str): Path to test file list.
        save_results (bool): Whether to save results to file.
    """
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
                detection_window_used = (
                    "pre_thresh_sec" in params and params["pre_thresh_sec"] > 0
                ) or (
                    "post_thresh_sec" in params and params["post_thresh_sec"] > 0
                )
                excel_suffix = "_detection_window" if detection_window_used else ""
                excel_path = f"/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/results/hp_tuning_mp_results_resp{excel_suffix}_relative.xlsx"

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
    downsample_freq = 8
    window_size_sec = 25

    val_excel_path = "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/configs/splits/by_subject_range_val_files_2042.xlsx"
    test_excel_path = "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/configs/splits/by_subject_range_test_files_753.xlsx"

    # Try different anomaly ratios: 1%, 2%, 5%
    parameter_grid_relative = {
        "amount_of_annomalies_per_record": [1500],  # Legacy fallback, will be ignored if anomaly_ratio is set
        "anomaly_ratio": [0.01, 0.02, 0.04, 0.06],        # Relative anomaly ratios
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [0.2, 1, 5, 10, 20, 30],
        "n_cons": [1, 3, 5, 10, 35],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [0],
        "post_thresh_sec": [0],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"]
    }
    parameter_grid_detection_window_relative = {
        "amount_of_annomalies_per_record": [1500],  # Legacy fallback
        "anomaly_ratio": [0.01, 0.02, 0.04, 0.06],
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [0.2, 1, 5, 10, 20, 30],
        "n_cons": [1, 3, 5, 10, 35],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [60 * 5],
        "post_thresh_sec": [60 * 3],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"]
    }

    from concurrent.futures import ThreadPoolExecutor
    # Run both grid searches in parallel threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_grid_search,
                parameter_grid_relative,
                produce_mp_results,
                val_excel_path,
                test_excel_path,
                True,
            ),
            executor.submit(
                run_grid_search,
                parameter_grid_detection_window_relative,
                produce_mp_results,
                val_excel_path,
                test_excel_path,
                True,
            ),
        ]

        for f in futures:
            f.result()

