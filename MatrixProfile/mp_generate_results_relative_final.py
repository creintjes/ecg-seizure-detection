from pathlib import Path
from typing import List
import pickle
import re
import os
import itertools
import csv
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
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

from typing import List, Tuple, Dict, Any, Optional
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

def extract_seizures_from_labels(labels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract contiguous seizure intervals from a binary label sequence.

    Args:
        labels (np.ndarray): 1D array of labels where seizure samples are marked as 1 (or truthy).

    Returns:
        List[Tuple[int, int]]: List of (start_index, end_index) pairs, inclusive start and inclusive end.
    """
    labels = np.asarray(labels).flatten().astype(int)
    seizures: List[Tuple[int, int]] = []
    if labels.size == 0:
        return seizures

    in_seizure = False
    start_idx = 0
    for i, val in enumerate(labels):
        if not in_seizure and val == 1:
            in_seizure = True
            start_idx = i
        elif in_seizure and val == 0:
            # close current seizure at i-1 (inclusive)
            seizures.append((start_idx, i - 1))
            in_seizure = False
    # if sequence ends while inside seizure, close it
    if in_seizure:
        seizures.append((start_idx, len(labels) - 1))
    return seizures

def write_per_seizure_results(rows: List[Dict[str, Any]], out_path: str) -> None:
    """
    Append per-seizure detection rows to an Excel file. If file exists, rows are appended; otherwise a new file is created.

    Args:
        rows (List[Dict[str, Any]]): List of dict rows with keys matching columns, e.g. sub, run, seizure_index, detected.
        out_path (str): Path to output Excel file.
    """
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if os.path.isfile(out_path):
        try:
            df_existing = pd.read_excel(out_path)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            # If reading fails for any reason, overwrite with new
            df_out = df_new
    else:
        df_out = df_new
    df_out.to_excel(out_path, index=False)


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
    anomaly_ratio: Optional[float] = None,
    responders: Optional[List[str]] = None,
    per_seizure_outfile: Optional[str] = None
) -> tuple:
    """
    Replacement produce_mp_results that additionally evaluates each seizure separately and writes per-seizure
    detection results to an Excel file with columns: sub, run, seizure_index, detected (boolean).

    The function is intended to replace the existing produce_mp_results implementation. It reuses existing helpers:
    - MatrixProfile.get_top_k_anomaly_indices
    - MatrixProfile.mean_of_all_consecutive_anomalies
    - compute_sensitivity_false_alarm_rate_timing_tolerance

    Args:
        amount_of_annomalies_per_record (int): Default number of anomalies per record if anomaly_ratio is not used.
        batch_size_load (int): Number of files to process before reporting.
        downsample_freq (int): Downsampling frequency (samples/second).
        max_gap_annos_in_sec (int): Maximum allowed gap in seconds for anomalies to be grouped.
        n_cons (int): Minimum consecutive anomalies to be grouped.
        window_size_sec (int): Window size in seconds.
        pre_thresh_sec (int): Pre detection threshold in seconds.
        post_thresh_sec (int): Post detection threshold in seconds.
        DIR_preprocessed (str): Directory with preprocessed files (contains label pickles).
        MPs_path (str): Path to MatrixProfile files.
        recording_list_excel (str): Excel file with list of recordings (column "subject_run" expected).
        verbose (bool): Whether to print verbose output.
        anomaly_ratio (Optional[float]): Ratio of anomalies to find based on MP length (e.g. 0.01 finds 1% anomalies).
        responders (Optional[List[str]]): List of responders (subjects) for responder metrics.
        per_seizure_outfile (Optional[str]): Optional path for per-seizure Excel output. If None, defaults to MPs_path/per_seizure_detection_results.xlsx
    Returns:
        tuple: (loaded_recs, loaded_recs_resp, sensitivity, false_alarms_per_hour, resp_sensitivity, resp_false_alarms_per_hour, overview)
    """
    subject_metric_rows = []

    # basic validation
    if not recording_list_excel:
        return 0, 0, 0.0, 0.0, 0.0, 0.0, {}

    if per_seizure_outfile is None:
        per_seizure_outfile = os.path.join(MPs_path if MPs_path else ".", "per_seizure_detection_results.xlsx")

    # read recording list; expects column "subject_run" formatted "sub_run"
    df = pd.read_excel(recording_list_excel)
    recs = [tuple(sr.split("_")) for sr in df["subject_run"].tolist()]

    # prepare accumulators for aggregated metrics
    tp_list = []
    fp_list = []
    hours_list = []
    total_events_list = []

    resp_tp_list = []
    resp_fp_list = []
    resp_total_events_list = []
    resp_hours_list = []

    grouped_recs = defaultdict(list)
    for subject, run in recs:
        grouped_recs[subject].append(run)
    grouped_recs = dict(grouped_recs)

    loaded_recs = 0
    loaded_recs_resp = 0

    per_seizure_rows_buffer: List[Dict[str, Any]] = []

    # process each subject/run
    for subject, runs in grouped_recs.items():
        true_positives_sub = []
        false_positives_sub = []
        hours_sub = []
        total_events_sub = []

        for run in runs:
            mp_filename = f"mp_{subject}_{run}.pkl"
            mp_path = os.path.join(MPs_path, mp_filename)
            preprocessed_path = os.path.join(DIR_preprocessed, mp_filename[3:-4] + "_preprocessed.pkl")

            try:
                with open(mp_path, "rb") as f:
                    mp_loaded = pickle.load(f)[:, 0].reshape(-1, 1)
                with open(preprocessed_path, "rb") as g:
                    preproc = pickle.load(g)
                    # assume labels located at preproc["channels"][0]["labels"][0] as in previous code
                    labels = preproc["channels"][0]["labels"][0]
            except Exception as e:
                if verbose:
                    print(f"Skipping {subject}_{run} due to load error: {e}")
                continue

            mp_length = len(mp_loaded)
            if anomaly_ratio is not None:
                k = max(1, int(mp_length * anomaly_ratio))
            else:
                k = amount_of_annomalies_per_record

            if mp_length < k:
                if verbose:
                    print(f"Skipping {subject}_{run}: mp_length {mp_length} < k {k}")
                continue

            # get anomaly indices and aggregate consecutive detections
            anomaly_indices = MatrixProfile.get_top_k_anomaly_indices(matrix_profile=mp_loaded.flatten(), k=k)
            anomaly_indices_cons = MatrixProfile.mean_of_all_consecutive_anomalies(
                indices=anomaly_indices, n=n_cons, max_gap=downsample_freq * max_gap_annos_in_sec
            )

            # compute overall metrics using existing helper
            true_positives, false_positives, hours, total_events = compute_sensitivity_false_alarm_rate_timing_tolerance(
                label_sequences=[labels],
                detection_indices=[anomaly_indices_cons],
                lower=pre_thresh_sec,
                upper=post_thresh_sec,
                frequency=downsample_freq
            )

            true_positives_sub.append(true_positives)
            false_positives_sub.append(false_positives)
            hours_sub.append(hours)
            total_events_sub.append(total_events)
            loaded_recs += 1

            # --- per-seizure evaluation ---
            seizures = extract_seizures_from_labels(np.asarray(labels))
            pre_thresh_samples = int(pre_thresh_sec * downsample_freq) if pre_thresh_sec is not None else 0
            post_thresh_samples = int(post_thresh_sec * downsample_freq) if post_thresh_sec is not None else 0

            detection_indices_array = np.asarray(anomaly_indices_cons).flatten() if anomaly_indices_cons is not None else np.array([], dtype=int)

            for s_idx, (s_start, s_end) in enumerate(seizures, start=1):
                window_start = max(0, s_start - pre_thresh_samples)
                window_end = min(len(labels) - 1, s_end + post_thresh_samples)
                detected = bool(np.any((detection_indices_array >= window_start) & (detection_indices_array <= window_end)))
                per_seizure_rows_buffer.append({
                    "sub": subject,
                    "run": run,
                    "seizure_index": s_idx,
                    "detected": bool(detected)
                })

            try:
                write_per_seizure_results(per_seizure_rows_buffer, per_seizure_outfile)
                per_seizure_rows_buffer = []
            except Exception as e:
                if verbose:
                    print(f"Failed to write per-seizure results for buffer: {e}")

            if loaded_recs % batch_size_load == 0 and verbose:
                print(f"Processed {loaded_recs} records (last: {subject}_{run})")

        # compute per-subject metrics
        tp_sub = sum(true_positives_sub)
        fp_sub = sum(false_positives_sub)
        hours_sub_total = sum(hours_sub)
        events_sub = sum(total_events_sub)

        sens_sub = 0.0 if events_sub == 0 else tp_sub / events_sub
        far_sub = 0.0 if hours_sub_total == 0 else fp_sub / hours_sub_total

        subject_metric_rows.append({
            "subject": subject,
            "sensitivity": sens_sub,
            "false_alarms_per_hour": far_sub
        })



        # subject-level aggregation for responders if needed
        if responders and subject in responders:
            resp_tp_list.append(sum(true_positives_sub))
            resp_fp_list.append(sum(false_positives_sub))
            resp_total_events_list.append(sum(total_events_sub))
            resp_hours_list.append(sum(hours_sub))
            loaded_recs_resp += sum(total_events_sub)

        tp_list.append(sum(true_positives_sub))
        fp_list.append(sum(false_positives_sub))
        hours_list.append(sum(hours_sub))
        total_events_list.append(sum(total_events_sub))

    # final flush of per-seizure buffer
    if per_seizure_rows_buffer:
        try:
            write_per_seizure_results(per_seizure_rows_buffer, per_seizure_outfile)
        except Exception as e:
            if verbose:
                print(f"Failed final write of per-seizure results: {e}")

    # compute aggregated metrics
    sensitivity = 0.0 if sum(total_events_list) == 0 else sum(tp_list) / sum(total_events_list)
    resp_sensitivity = 0.0 if sum(resp_total_events_list) == 0 else sum(resp_tp_list) / sum(resp_total_events_list)
    resp_false_alarms_per_hour = 0.0 if sum(resp_hours_list) == 0 else sum(resp_fp_list) / sum(resp_hours_list)
    false_alarms_per_hour = 0.0 if sum(hours_list) == 0 else sum(fp_list) / sum(hours_list)

    overview = {"# TP": int(sum(tp_list)), "# FP": int(sum(fp_list)), "# Total seizures": int(sum(total_events_list))}
    df_subjects = pd.DataFrame(subject_metric_rows)
    df_subjects.to_excel(os.path.join(f"subject_level_metrics{per_seizure_outfile[16:-5]}.xlsx"), index=False)
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
    test_excel_path: str,
    save_results: bool = False,
    responders: List[str] = None
) -> None:
    """
    Runs a grid search on the target function.

    Args:
        param_grid (Dict[str, List[Any]]): Dictionary of parameters to try.
        target_function (Callable): Function to evaluate.
        test_excel_path (str): Path to test file list.
        save_results (bool): Whether to save results to file.
    """
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))

    for values in combinations:
        params = dict(zip(keys, values))
        print(f"Testing combination: {params}")
        try:
            test_results = target_function(**params, recording_list_excel=test_excel_path, responders=responders)

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
            }

            if save_results:
                detection_window_used = (
                    "pre_thresh_sec" in params and params["pre_thresh_sec"] > 0
                ) or (
                    "post_thresh_sec" in params and params["post_thresh_sec"] > 0
                )
                excel_suffix = "_detection_window" if detection_window_used else ""
                excel_path = f"/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/results/results_final_final_evaluation/hp_tuning_mp_results_resp{excel_suffix}_relative.xlsx"

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

    # Note: Amount of files is lower than 753 due to manual exclusion of files as explained in the paper. Naming has keept the same for consistency. 
    test_excel_path = "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/configs/splits/evaluation/by_subject_range_test_files_753.xlsx"

    # FAR & no SDW:
    parameter_grid_relative_1 = {
        "amount_of_annomalies_per_record": [1500],  # Legacy fallback, will be ignored if anomaly_ratio is set
        "anomaly_ratio": [0.01,],        # Relative anomaly ratios
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [30],
        "n_cons": [35],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [0],
        "post_thresh_sec": [0],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"],
        "per_seizure_outfile" : ["seizure_analysis_FAR & no SDW.xlsx"]

    }
    # Sensitivity & no SDW:
    parameter_grid_relative_2 = {
        "amount_of_annomalies_per_record": [1500],  # Legacy fallback, will be ignored if anomaly_ratio is set
        "anomaly_ratio": [0.06,],        # Relative anomaly ratios
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [0.2],
        "n_cons": [1],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [0],
        "post_thresh_sec": [0],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"],
        "per_seizure_outfile" : ["seizure_analysis_Sensitivity & no SDW.xlsx"]

    }
    
    # HMC & no SDW:
    parameter_grid_relative_3 = {
        "amount_of_annomalies_per_record": [1500],  # Legacy fallback, will be ignored if anomaly_ratio is set
        "anomaly_ratio": [0.06,],        # Relative anomaly ratios
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [5],
        "n_cons": [1],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [0],
        "post_thresh_sec": [0],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"],
        "per_seizure_outfile" : ["seizure_analysis_HMC & no SDW.xlsx"]
    }
    
    # FAR & SDW:
    
    parameter_grid_detection_window_relative_1 = {
        "amount_of_annomalies_per_record": [1500],  # Legacy fallback
        "anomaly_ratio": [0.01,],
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [30,],
        "n_cons": [35],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [60 * 5],
        "post_thresh_sec": [60 * 3],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"],
        "per_seizure_outfile" : ["seizure_analysis_FAR & SDW.xlsx"]

    }

    
    # Sensitivity & SDW:
    parameter_grid_detection_window_relative_2 = {
        "amount_of_annomalies_per_record": [1500],  # Legacy fallback
        "anomaly_ratio": [0.06,],
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [20,],
        "n_cons": [1],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [60 * 5],
        "post_thresh_sec": [60 * 3],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"],
        "per_seizure_outfile" : ["seizure_analysis_Sensitivity & SDW.xlsx"]
    }
    
    # HMC & SDW:
    parameter_grid_detection_window_relative_3 = {
        "amount_of_annomalies_per_record": [1500],  # Legacy fallback
        "anomaly_ratio": [0.06,],
        "batch_size_load": [100],
        "downsample_freq": [downsample_freq],
        "max_gap_annos_in_sec": [30,],
        "n_cons": [1],
        "window_size_sec": [window_size_sec],
        "pre_thresh_sec": [60 * 5],
        "post_thresh_sec": [60 * 3],
        "verbose": [False],
        "DIR_preprocessed": [f"/home/swolf/asim_shared/preprocessed_data/downsample_freq={downsample_freq},no_windows"],
        "MPs_path": [f"/home/swolf/asim_shared/results/MP/downsample_freq={downsample_freq},no_windows/seq_len{window_size_sec}sec"],
        "per_seizure_outfile" : ["seizure_analysis_HMC & SDW.xlsx"]
    }

    from concurrent.futures import ThreadPoolExecutor
    # Run both grid searches in parallel threads

    responders = ['sub-098', 'sub-100', 'sub-105', 'sub-106', 'sub-107', 'sub-110', 'sub-111', 'sub-114', 
        'sub-115', 'sub-116', 'sub-118', 'sub-122', 'sub-123', 'sub-124', 'sub-125']

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_grid_search,
                parameter_grid_relative_1,
                produce_mp_results,
                test_excel_path,
                True,
                responders=responders,
            ),
            executor.submit(
                run_grid_search,
                parameter_grid_detection_window_relative_1,
                produce_mp_results,
                test_excel_path,
                True,
                responders=responders,
            ),
        ]

        for f in futures:
            f.result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_grid_search,
                parameter_grid_relative_2,
                produce_mp_results,
                test_excel_path,
                True,
                responders=responders,
            ),
            executor.submit(
                run_grid_search,
                parameter_grid_detection_window_relative_2,
                produce_mp_results,
                test_excel_path,
                True,
                responders=responders,
            ),
        ]

        for f in futures:
            f.result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_grid_search,
                parameter_grid_relative_3,
                produce_mp_results,
                test_excel_path,
                True,
                responders=responders,
            ),
            executor.submit(
                run_grid_search,
                parameter_grid_detection_window_relative_3,
                produce_mp_results,
                test_excel_path,
                True,
                responders=responders,
            ),
        ]

        for f in futures:
            f.result()