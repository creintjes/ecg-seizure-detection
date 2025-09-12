import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd

def extract_subject_run_pairs(mps_path: str) -> List[str]:
    """
    Extract subject_run strings from MP filenames using same logic as produce_mp_results.

    Args:
        mps_path (str): Path where MP files are stored.

    Returns:
        List[str]: List of subject_run identifiers, e.g., "sub-119_run-34"
    """
    mps_filenames: List[str] = [
        f for f in os.listdir(mps_path)
        if f.endswith(".pkl") and f.startswith("mp_")
    ]
    # remove prefix "mp_" and suffix ".pkl", e.g. mp_sub-119_run-34.pkl -> sub-119_run-34
    subject_run_pairs: List[str] = [filename[3:-4] for filename in mps_filenames]
    return subject_run_pairs


def _subject_str(n: int) -> str:
    """
    Create a zero-padded subject string like 'sub-001' from an integer.

    Args:
        n (int): Subject number.

    Returns:
        str: Subject label of form 'sub-XYZ'.
    """
    # Ensure 3-digit zero padding (001..999)
    return f"sub-{n:03d}"


def _parse_subject_and_run(sr: str) -> Tuple[str, int]:
    """
    Parse 'sub-XYZ_run-NN' into ('sub-XYZ', NN).

    Args:
        sr (str): Subject_run string, e.g. 'sub-119_run-34'.

    Returns:
        Tuple[str, int]: (subject_label, run_number)
    """
    # Robust regex for sub and run fields
    m = re.match(r"^(sub-\d{3})_run-(\d+)$", sr)
    if not m:
        raise ValueError(f"Invalid subject_run format: {sr}")
    subject_label: str = m.group(1)
    run_number: int = int(m.group(2))
    return subject_label, run_number

def create_val_test_split_by_subject_ranges(
    mps_path: str,
    output_dir: str,
    val_start: int = 1,
    val_end: int = 96,
    test_start: int = 97,
    test_end: int = 125,
    excel_prefix: str = ""
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a deterministic split into val and test based on subject ID ranges.

    The val set corresponds to subjects sub-001..sub-096 and the test set to
    sub-097..sub-125 by default (configurable via arguments). Only files present
    in `mps_path` are included; missing subjects are ignored with a warning.

    Args:
        mps_path (str): Directory containing MP files (named like 'mp_sub-XXX_run-YY.pkl').
        output_dir (str): Directory where the Excel files will be saved.
        val_start (int): First subject number (inclusive) for validation set. Default 1.
        val_end (int): Last subject number (inclusive) for validation set. Default 96.
        test_start (int): First subject number (inclusive) for test set. Default 97.
        test_end (int): Last subject number (inclusive) for test set. Default 125.
        excel_prefix (str): Optional prefix for output Excel filenames.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (val_df, test_df) with dtype 'string' for 'subject_run'.
    """
    # Collect all available subject_run pairs
    all_subject_runs: List[str] = extract_subject_run_pairs(mps_path)

    # Build subject label sets based on the requested ranges
    val_subjects = { _subject_str(i) for i in range(val_start, val_end + 1) }
    test_subjects = { _subject_str(i) for i in range(test_start, test_end + 1) }

    # Container for split results
    val_items: List[Tuple[str, str, int]] = []   # (subject_run, subject_label, run)
    test_items: List[Tuple[str, str, int]] = []

    # Split deterministically by subject label membership
    for sr in all_subject_runs:
        subj, run = _parse_subject_and_run(sr)
        if subj in val_subjects:
            val_items.append((sr, subj, run))
        elif subj in test_subjects:
            test_items.append((sr, subj, run))
        else:
            # Not in either requested range -> ignore
            continue

    # Sort by subject then run for reproducibility and readability
    val_items.sort(key=lambda x: (x[1], x[2]))
    test_items.sort(key=lambda x: (x[1], x[2]))

    # Create DataFrames with explicit dtype 'string'
    val_df: pd.DataFrame = pd.DataFrame(
        pd.Series([x[0] for x in val_items], name="subject_run", dtype="string")
    )
    test_df: pd.DataFrame = pd.DataFrame(
        pd.Series([x[0] for x in test_items], name="subject_run", dtype="string")
    )

    # Ensure output dir exists
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Compose filenames
    prefix = f"{excel_prefix}_" if excel_prefix else ""
    val_path = outdir / f"{prefix}val_files_{len(val_df)}.xlsx"
    test_path = outdir / f"{prefix}test_files_{len(test_df)}.xlsx"

    # Save to Excel
    # Note: index=False to avoid Excel index column
    val_df.to_excel(val_path, index=False)
    test_df.to_excel(test_path, index=False)

    # Simple reporting
    # (In case some subjects are missing entirely in the folder, this helps debugging.)
    available_subjects = { _parse_subject_and_run(sr)[0] for sr in all_subject_runs }
    missing_val = sorted(val_subjects - available_subjects)
    missing_test = sorted(test_subjects - available_subjects)
    if missing_val:
        print(f"[WARN] {len(missing_val)} val subjects not found in {mps_path}: {', '.join(missing_val[:10])}...")
    if missing_test:
        print(f"[WARN] {len(missing_test)} test subjects not found in {mps_path}: {', '.join(missing_test[:10])}...")

    print(f"Val list saved to: {val_path}  (n={len(val_df)})")
    print(f"Test list saved to: {test_path} (n={len(test_df)})")

    return val_df, test_df

if __name__ == "__main__":
    # Example usage with your paths
    create_val_test_split_by_subject_ranges(
        mps_path="/home/swolf/asim_shared/results/MP/downsample_freq=8,no_windows/seq_len25sec",
        output_dir="/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/configs/splits",
        # Defaults already match your requirement:
        # val:  sub-001..sub-096
        # test: sub-097..sub-125
        excel_prefix="by_subject_range"
    )
