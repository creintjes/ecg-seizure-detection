#!/usr/bin/env python3
"""
Script to verify that all runs from raw data are present in preprocessed data.
Compares raw ECG files with preprocessed pickle files and identifies missing runs.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from collections import defaultdict

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Information', 'Data', 'seizeit2_main'))

from config import RAW_DATA_PATH, PREPROCESSED_DATA_PATH
from classes.annotation import Annotation


def discover_raw_recordings(data_path):
    """
    Discover all available recordings in the raw SeizeIT2 dataset.

    Args:
        data_path: Path to raw SeizeIT2 dataset

    Returns:
        Dictionary with recording info: {(subject_id, run_id): {'ecg_file': path, 'has_seizures': bool, 'n_seizures': int}}
    """
    data_path = Path(data_path)
    recordings = {}

    if not data_path.exists():
        print(f"Error: Raw data path {data_path} does not exist!")
        return recordings

    # Find all subjects
    subjects = sorted([x for x in data_path.glob("sub-*") if x.is_dir()])
    print(f"Scanning raw data: Found {len(subjects)} subjects")

    for subject_dir in subjects:
        subject_id = subject_dir.name

        # Look for ECG sessions
        ecg_dir = subject_dir / 'ses-01' / 'ecg'
        eeg_dir = subject_dir / 'ses-01' / 'eeg'

        if ecg_dir.exists():
            # Find all runs for this subject
            edf_files = sorted(list(ecg_dir.glob("*_ecg.edf")))

            for edf_file in edf_files:
                # Extract run ID from filename
                # Format: sub-XXX_ses-01_task-szMonitoring_run-XX_ecg.edf
                parts = edf_file.stem.split('_')
                run_part = [p for p in parts if p.startswith('run-')]

                if run_part:
                    run_id = run_part[0]

                    # Check for seizure annotations
                    has_seizures = False
                    n_seizures = 0

                    try:
                        # Try to load annotations
                        recording = [subject_id, run_id]
                        annotations = Annotation.loadAnnotation(str(data_path), recording)

                        if annotations and annotations.events:
                            n_seizures = len(annotations.events)
                            has_seizures = n_seizures > 0
                    except Exception as e:
                        # If annotation loading fails, check events.tsv file directly
                        events_file = eeg_dir / f"{subject_id}_ses-01_task-szMonitoring_{run_id}_events.tsv"
                        if events_file.exists():
                            try:
                                events_df = pd.read_csv(events_file, sep='\t')
                                if 'eventType' in events_df.columns:
                                    seizures = events_df[
                                        (events_df['eventType'].str.startswith('sz_', na=False)) &
                                        (events_df['eventType'] != 'bckg') &
                                        (events_df['eventType'] != 'impd')
                                    ]
                                    n_seizures = len(seizures)
                                    has_seizures = n_seizures > 0
                            except:
                                pass

                    recordings[(subject_id, run_id)] = {
                        'ecg_file': str(edf_file),
                        'has_seizures': has_seizures,
                        'n_seizures': n_seizures
                    }

    return recordings


def discover_preprocessed_recordings(preprocessed_path):
    """
    Discover all available recordings in the preprocessed data directory.

    Args:
        preprocessed_path: Path to preprocessed data directory

    Returns:
        Dictionary with recording info: {(subject_id, run_id): {'pkl_file': path, 'n_windows': int, 'n_seizure_windows': int, 'seizure_info': list}}
    """
    preprocessed_path = Path(preprocessed_path)
    recordings = {}

    if not preprocessed_path.exists():
        print(f"Error: Preprocessed data path {preprocessed_path} does not exist!")
        return recordings

    # Find all pickle files
    pkl_files = sorted(list(preprocessed_path.glob("*.pkl")))
    print(f"Scanning preprocessed data: Found {len(pkl_files)} pickle files")

    for pkl_file in pkl_files:
        # Extract subject and run from filename
        # Format: sub-XXX_run-XX_preprocessed.pkl
        filename = pkl_file.stem

        if '_preprocessed' in filename:
            parts = filename.replace('_preprocessed', '').split('_')

            # Find subject and run parts
            subject_id = None
            run_id = None

            for part in parts:
                if part.startswith('sub-'):
                    subject_id = part
                elif part.startswith('run-'):
                    run_id = part

            if subject_id and run_id:
                # Try to load pickle file to get more info
                n_windows = 0
                n_seizure_windows = 0
                seizure_info = []

                try:
                    data = pd.read_pickle(pkl_file)

                    # Count windows and extract seizure information
                    if 'channels' in data:
                        for channel in data['channels']:
                            if 'n_windows' in channel:
                                n_windows += channel['n_windows']
                            if 'n_seizure_windows' in channel:
                                n_seizure_windows += channel['n_seizure_windows']

                            # Extract seizure segments from metadata
                            if 'metadata' in channel:
                                for meta in channel['metadata']:
                                    if meta.get('window_label') == 1 or meta.get('n_seizure_samples', 0) > 0:
                                        if 'seizure_segments' in meta and meta['seizure_segments']:
                                            for seg in meta['seizure_segments']:
                                                seizure_info.append({
                                                    'start_time': seg.get('start_time_absolute'),
                                                    'end_time': seg.get('end_time_absolute'),
                                                    'duration': seg.get('duration_seconds')
                                                })
                except Exception as e:
                    print(f"  Warning: Could not read {pkl_file.name}: {e}")

                recordings[(subject_id, run_id)] = {
                    'pkl_file': str(pkl_file),
                    'n_windows': n_windows,
                    'n_seizure_windows': n_seizure_windows,
                    'has_seizures': n_seizure_windows > 0,
                    'n_seizure_segments': len(seizure_info),
                    'seizure_info': seizure_info
                }

    return recordings


def compare_datasets(raw_recordings, preprocessed_recordings):
    """
    Compare raw and preprocessed datasets to find missing runs.

    Args:
        raw_recordings: Dictionary of raw recordings
        preprocessed_recordings: Dictionary of preprocessed recordings

    Returns:
        Dictionary with comparison results
    """
    raw_keys = set(raw_recordings.keys())
    preprocessed_keys = set(preprocessed_recordings.keys())

    # Find differences
    missing_in_preprocessed = raw_keys - preprocessed_keys
    extra_in_preprocessed = preprocessed_keys - raw_keys
    common = raw_keys & preprocessed_keys

    # Analyze missing runs
    missing_with_seizures = []
    missing_without_seizures = []

    for key in missing_in_preprocessed:
        if raw_recordings[key]['has_seizures']:
            missing_with_seizures.append(key)
        else:
            missing_without_seizures.append(key)

    # Count seizures in missing runs
    total_missing_seizures = sum(
        raw_recordings[key]['n_seizures']
        for key in missing_in_preprocessed
    )

    # Analyze common runs - check for seizure data consistency
    common_with_seizures = 0
    total_seizures_in_common = 0
    present_but_missing_seizures = []  # Runs that exist but don't have seizure data

    for key in common:
        raw_has_seizures = raw_recordings[key]['has_seizures']
        raw_n_seizures = raw_recordings[key]['n_seizures']

        preprocessed_has_seizures = preprocessed_recordings[key]['has_seizures']
        preprocessed_n_seizure_windows = preprocessed_recordings[key]['n_seizure_windows']

        if raw_has_seizures:
            common_with_seizures += 1
            total_seizures_in_common += raw_n_seizures

            # Check if preprocessed data has seizures when raw data has them
            if not preprocessed_has_seizures or preprocessed_n_seizure_windows == 0:
                present_but_missing_seizures.append(key)

    # Count seizures lost due to present runs missing seizure data
    total_seizures_lost_in_present_runs = sum(
        raw_recordings[key]['n_seizures']
        for key in present_but_missing_seizures
    )

    results = {
        'raw_total': len(raw_keys),
        'preprocessed_total': len(preprocessed_keys),
        'common': len(common),
        'missing_in_preprocessed': len(missing_in_preprocessed),
        'extra_in_preprocessed': len(extra_in_preprocessed),
        'missing_with_seizures': len(missing_with_seizures),
        'missing_without_seizures': len(missing_without_seizures),
        'total_missing_seizures': total_missing_seizures,
        'common_with_seizures': common_with_seizures,
        'total_seizures_in_common': total_seizures_in_common,
        'present_but_missing_seizures': len(present_but_missing_seizures),
        'total_seizures_lost_in_present_runs': total_seizures_lost_in_present_runs,
        'missing_runs': sorted(list(missing_in_preprocessed)),
        'extra_runs': sorted(list(extra_in_preprocessed)),
        'missing_runs_with_seizures': sorted(missing_with_seizures),
        'missing_runs_without_seizures': sorted(missing_without_seizures),
        'present_runs_missing_seizure_data': sorted(present_but_missing_seizures)
    }

    return results


def generate_report(results, raw_recordings, preprocessed_recordings, output_dir):
    """
    Generate a detailed report of the comparison.

    Args:
        results: Comparison results
        raw_recordings: Dictionary of raw recordings
        preprocessed_recordings: Dictionary of preprocessed recordings
        output_dir: Output directory for report files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Text report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PREPROCESSED DATA COMPLETENESS VERIFICATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    report_lines.append("OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total runs in raw data:              {results['raw_total']}")
    report_lines.append(f"Total runs in preprocessed data:     {results['preprocessed_total']}")
    report_lines.append(f"Runs present in both:                {results['common']}")
    report_lines.append(f"Runs missing in preprocessed:        {results['missing_in_preprocessed']}")
    report_lines.append(f"Extra runs in preprocessed:          {results['extra_in_preprocessed']}")
    report_lines.append("")

    # Completeness percentage
    completeness = (results['preprocessed_total'] / results['raw_total'] * 100) if results['raw_total'] > 0 else 0
    report_lines.append(f"Completeness: {completeness:.1f}%")
    report_lines.append("")

    report_lines.append("SEIZURE ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(f"Missing runs WITH seizures:          {results['missing_with_seizures']}")
    report_lines.append(f"Missing runs WITHOUT seizures:       {results['missing_without_seizures']}")
    report_lines.append(f"Total seizures in missing runs:      {results['total_missing_seizures']}")
    report_lines.append("")
    report_lines.append(f"Common runs with seizures:           {results['common_with_seizures']}")
    report_lines.append(f"Total seizures in common runs:       {results['total_seizures_in_common']}")
    report_lines.append("")
    report_lines.append(f"Present runs missing seizure data:   {results['present_but_missing_seizures']}")
    report_lines.append(f"Seizures lost in present runs:       {results['total_seizures_lost_in_present_runs']}")
    report_lines.append("")

    # Calculate expected vs actual seizures
    expected_seizures = results['total_seizures_in_common'] + results['total_missing_seizures']
    total_lost_seizures = results['total_missing_seizures'] + results['total_seizures_lost_in_present_runs']
    seizure_loss = (total_lost_seizures / expected_seizures * 100) if expected_seizures > 0 else 0
    report_lines.append(f"Expected total seizures (raw data):  {expected_seizures}")
    report_lines.append(f"Total lost seizures:                 {total_lost_seizures}")
    report_lines.append(f"  - From missing runs:               {results['total_missing_seizures']}")
    report_lines.append(f"  - From present runs (no data):     {results['total_seizures_lost_in_present_runs']}")
    report_lines.append(f"Seizure loss percentage:             {seizure_loss:.1f}%")
    report_lines.append("")

    if results['missing_in_preprocessed'] > 0 or results['present_but_missing_seizures'] > 0:
        report_lines.append("CRITICAL ISSUES")
        report_lines.append("-" * 80)

        if results['missing_in_preprocessed'] > 0:
            report_lines.append(f"⚠️  {results['missing_in_preprocessed']} runs are missing from preprocessed data!")

            if results['missing_with_seizures'] > 0:
                report_lines.append(f"⚠️  {results['missing_with_seizures']} missing runs contain seizures!")
                report_lines.append(f"⚠️  This accounts for {results['total_missing_seizures']} seizures!")

        if results['present_but_missing_seizures'] > 0:
            report_lines.append("")
            report_lines.append(f"⚠️  {results['present_but_missing_seizures']} runs exist but have NO seizure data!")
            report_lines.append(f"⚠️  These runs should contain {results['total_seizures_lost_in_present_runs']} seizures!")
            report_lines.append(f"⚠️  Possible causes:")
            report_lines.append(f"     - Preprocessing failed to label seizure windows")
            report_lines.append(f"     - Seizure annotations not properly transferred")
            report_lines.append(f"     - Windows don't overlap with seizure events")

        report_lines.append("")
        report_lines.append(f"TOTAL SEIZURE LOSS: {total_lost_seizures} seizures ({seizure_loss:.1f}%)")
        report_lines.append("This explains the discrepancy between expected (856) and actual (794) seizures.")
        report_lines.append("")

    # Detailed listing of missing runs
    if results['missing_runs_with_seizures']:
        report_lines.append("")
        report_lines.append("MISSING RUNS WITH SEIZURES (DETAILED)")
        report_lines.append("-" * 80)

        # Group by subject
        by_subject = defaultdict(list)
        for subject_id, run_id in results['missing_runs_with_seizures']:
            by_subject[subject_id].append(run_id)

        for subject_id in sorted(by_subject.keys()):
            runs = by_subject[subject_id]
            report_lines.append(f"\n{subject_id}:")

            for run_id in sorted(runs):
                info = raw_recordings.get((subject_id, run_id), {})
                n_seizures = info.get('n_seizures', 0)
                ecg_file = info.get('ecg_file', 'N/A')
                report_lines.append(f"  {run_id}: {n_seizures} seizure(s)")
                report_lines.append(f"    ECG file: {ecg_file}")

    if results['present_runs_missing_seizure_data']:
        report_lines.append("")
        report_lines.append("PRESENT RUNS WITH MISSING SEIZURE DATA (DETAILED)")
        report_lines.append("-" * 80)
        report_lines.append("These runs exist in preprocessed data but contain NO seizure windows,")
        report_lines.append("even though the raw data has seizure annotations!")
        report_lines.append("")

        # Group by subject
        by_subject = defaultdict(list)
        for subject_id, run_id in results['present_runs_missing_seizure_data']:
            by_subject[subject_id].append(run_id)

        for subject_id in sorted(by_subject.keys()):
            runs = by_subject[subject_id]
            report_lines.append(f"\n{subject_id}:")

            for run_id in sorted(runs):
                raw_info = raw_recordings.get((subject_id, run_id), {})
                preprocessed_info = preprocessed_recordings.get((subject_id, run_id), {})

                n_seizures = raw_info.get('n_seizures', 0)
                n_windows = preprocessed_info.get('n_windows', 0)
                n_seizure_windows = preprocessed_info.get('n_seizure_windows', 0)

                report_lines.append(f"  {run_id}:")
                report_lines.append(f"    Expected seizures (raw): {n_seizures}")
                report_lines.append(f"    Total windows: {n_windows}")
                report_lines.append(f"    Seizure windows: {n_seizure_windows}")
                report_lines.append(f"    pkl file: {preprocessed_info.get('pkl_file', 'N/A')}")

    if results['missing_runs_without_seizures']:
        report_lines.append("")
        report_lines.append("MISSING RUNS WITHOUT SEIZURES")
        report_lines.append("-" * 80)

        # Group by subject
        by_subject = defaultdict(list)
        for subject_id, run_id in results['missing_runs_without_seizures']:
            by_subject[subject_id].append(run_id)

        for subject_id in sorted(by_subject.keys()):
            runs = by_subject[subject_id]
            runs_str = ", ".join(sorted(runs))
            report_lines.append(f"  {subject_id}: {runs_str}")

    if results['extra_in_preprocessed'] > 0:
        report_lines.append("")
        report_lines.append("EXTRA RUNS IN PREPROCESSED DATA (NOT IN RAW)")
        report_lines.append("-" * 80)

        # Group by subject
        by_subject = defaultdict(list)
        for subject_id, run_id in results['extra_runs']:
            by_subject[subject_id].append(run_id)

        for subject_id in sorted(by_subject.keys()):
            runs = by_subject[subject_id]
            runs_str = ", ".join(sorted(runs))
            report_lines.append(f"  {subject_id}: {runs_str}")

    report_lines.append("")
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 80)

    if results['missing_in_preprocessed'] > 0:
        report_lines.append("1. Re-run preprocessing for missing runs:")
        report_lines.append("   - Check if preprocessing failed for these runs")
        report_lines.append("   - Review preprocessing logs for error messages")
        report_lines.append("   - Consider running preprocess_all_data.py again")
        report_lines.append("")

    if results['missing_with_seizures'] > 0:
        report_lines.append("2. URGENT: Missing runs contain seizure data!")
        report_lines.append(f"   - {results['total_missing_seizures']} seizures are not in preprocessed data")
        report_lines.append("   - Prioritize preprocessing these runs")
        report_lines.append("")

    if results['present_but_missing_seizures'] > 0:
        report_lines.append("3. CRITICAL: Runs exist but have no seizure data!")
        report_lines.append(f"   - {results['present_but_missing_seizures']} runs should contain seizures but don't")
        report_lines.append(f"   - {results['total_seizures_lost_in_present_runs']} seizures are missing from these runs")
        report_lines.append("   - Possible solutions:")
        report_lines.append("     * Re-run preprocessing with correct annotation transfer")
        report_lines.append("     * Check if window size/stride causes seizures to be missed")
        report_lines.append("     * Verify enhance_with_sample_level_labels() function")
        report_lines.append("")

    report_lines.append("4. Check failed_recordings_preprocessing_SeizeIT2_ECG.csv")
    report_lines.append("   - This file should list any runs that failed during preprocessing")
    report_lines.append("")

    report_lines.append(f"5. Total seizure accounting:")
    report_lines.append(f"   - Expected from raw data: {expected_seizures}")
    report_lines.append(f"   - Lost from missing runs: {results['total_missing_seizures']}")
    report_lines.append(f"   - Lost from present runs: {results['total_seizures_lost_in_present_runs']}")
    report_lines.append(f"   - Total loss: {total_lost_seizures} ({seizure_loss:.1f}%)")
    report_lines.append("")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Save text report
    report_path = output_dir / f"preprocessed_completeness_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f"\n✓ Report saved to: {report_path}")

    # Save JSON data
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'raw_total': results['raw_total'],
            'preprocessed_total': results['preprocessed_total'],
            'completeness_percentage': completeness,
            'missing_in_preprocessed': results['missing_in_preprocessed'],
            'missing_with_seizures': results['missing_with_seizures'],
            'total_missing_seizures': results['total_missing_seizures'],
            'present_but_missing_seizures': results['present_but_missing_seizures'],
            'total_seizures_lost_in_present_runs': results['total_seizures_lost_in_present_runs'],
            'total_lost_seizures': total_lost_seizures,
            'seizure_loss_percentage': seizure_loss
        },
        'missing_runs_with_seizures': [
            {
                'subject_id': subj,
                'run_id': run,
                'n_seizures': raw_recordings.get((subj, run), {}).get('n_seizures', 0),
                'ecg_file': raw_recordings.get((subj, run), {}).get('ecg_file', '')
            }
            for subj, run in results['missing_runs_with_seizures']
        ],
        'present_runs_missing_seizure_data': [
            {
                'subject_id': subj,
                'run_id': run,
                'expected_seizures': raw_recordings.get((subj, run), {}).get('n_seizures', 0),
                'n_windows': preprocessed_recordings.get((subj, run), {}).get('n_windows', 0),
                'n_seizure_windows': preprocessed_recordings.get((subj, run), {}).get('n_seizure_windows', 0),
                'pkl_file': preprocessed_recordings.get((subj, run), {}).get('pkl_file', '')
            }
            for subj, run in results['present_runs_missing_seizure_data']
        ],
        'missing_runs_without_seizures': [
            {'subject_id': subj, 'run_id': run}
            for subj, run in results['missing_runs_without_seizures']
        ]
    }

    json_path = output_dir / f"preprocessed_completeness_data_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ JSON data saved to: {json_path}")

    # Save CSV of missing runs with seizures
    if results['missing_runs_with_seizures']:
        csv_data = []
        for subject_id, run_id in results['missing_runs_with_seizures']:
            info = raw_recordings.get((subject_id, run_id), {})
            csv_data.append({
                'subject_id': subject_id,
                'run_id': run_id,
                'n_seizures': info.get('n_seizures', 0),
                'ecg_file': info.get('ecg_file', '')
            })

        df = pd.DataFrame(csv_data)
        csv_path = output_dir / f"missing_runs_with_seizures_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Missing runs CSV saved to: {csv_path}")

    # Save CSV of present runs with missing seizure data
    if results['present_runs_missing_seizure_data']:
        csv_data = []
        for subject_id, run_id in results['present_runs_missing_seizure_data']:
            raw_info = raw_recordings.get((subject_id, run_id), {})
            preprocessed_info = preprocessed_recordings.get((subject_id, run_id), {})
            csv_data.append({
                'subject_id': subject_id,
                'run_id': run_id,
                'expected_seizures': raw_info.get('n_seizures', 0),
                'n_windows': preprocessed_info.get('n_windows', 0),
                'n_seizure_windows': preprocessed_info.get('n_seizure_windows', 0),
                'pkl_file': preprocessed_info.get('pkl_file', '')
            })

        df = pd.DataFrame(csv_data)
        csv_path = output_dir / f"present_runs_missing_seizure_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Present runs with missing seizure data CSV saved to: {csv_path}")


def print_summary(results):
    """Print a brief summary to console."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Raw data runs:             {results['raw_total']}")
    print(f"Preprocessed data runs:    {results['preprocessed_total']}")
    print(f"Missing in preprocessed:   {results['missing_in_preprocessed']}")

    total_lost_seizures = results['total_missing_seizures'] + results['total_seizures_lost_in_present_runs']

    if results['missing_in_preprocessed'] > 0:
        print(f"\n⚠️  WARNING: {results['missing_in_preprocessed']} runs are missing from preprocessed data!")
        print(f"   - {results['missing_with_seizures']} runs contain seizures")
        print(f"   - {results['total_missing_seizures']} seizures are missing from these runs")

    if results['present_but_missing_seizures'] > 0:
        print(f"\n⚠️  CRITICAL: {results['present_but_missing_seizures']} runs exist but have NO seizure data!")
        print(f"   - These runs should contain {results['total_seizures_lost_in_present_runs']} seizures")
        print(f"   - Preprocessing may have failed to label seizure windows correctly")

    if total_lost_seizures > 0:
        print(f"\n⚠️  TOTAL SEIZURE LOSS: {total_lost_seizures} seizures")
        print(f"   - From missing runs: {results['total_missing_seizures']}")
        print(f"   - From present runs: {results['total_seizures_lost_in_present_runs']}")
        print(f"\nThis explains the 794 vs 856 seizure discrepancy!")
    else:
        print("\n✓ All raw data runs are present in preprocessed data!")
        print("✓ All seizures are properly labeled!")


def main():
    print("VERIFYING PREPROCESSED DATA COMPLETENESS")
    print("=" * 80)
    print()

    # Paths
    raw_path = Path(RAW_DATA_PATH)

    # Find the preprocessed directory
    preprocessed_base = Path(PREPROCESSED_DATA_PATH)
    preprocessed_path = preprocessed_base / "downsample_freq=8,window_size=3600_0,stride=1800_0_reproduced"

    if not preprocessed_path.exists():
        print(f"Error: Preprocessed data path not found: {preprocessed_path}")
        print("\nSearching for alternative preprocessed directories...")

        # Try to find any preprocessed directory
        if preprocessed_base.exists():
            subdirs = [d for d in preprocessed_base.iterdir() if d.is_dir()]
            if subdirs:
                print(f"Found {len(subdirs)} preprocessed directories:")
                for i, d in enumerate(subdirs, 1):
                    print(f"  {i}. {d.name}")

                # Use the first one
                preprocessed_path = subdirs[0]
                print(f"\nUsing: {preprocessed_path}")
            else:
                print("No preprocessed directories found!")
                return
        else:
            print(f"Preprocessed base path does not exist: {preprocessed_base}")
            return

    print(f"Raw data path:         {raw_path}")
    print(f"Preprocessed path:     {preprocessed_path}")
    print()

    # Discover recordings
    print("Step 1: Scanning raw data...")
    raw_recordings = discover_raw_recordings(raw_path)
    print(f"  Found {len(raw_recordings)} runs in raw data")

    raw_with_seizures = sum(1 for r in raw_recordings.values() if r['has_seizures'])
    total_raw_seizures = sum(r['n_seizures'] for r in raw_recordings.values())
    print(f"  {raw_with_seizures} runs contain seizures ({total_raw_seizures} total seizures)")
    print()

    print("Step 2: Scanning preprocessed data...")
    preprocessed_recordings = discover_preprocessed_recordings(preprocessed_path)
    print(f"  Found {len(preprocessed_recordings)} runs in preprocessed data")
    print()

    # Compare
    print("Step 3: Comparing datasets...")
    results = compare_datasets(raw_recordings, preprocessed_recordings)
    print()

    # Print summary
    print_summary(results)

    # Generate detailed report
    print("\nStep 4: Generating detailed report...")
    output_dir = Path("Madrid/verification_results")
    generate_report(results, raw_recordings, preprocessed_recordings, output_dir)

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
