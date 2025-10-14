#!/usr/bin/env python3
"""
Analyze ds005873-1.1.0_example data to identify:
1. Patients with missing or empty ECG signals
2. Count annotated seizures vs actual seizures in ECG data
3. Generate report of usable vs unusable data
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def analyze_example_data():
    base_dir = Path("ds005873-1.1.0_example")

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found")
        return

    results = {
        'patients': {},
        'summary': {
            'total_patients': 0,
            'total_runs': 0,
            'runs_with_seizures': 0,
            'runs_with_empty_ecg': 0,
            'runs_with_missing_ecg': 0,
            'total_seizures_annotated': 0,
            'usable_runs': 0,
            'unusable_runs': 0
        }
    }

    # Get all subjects
    subjects = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    results['summary']['total_patients'] = len(subjects)

    print(f"Found {len(subjects)} subjects: {[s.name for s in subjects]}")
    print("\nAnalyzing each subject and run...\n")

    for subject in sorted(subjects):
        subject_id = subject.name
        results['patients'][subject_id] = {
            'runs': {},
            'total_runs': 0,
            'runs_with_seizures': 0,
            'runs_with_empty_ecg': 0,
            'runs_with_missing_ecg': 0,
            'total_seizures': 0,
            'usable_runs': 0
        }

        # Get sessions (should be ses-01)
        sessions = [d for d in subject.iterdir() if d.is_dir() and d.name.startswith('ses-')]

        for session in sessions:
            ecg_dir = session / "ecg"
            eeg_dir = session / "eeg"

            if not ecg_dir.exists():
                print(f"  Warning: No ECG directory for {subject_id}")
                continue

            # Get all runs for this subject
            ecg_files = list(ecg_dir.glob("*_ecg.edf"))

            for ecg_file in sorted(ecg_files):
                # Extract run number from filename
                run_match = ecg_file.name.split('_run-')[1].split('_')[0]
                run_id = f"run-{run_match.zfill(2)}"

                results['patients'][subject_id]['total_runs'] += 1
                results['summary']['total_runs'] += 1

                # Initialize run data
                run_data = {
                    'ecg_file': str(ecg_file),
                    'events_file': None,
                    'ecg_exists': ecg_file.exists(),
                    'ecg_filesize': ecg_file.stat().st_size if ecg_file.exists() else 0,
                    'ecg_empty': False,
                    'ecg_duration': 0,
                    'seizures_annotated': 0,
                    'seizure_events': [],
                    'usable': False
                }

                # Check corresponding events file
                events_file = eeg_dir / f"{subject_id}_ses-01_task-szMonitoring_{run_id}_events.tsv"
                if events_file.exists():
                    run_data['events_file'] = str(events_file)

                    # Read seizure annotations (SeizeIT2 format)
                    try:
                        events_df = pd.read_csv(events_file, sep='\t')
                        if 'eventType' in events_df.columns:
                            # In SeizeIT2, seizures start with 'sz_' and exclude 'bckg' and 'impd'
                            seizures = events_df[
                                (events_df['eventType'].str.startswith('sz_', na=False)) &
                                (events_df['eventType'] != 'bckg') &
                                (events_df['eventType'] != 'impd')
                            ]
                            run_data['seizures_annotated'] = len(seizures)
                            run_data['seizure_events'] = seizures[['onset', 'duration', 'eventType']].to_dict('records')

                            if len(seizures) > 0:
                                results['patients'][subject_id]['runs_with_seizures'] += 1
                                results['summary']['runs_with_seizures'] += 1

                            results['patients'][subject_id]['total_seizures'] += len(seizures)
                            results['summary']['total_seizures_annotated'] += len(seizures)

                            # Also log all event types for debugging
                            all_events = events_df['eventType'].unique()
                            run_data['all_event_types'] = list(all_events)
                    except Exception as e:
                        print(f"    Error reading events file {events_file}: {e}")

                # Analyze ECG file (simplified - just check file size and existence)
                if run_data['ecg_exists']:
                    try:
                        # Basic file analysis without pyedflib
                        file_size = run_data['ecg_filesize']

                        # Assume files smaller than 10KB are likely empty/corrupted
                        if file_size < 10000:
                            run_data['ecg_empty'] = True
                            print(f"    {subject_id} {run_id}: ECG file too small ({file_size} bytes)")
                        else:
                            # Estimate duration based on typical EDF file sizes (rough approximation)
                            # A typical 1-hour ECG at 250Hz with 1 channel ≈ 3.6MB
                            estimated_hours = file_size / (3.6 * 1024 * 1024)
                            run_data['ecg_duration'] = estimated_hours * 3600

                    except Exception as e:
                        print(f"    {subject_id} {run_id}: Error analyzing ECG file: {e}")
                        run_data['ecg_empty'] = True

                else:
                    run_data['ecg_empty'] = True
                    results['patients'][subject_id]['runs_with_missing_ecg'] += 1
                    results['summary']['runs_with_missing_ecg'] += 1

                # Count empty ECG runs
                if run_data['ecg_empty']:
                    results['patients'][subject_id]['runs_with_empty_ecg'] += 1
                    results['summary']['runs_with_empty_ecg'] += 1

                # Determine if run is usable (has ECG data)
                run_data['usable'] = (not run_data['ecg_empty'] and
                                    run_data['ecg_filesize'] > 10000)  # At least 10KB

                if run_data['usable']:
                    results['patients'][subject_id]['usable_runs'] += 1
                    results['summary']['usable_runs'] += 1
                else:
                    results['summary']['unusable_runs'] += 1

                results['patients'][subject_id]['runs'][run_id] = run_data

                # Print status for each run
                status = "✓ USABLE" if run_data['usable'] else "✗ UNUSABLE"
                seizure_info = f", {run_data['seizures_annotated']} seizures" if run_data['seizures_annotated'] > 0 else ""
                duration_info = f", {run_data['ecg_duration']:.1f}s" if run_data['ecg_duration'] > 0 else ""
                problem = ""
                if run_data['ecg_empty']:
                    problem = " (ECG empty/missing)"
                elif run_data['ecg_filesize'] < 10000:
                    problem = " (ECG file too small)"

                event_types_info = ""
                if 'all_event_types' in run_data and len(run_data['all_event_types']) > 0:
                    event_types_info = f", events: {run_data['all_event_types']}"

                print(f"  {subject_id} {run_id}: {status}{seizure_info}{duration_info}{event_types_info}{problem}")

    return results

def print_summary_report(results):
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)

    summary = results['summary']

    print(f"\nOVERALL STATISTICS:")
    print(f"  Total patients: {summary['total_patients']}")
    print(f"  Total runs: {summary['total_runs']}")
    print(f"  Usable runs: {summary['usable_runs']} ({summary['usable_runs']/summary['total_runs']*100:.1f}%)")
    print(f"  Unusable runs: {summary['unusable_runs']} ({summary['unusable_runs']/summary['total_runs']*100:.1f}%)")

    print(f"\nPROBLEMS IDENTIFIED:")
    print(f"  Runs with missing ECG files: {summary['runs_with_missing_ecg']}")
    print(f"  Runs with empty/unusable ECG: {summary['runs_with_empty_ecg']}")

    print(f"\nSEIZURE ANNOTATIONS:")
    print(f"  Total annotated seizures: {summary['total_seizures_annotated']}")
    print(f"  Runs with seizures: {summary['runs_with_seizures']}")
    print(f"  Runs with seizures that are usable: {len([p for p in results['patients'].values() if any(r['usable'] and r['seizures_annotated'] > 0 for r in p['runs'].values())])}")

    print(f"\nPER-PATIENT BREAKDOWN:")
    for patient_id, patient_data in results['patients'].items():
        usable_runs_with_seizures = sum(1 for r in patient_data['runs'].values() if r['usable'] and r['seizures_annotated'] > 0)
        print(f"  {patient_id}:")
        print(f"    Total runs: {patient_data['total_runs']}")
        print(f"    Usable runs: {patient_data['usable_runs']}")
        print(f"    Total seizures: {patient_data['total_seizures']}")
        print(f"    Usable runs with seizures: {usable_runs_with_seizures}")

        if patient_data['runs_with_empty_ecg'] > 0 or patient_data['runs_with_missing_ecg'] > 0:
            print(f"    ⚠️  Problems: {patient_data['runs_with_empty_ecg']} empty ECG, {patient_data['runs_with_missing_ecg']} missing ECG")

def main():
    print("Analyzing ds005873-1.1.0_example data...")
    print("="*60)

    try:
        results = analyze_example_data()
        print_summary_report(results)

        # Save detailed results to JSON
        with open('example_data_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✓ Detailed results saved to 'example_data_analysis.json'")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()