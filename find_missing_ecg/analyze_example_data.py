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
import pyedflib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def analyze_ecg_signal(ecg_file_path, subject_id, run_id):
    """
    Detailed analysis of ECG signal quality
    Returns dict with analysis results
    """
    analysis = {
        'ecg_empty': False,
        'ecg_duration': 0,
        'ecg_channels_found': [],
        'ecg_sampling_rates': [],
        'ecg_problem': None,
        'signal_quality': 'unknown',
        'zero_signal_channels': 0,
        'saturated_signal_channels': 0,
        'total_channels': 0
    }

    try:
        with pyedflib.EdfReader(ecg_file_path) as f:
            # Basic file info
            analysis['ecg_duration'] = f.file_duration
            analysis['total_channels'] = f.signals_in_file
            signal_labels = f.getSignalLabels()

            # Find ECG channels
            ecg_channel_indices = []
            for i, label in enumerate(signal_labels):
                label_lower = label.lower()
                if any(ecg_keyword in label_lower for ecg_keyword in ['ecg', 'ekg', 'lead']):
                    ecg_channel_indices.append(i)
                    analysis['ecg_channels_found'].append(label)
                    analysis['ecg_sampling_rates'].append(f.getSampleFrequency(i))

            # Check if any ECG channels found
            if not ecg_channel_indices:
                analysis['ecg_empty'] = True
                analysis['ecg_problem'] = f"no ECG channels found (available: {signal_labels})"
                return analysis

            # Analyze each ECG channel for signal quality
            zero_channels = 0
            saturated_channels = 0

            for channel_idx in ecg_channel_indices:
                try:
                    # Read a sample of the signal (first 30 seconds or max 10000 samples)
                    sample_size = min(200000, int(f.getSampleFrequency(channel_idx) * 600))
                    if f.getNSamples()[channel_idx] < sample_size:
                        sample_size = f.getNSamples()[channel_idx]

                    if sample_size == 0:
                        zero_channels += 1
                        continue

                    signal_data = f.readSignal(channel_idx, start=0, n=sample_size)

                    if len(signal_data) == 0:
                        zero_channels += 1
                        continue

                    # Check for all-zero signal
                    if np.all(signal_data == 0):
                        zero_channels += 1
                        continue

                    # Check for saturated signal (all values at same level)
                    if len(np.unique(signal_data)) < 10:  # Less than 10 unique values suggests saturation
                        saturated_channels += 1
                        continue

                    # Check for reasonable signal range (ECG should be in mV range, typically -5 to +5 mV)
                    signal_range = np.max(signal_data) - np.min(signal_data)
                    if signal_range < 0.001:  # Less than 1 microV range
                        saturated_channels += 1
                        continue

                except Exception as e:
                    print(f"      Error reading channel {channel_idx} ({signal_labels[channel_idx]}): {e}")
                    zero_channels += 1
                    continue

            analysis['zero_signal_channels'] = zero_channels
            analysis['saturated_signal_channels'] = saturated_channels

            # Determine overall signal quality
            usable_channels = len(ecg_channel_indices) - zero_channels - saturated_channels

            if usable_channels == 0:
                analysis['ecg_empty'] = True
                if zero_channels > 0 and saturated_channels > 0:
                    analysis['ecg_problem'] = f"{zero_channels} zero channels, {saturated_channels} saturated channels"
                elif zero_channels > 0:
                    analysis['ecg_problem'] = f"all {zero_channels} ECG channels contain zero signal"
                elif saturated_channels > 0:
                    analysis['ecg_problem'] = f"all {saturated_channels} ECG channels are saturated"
            elif usable_channels < len(ecg_channel_indices):
                analysis['signal_quality'] = 'partial'
                analysis['ecg_problem'] = f"only {usable_channels}/{len(ecg_channel_indices)} ECG channels usable"
            else:
                analysis['signal_quality'] = 'good'

    except Exception as e:
        analysis['ecg_empty'] = True
        analysis['ecg_problem'] = f"EDF read error: {str(e)}"

    return analysis

import re

def parse_run_id(name: str) -> str:
    """Extrahiert run-ID als 'run-XX' (zweistellig). Fallback: Hash."""
    m = re.search(r"_run-(\d+)\b", name)
    if m:
        return f"run-{int(m.group(1)):02d}"
    base = Path(name).stem
    return f"run-{hash(base) & 0xffff:04x}"

def count_seizures_in_events_file(events_file: Path) -> tuple[int, list[dict]]:
    """Liest *_events.tsv und zählt Seizures im SeizeIT2-Format. Gibt (count, events_list) zurück."""
    try:
        df = pd.read_csv(events_file, sep="\t")
        if "eventType" not in df.columns:
            return 0, []
        seizures = df[
            (df["eventType"].str.startswith("sz_", na=False)) &
            (df["eventType"] != "bckg") &
            (df["eventType"] != "impd")
        ]
        events_list = seizures[["onset", "duration", "eventType"]].to_dict("records")
        return len(seizures), events_list
    except Exception as e:
        print(f"    Error reading events file {events_file}: {e}")
        return 0, []


def analyze_example_data():
    base_dir = Path("/home/swolf/asim_shared/raw_data/ds005873-1.1.0")

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
            'unusable_runs': 0,
            'orphaned_events': 0,  # Events ohne entsprechende ECG-Datei
            'total_event_files': 0
        },
        'orphaned_event_files': []  # Liste von Event-Dateien ohne ECG
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

                # Analyze ECG file (detailed analysis with pyedflib)
                if run_data['ecg_exists']:
                    try:
                        # Check file size first
                        file_size = run_data['ecg_filesize']
                        if file_size < 10000:
                            run_data['ecg_empty'] = True
                            run_data['ecg_problem'] = f"file too small ({file_size} bytes)"
                            print(f"    {subject_id} {run_id}: ECG file too small ({file_size} bytes)")
                        else:
                            # Detailed EDF analysis
                            ecg_analysis = analyze_ecg_signal(str(ecg_file), subject_id, run_id)
                            run_data.update(ecg_analysis)

                            if ecg_analysis['ecg_empty']:
                                print(f"    {subject_id} {run_id}: ECG unusable - {ecg_analysis.get('ecg_problem', 'unknown issue')}")

                    except Exception as e:
                        print(f"    {subject_id} {run_id}: Error analyzing ECG file: {e}")
                        run_data['ecg_empty'] = True
                        run_data['ecg_problem'] = f"file read error: {str(e)}"

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

                # ECG quality info
                ecg_quality_info = ""
                if 'ecg_channels_found' in run_data and run_data['ecg_channels_found']:
                    channels = len(run_data['ecg_channels_found'])
                    quality = run_data.get('signal_quality', 'unknown')
                    ecg_quality_info = f", ECG: {channels} channels ({quality})"

                problem = ""
                if run_data['ecg_empty']:
                    if 'ecg_problem' in run_data:
                        problem = f" (ECG: {run_data['ecg_problem']})"
                    else:
                        problem = " (ECG empty/missing)"

                event_types_info = ""
                if 'all_event_types' in run_data and len(run_data['all_event_types']) > 0:
                    event_types_info = f", events: {run_data['all_event_types']}"

                print(f"  {subject_id} {run_id}: {status}{seizure_info}{duration_info}{ecg_quality_info}{event_types_info}{problem}")

    # After processing all ECG files, check for orphaned event files
    print("\n" + "="*60)
    print("Checking for event files without corresponding ECG files...")
    print("="*60 + "\n")

    for subject in sorted(subjects):
        subject_id = subject.name
        sessions = [d for d in subject.iterdir() if d.is_dir() and d.name.startswith('ses-')]

        for session in sessions:
            eeg_dir = session / "eeg"
            ecg_dir = session / "ecg"

            if not eeg_dir.exists():
                continue

            # Get all event files
            event_files = list(eeg_dir.glob("*_events.tsv"))
            results['summary']['total_event_files'] += len(event_files)

            for event_file in sorted(event_files):
                # Extract run ID from event filename
                run_id = parse_run_id(event_file.name)

                # Check if this run_id is in the results (meaning ECG file exists)
                if run_id not in results['patients'][subject_id]['runs']:
                    # This is an orphaned event file - no corresponding ECG
                    seizure_count, seizure_events = count_seizures_in_events_file(event_file)

                    orphan_info = {
                        'patient_id': subject_id,
                        'run_id': run_id,
                        'events_file': str(event_file),
                        'expected_ecg_file': str(ecg_dir / f"{subject_id}_ses-01_task-szMonitoring_{run_id}_ecg.edf"),
                        'seizures_annotated': seizure_count,
                        'seizure_events': seizure_events
                    }

                    results['orphaned_event_files'].append(orphan_info)
                    results['summary']['orphaned_events'] += 1

                    if seizure_count > 0:
                        results['summary']['total_seizures_annotated'] += seizure_count

                    print(f"  ⚠️  {subject_id} {run_id}: Event file exists but ECG file missing!")
                    if seizure_count > 0:
                        print(f"      Contains {seizure_count} annotated seizures - potential DATA LOSS!")

    if results['summary']['orphaned_events'] == 0:
        print("  ✓ No orphaned event files found - all events have corresponding ECG files.")

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

    print(f"\nECG QUALITY ANALYSIS:")
    print(f"  Runs with missing ECG files: {summary['runs_with_missing_ecg']}")
    print(f"  Runs with empty/unusable ECG: {summary['runs_with_empty_ecg']}")
    print(f"\nEVENT FILE ANALYSIS:")
    print(f"  Total event files found: {summary['total_event_files']}")
    print(f"  Orphaned event files (no ECG): {summary['orphaned_events']}")
    if summary['orphaned_events'] > 0:
        orphaned_with_seizures = sum(1 for o in results['orphaned_event_files'] if o['seizures_annotated'] > 0)
        total_orphaned_seizures = sum(o['seizures_annotated'] for o in results['orphaned_event_files'])
        print(f"    ⚠️  {orphaned_with_seizures} orphaned event files contain {total_orphaned_seizures} seizures!")

    # Count ECG quality issues
    ecg_quality_stats = {
        'no_ecg_channels': 0,
        'zero_signal': 0,
        'saturated_signal': 0,
        'partial_quality': 0,
        'good_quality': 0,
        'file_errors': 0
    }

    for patient_data in results['patients'].values():
        for run_data in patient_data['runs'].values():
            if 'ecg_problem' in run_data and run_data['ecg_problem']:
                if 'no ECG channels found' in run_data['ecg_problem']:
                    ecg_quality_stats['no_ecg_channels'] += 1
                elif 'zero signal' in run_data['ecg_problem']:
                    ecg_quality_stats['zero_signal'] += 1
                elif 'saturated' in run_data['ecg_problem']:
                    ecg_quality_stats['saturated_signal'] += 1
                elif 'EDF read error' in run_data['ecg_problem'] or 'file read error' in run_data['ecg_problem']:
                    ecg_quality_stats['file_errors'] += 1
            elif run_data.get('signal_quality') == 'partial':
                ecg_quality_stats['partial_quality'] += 1
            elif run_data.get('signal_quality') == 'good':
                ecg_quality_stats['good_quality'] += 1

    print(f"  ECG Quality Breakdown:")
    print(f"    Good quality: {ecg_quality_stats['good_quality']}")
    print(f"    Partial quality: {ecg_quality_stats['partial_quality']}")
    print(f"    No ECG channels found: {ecg_quality_stats['no_ecg_channels']}")
    print(f"    Zero/empty signal: {ecg_quality_stats['zero_signal']}")
    print(f"    Saturated signal: {ecg_quality_stats['saturated_signal']}")
    print(f"    File read errors: {ecg_quality_stats['file_errors']}")

    print(f"\nSEIZURE ANNOTATIONS:")
    print(f"  Total annotated seizures: {summary['total_seizures_annotated']}")
    print(f"  Runs with seizures: {summary['runs_with_seizures']}")
    usable_runs_with_seizures = sum(sum(1 for r in patient_data['runs'].values() if r['usable'] and r['seizures_annotated'] > 0) for patient_data in results['patients'].values())
    print(f"  Runs with seizures that are usable: {usable_runs_with_seizures}")

    patients_with_usable_seizure_runs = len([p for p in results['patients'].values() if any(r['usable'] and r['seizures_annotated'] > 0 for r in p['runs'].values())])
    print(f"  Patients with usable seizure runs: {patients_with_usable_seizure_runs}")

    # Find patients without usable seizure runs and analyze reasons
    patients_without_usable_seizure_runs = []
    for patient_id, patient_data in results['patients'].items():
        if not any(r['usable'] and r['seizures_annotated'] > 0 for r in patient_data['runs'].values()):
            # Analyze why this patient has no usable seizure runs
            total_seizures = patient_data['total_seizures']
            usable_runs = patient_data['usable_runs']
            total_runs = patient_data['total_runs']

            reasons = []
            if total_seizures == 0:
                reasons.append("no seizures annotated")
            else:
                # Patient has seizures, check why they're not usable
                seizure_runs_unusable = 0
                for run_id, run_data in patient_data['runs'].items():
                    if run_data['seizures_annotated'] > 0 and not run_data['usable']:
                        seizure_runs_unusable += 1

                if seizure_runs_unusable > 0:
                    reasons.append(f"{seizure_runs_unusable} seizure runs with unusable ECG")
                if usable_runs < total_runs:
                    reasons.append(f"{total_runs - usable_runs} total unusable runs")

            patients_without_usable_seizure_runs.append({
                'id': patient_id,
                'total_seizures': total_seizures,
                'usable_runs': usable_runs,
                'total_runs': total_runs,
                'reasons': reasons
            })

    print(f"  Patients without usable seizure runs: {len(patients_without_usable_seizure_runs)}")

    if patients_without_usable_seizure_runs:
        print(f"\nDETAILED BREAKDOWN OF PATIENTS WITHOUT USABLE SEIZURE RUNS:")
        for patient in sorted(patients_without_usable_seizure_runs, key=lambda x: x['id']):
            reason_text = "; ".join(patient['reasons']) if patient['reasons'] else "unknown reason"
            print(f"    {patient['id']}: {patient['total_seizures']} seizures, {patient['usable_runs']}/{patient['total_runs']} usable runs - {reason_text}")

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

def generate_problems_report(results):
    """
    Generate a detailed TXT report listing all problems found in the data.

    Args:
        results: Dictionary containing analysis results

    Returns:
        str: Formatted report text
    """
    from datetime import datetime

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ECG DATA QUALITY PROBLEMS REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: ds005873-1.1.0_example")
    report_lines.append("")

    # Collect all problems
    problems = []

    # First, add orphaned event files as problems
    for orphan in results.get('orphaned_event_files', []):
        problem = {
            'patient_id': orphan['patient_id'],
            'run_id': orphan['run_id'],
            'ecg_file': orphan['expected_ecg_file'],
            'problem_type': 'Orphaned Event File',
            'description': 'Event file exists but corresponding ECG file is missing',
            'severity': 'CRITICAL',
            'details': {
                'events_file': orphan['events_file'],
                'expected_ecg_file': orphan['expected_ecg_file'],
                'has_seizures': orphan['seizures_annotated'] > 0,
                'seizure_count': orphan['seizures_annotated'],
                'seizure_events': orphan.get('seizure_events', [])
            }
        }
        problems.append(problem)

    # Then add problems from runs with ECG files
    for patient_id, patient_data in sorted(results['patients'].items()):
        for run_id, run_data in sorted(patient_data['runs'].items()):
            # Check if run has any problems
            if run_data.get('ecg_empty') or not run_data.get('usable'):
                problem = {
                    'patient_id': patient_id,
                    'run_id': run_id,
                    'ecg_file': run_data.get('ecg_file', 'N/A'),
                    'problem_type': 'Unknown',
                    'description': '',
                    'severity': 'CRITICAL',
                    'details': {}
                }

                # Determine problem type and severity
                if not run_data.get('ecg_exists'):
                    problem['problem_type'] = 'Missing ECG File'
                    problem['description'] = 'ECG file does not exist'
                    problem['severity'] = 'CRITICAL'

                elif run_data.get('ecg_filesize', 0) < 10000:
                    problem['problem_type'] = 'File Too Small'
                    problem['description'] = f"ECG file too small ({run_data.get('ecg_filesize', 0)} bytes)"
                    problem['severity'] = 'CRITICAL'
                    problem['details']['file_size'] = run_data.get('ecg_filesize', 0)

                elif 'ecg_problem' in run_data and run_data['ecg_problem']:
                    ecg_problem = run_data['ecg_problem']

                    if 'no ECG channels found' in ecg_problem:
                        problem['problem_type'] = 'No ECG Channels'
                        problem['description'] = ecg_problem
                        problem['severity'] = 'CRITICAL'

                    elif 'zero signal' in ecg_problem.lower():
                        problem['problem_type'] = 'Zero/Empty Signal'
                        problem['description'] = ecg_problem
                        problem['severity'] = 'CRITICAL'
                        problem['details']['zero_channels'] = run_data.get('zero_signal_channels', 0)

                    elif 'saturated' in ecg_problem.lower():
                        problem['problem_type'] = 'Saturated Signal'
                        problem['description'] = ecg_problem
                        problem['severity'] = 'CRITICAL'
                        problem['details']['saturated_channels'] = run_data.get('saturated_signal_channels', 0)

                    elif 'only' in ecg_problem.lower() and 'usable' in ecg_problem.lower():
                        problem['problem_type'] = 'Partial Signal Quality'
                        problem['description'] = ecg_problem
                        problem['severity'] = 'WARNING'

                    elif 'error' in ecg_problem.lower():
                        problem['problem_type'] = 'File Read Error'
                        problem['description'] = ecg_problem
                        problem['severity'] = 'CRITICAL'

                    else:
                        problem['problem_type'] = 'Other ECG Problem'
                        problem['description'] = ecg_problem
                        problem['severity'] = 'WARNING'

                # Add additional context
                problem['details']['duration'] = run_data.get('ecg_duration', 0)
                problem['details']['channels_found'] = run_data.get('ecg_channels_found', [])
                problem['details']['sampling_rates'] = run_data.get('ecg_sampling_rates', [])
                problem['details']['signal_quality'] = run_data.get('signal_quality', 'unknown')
                problem['details']['has_seizures'] = run_data.get('seizures_annotated', 0) > 0
                problem['details']['seizure_count'] = run_data.get('seizures_annotated', 0)

                problems.append(problem)

    # Summary statistics
    report_lines.append("SUMMARY")
    report_lines.append("-"*80)
    report_lines.append(f"Total problems found: {len(problems)}")

    # Count by severity
    critical_count = sum(1 for p in problems if p['severity'] == 'CRITICAL')
    warning_count = sum(1 for p in problems if p['severity'] == 'WARNING')
    report_lines.append(f"  - CRITICAL: {critical_count}")
    report_lines.append(f"  - WARNING: {warning_count}")
    report_lines.append("")

    # Count by problem type
    problem_types = {}
    for p in problems:
        ptype = p['problem_type']
        problem_types[ptype] = problem_types.get(ptype, 0) + 1

    report_lines.append("Problems by type:")
    for ptype, count in sorted(problem_types.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"  - {ptype}: {count}")
    report_lines.append("")

    # Problems with seizure annotations
    problems_with_seizures = [p for p in problems if p['details']['has_seizures']]
    if problems_with_seizures:
        total_lost_seizures = sum(p['details']['seizure_count'] for p in problems_with_seizures)
        report_lines.append(f"⚠️  CRITICAL: {len(problems_with_seizures)} problematic runs contain {total_lost_seizures} annotated seizures!")
        report_lines.append("")

    # Detailed problem listing
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("DETAILED PROBLEM LISTING")
    report_lines.append("="*80)
    report_lines.append("")

    # Group by patient
    problems_by_patient = {}
    for p in problems:
        patient_id = p['patient_id']
        if patient_id not in problems_by_patient:
            problems_by_patient[patient_id] = []
        problems_by_patient[patient_id].append(p)

    for patient_id in sorted(problems_by_patient.keys()):
        patient_problems = problems_by_patient[patient_id]
        report_lines.append(f"PATIENT: {patient_id}")
        report_lines.append("-"*80)

        for problem in sorted(patient_problems, key=lambda x: x['run_id']):
            report_lines.append(f"  Run: {problem['run_id']}")
            report_lines.append(f"  Severity: {problem['severity']}")
            report_lines.append(f"  Problem Type: {problem['problem_type']}")
            report_lines.append(f"  Description: {problem['description']}")

            # For orphaned events, show both event file and expected ECG file
            if problem['problem_type'] == 'Orphaned Event File':
                report_lines.append(f"  Event File: {problem['details'].get('events_file', 'N/A')}")
                report_lines.append(f"  Expected ECG File: {problem['details'].get('expected_ecg_file', 'N/A')}")
            else:
                report_lines.append(f"  File: {problem['ecg_file']}")

            # Additional details
            if problem['details'].get('duration', 0) > 0:
                report_lines.append(f"  Duration: {problem['details']['duration']:.1f} seconds")

            if problem['details'].get('channels_found'):
                channels_str = ", ".join(problem['details']['channels_found'])
                report_lines.append(f"  Channels Found: {channels_str}")

            if problem['details'].get('sampling_rates'):
                rates_str = ", ".join(str(r) for r in problem['details']['sampling_rates'])
                report_lines.append(f"  Sampling Rates: {rates_str} Hz")

            if problem['details'].get('signal_quality') != 'unknown' and problem['details'].get('signal_quality'):
                report_lines.append(f"  Signal Quality: {problem['details']['signal_quality']}")

            if problem['details'].get('has_seizures'):
                seizure_count = problem['details']['seizure_count']
                report_lines.append(f"  ⚠️  Contains {seizure_count} annotated seizure(s) - DATA LOSS!")

            report_lines.append("")

    # Add recommendations section
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("="*80)
    report_lines.append("")

    if critical_count > 0:
        report_lines.append("1. CRITICAL ISSUES FOUND:")
        report_lines.append("   - Review files with missing or corrupted ECG data")
        report_lines.append("   - Consider re-downloading or re-processing affected files")
        if problems_with_seizures:
            report_lines.append(f"   - URGENT: {len(problems_with_seizures)} runs with seizures are unusable!")
        report_lines.append("")

    # Count specific recommendations
    orphaned_events = sum(1 for p in problems if p['problem_type'] == 'Orphaned Event File')
    if orphaned_events > 0:
        report_lines.append(f"2. ORPHANED EVENT FILES ({orphaned_events} files):")
        report_lines.append("   - Event files exist but corresponding ECG files are missing")
        report_lines.append("   - Check if ECG files were not recorded or lost during transfer")
        report_lines.append("   - Verify file naming conventions match between EEG and ECG directories")
        orphaned_with_seizures = sum(1 for p in problems if p['problem_type'] == 'Orphaned Event File' and p['details']['has_seizures'])
        if orphaned_with_seizures > 0:
            report_lines.append(f"   - ⚠️  CRITICAL: {orphaned_with_seizures} orphaned files contain seizure annotations!")
        report_lines.append("")

    no_channels = sum(1 for p in problems if p['problem_type'] == 'No ECG Channels')
    if no_channels > 0:
        report_lines.append(f"3. NO ECG CHANNELS ({no_channels} files):")
        report_lines.append("   - Verify that ECG data was recorded for these sessions")
        report_lines.append("   - Check if different channel naming conventions are used")
        report_lines.append("")

    zero_signal = sum(1 for p in problems if p['problem_type'] == 'Zero/Empty Signal')
    if zero_signal > 0:
        report_lines.append(f"4. ZERO/EMPTY SIGNALS ({zero_signal} files):")
        report_lines.append("   - These files may be corrupted during recording or transfer")
        report_lines.append("   - Consider excluding from analysis")
        report_lines.append("")

    saturated = sum(1 for p in problems if p['problem_type'] == 'Saturated Signal')
    if saturated > 0:
        report_lines.append(f"5. SATURATED SIGNALS ({saturated} files):")
        report_lines.append("   - Check recording equipment calibration")
        report_lines.append("   - May indicate amplifier saturation or disconnected electrodes")
        report_lines.append("")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    return "\n".join(report_lines)

def plot_saturated_signals(results, max_plots=10):
    """
    Plot ECG signals from files with saturated signals for visual inspection.

    Args:
        results: Dictionary containing analysis results
        max_plots: Maximum number of files to plot (default: 3)
    """
    print("\nSearching for files with saturated signals to plot...")

    # Find files with saturated signals
    saturated_files = []
    for patient_id, patient_data in results['patients'].items():
        for run_id, run_data in patient_data['runs'].items():
            if 'ecg_problem' in run_data and run_data['ecg_problem']:
                if 'saturated' in run_data['ecg_problem'].lower():
                    saturated_files.append({
                        'patient_id': patient_id,
                        'run_id': run_id,
                        'ecg_file': run_data['ecg_file'],
                        'problem': run_data['ecg_problem'],
                        'channels': run_data.get('ecg_channels_found', []),
                        'sampling_rates': run_data.get('ecg_sampling_rates', [])
                    })

    if not saturated_files:
        print("No files with saturated signals found.")
        return

    print(f"Found {len(saturated_files)} files with saturated signals.")
    print(f"Plotting first {min(max_plots, len(saturated_files))} files...\n")

    # Create output directory for plots
    output_dir = Path("saturated_signal_plots")
    output_dir.mkdir(exist_ok=True)

    # Plot up to max_plots files
    for idx, file_info in enumerate(saturated_files[:max_plots]):
        try:
            print(f"Plotting {idx+1}/{min(max_plots, len(saturated_files))}: {file_info['patient_id']} {file_info['run_id']}")

            ecg_file_path = file_info['ecg_file']

            # Read the ECG file
            with pyedflib.EdfReader(ecg_file_path) as f:
                signal_labels = f.getSignalLabels()

                # Find ECG channels
                ecg_channel_indices = []
                for i, label in enumerate(signal_labels):
                    label_lower = label.lower()
                    if any(ecg_keyword in label_lower for ecg_keyword in ['ecg', 'ekg', 'lead']):
                        ecg_channel_indices.append(i)

                if not ecg_channel_indices:
                    print(f"  Skipping - no ECG channels found")
                    continue

                # Read data from all ECG channels
                num_channels = len(ecg_channel_indices)
                fig, axes = plt.subplots(num_channels, 1, figsize=(15, 4*num_channels))

                if num_channels == 1:
                    axes = [axes]  # Make it iterable

                for plot_idx, channel_idx in enumerate(ecg_channel_indices):
                    # Read signal (first 60 seconds or max 15000 samples for better overview)
                    sample_size = min(150000, int(f.getSampleFrequency(channel_idx) * 600))
                    if f.getNSamples()[channel_idx] < sample_size:
                        sample_size = f.getNSamples()[channel_idx]

                    signal_data = f.readSignal(channel_idx, start=0, n=sample_size)
                    sampling_rate = f.getSampleFrequency(channel_idx)

                    # Create time axis
                    time_axis = np.arange(len(signal_data)) / sampling_rate

                    # Plot
                    axes[plot_idx].plot(time_axis, signal_data, linewidth=0.5)
                    axes[plot_idx].set_xlabel('Time (seconds)', fontsize=10)
                    axes[plot_idx].set_ylabel('Amplitude', fontsize=10)
                    axes[plot_idx].set_title(f'Channel: {signal_labels[channel_idx]} (Sample Rate: {sampling_rate} Hz)',
                                            fontsize=11)
                    axes[plot_idx].grid(True, alpha=0.3)

                    # Add statistics to plot
                    stats_text = (f'Mean: {np.mean(signal_data):.4f}\n'
                                 f'Std: {np.std(signal_data):.4f}\n'
                                 f'Min: {np.min(signal_data):.4f}\n'
                                 f'Max: {np.max(signal_data):.4f}\n'
                                 f'Range: {np.max(signal_data) - np.min(signal_data):.4f}\n'
                                 f'Unique values: {len(np.unique(signal_data))}')

                    axes[plot_idx].text(0.02, 0.98, stats_text,
                                       transform=axes[plot_idx].transAxes,
                                       fontsize=9,
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                # Add overall title
                fig.suptitle(f'Saturated Signal Analysis: {file_info["patient_id"]} {file_info["run_id"]}\n'
                           f'Problem: {file_info["problem"]}',
                           fontsize=14, fontweight='bold')

                plt.tight_layout()

                # Save plot
                output_filename = output_dir / f"saturated_{file_info['patient_id']}_{file_info['run_id']}.png"
                plt.savefig(output_filename, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"  ✓ Saved plot to: {output_filename}")

        except Exception as e:
            print(f"  ✗ Error plotting {file_info['patient_id']} {file_info['run_id']}: {e}")
            continue

    print(f"\n✓ Plots saved to directory: {output_dir}/")
    print(f"  Total plots created: {min(max_plots, len(saturated_files))}")

# --- NEU: Hilfsfunktion zur Sättigungs-Detektion ---
def is_signal_saturated(sig: np.ndarray, digital_min=None, digital_max=None):
    """
    Bestimmt, ob ein ECG-Segment 'gesättigt' ist.
    Heuristik:
      - sehr wenige einzigartige Werte (hart quantisiert/flat)
      - sehr kleine Dynamik (ptp)
      - (optional) signifikanter Anteil der Samples am digitalen Min/Max (Clipping)
    Rückgabe: (bool saturated, dict metrics, str reason)
    """
    if sig is None or len(sig) == 0:
        return True, {"uniq": 0, "ptp": 0.0, "clip_frac": 0.0}, "empty segment"

    uniq = len(np.unique(sig))
    ptp = float(np.ptp(sig))
    reason_parts = []
    saturated = False

    # 1) wenige einzigartige Werte
    if uniq < 10:
        saturated = True
        reason_parts.append(f"low unique values ({uniq})")

    # 2) extrem kleine Dynamik (wie in deinem Code)
    if ptp < 1e-3:
        saturated = True
        reason_parts.append(f"low range (ptp={ptp:.3g})")

    # 3) Clipping an digitalen Grenzen (falls bekannt)
    clip_frac = 0.0
    if digital_min is not None and digital_max is not None and digital_max > digital_min:
        clip_mask = (sig <= digital_min) | (sig >= digital_max)
        clip_frac = float(np.mean(clip_mask))
        if clip_frac > 0.05:  # >5% am Rand → starkes Clipping
            saturated = True
            reason_parts.append(f"clipping {clip_frac*100:.1f}%")

    reason = ", ".join(reason_parts) if reason_parts else "not saturated"
    metrics = {"uniq": uniq, "ptp": ptp, "clip_frac": clip_frac}
    return saturated, metrics, reason


# --- NEU: Seizure-bezogene Prüfung & Plot ---
def plot_seizure_saturation(results, pre_pad=5.0, post_pad=5.0, max_plots=50):
    """
    Für jede Seizure (onset/duration) je ECG-Kanal prüfen, ob Segment gesättigt ist.
    Falls ja, Segment plotten und als PNG speichern.
    """
    out_dir = Path("seizure_saturation_plots")
    out_dir.mkdir(exist_ok=True)

    made_plots = 0
    print("\nScanning seizure segments for saturation and plotting saturated segments...")

    for patient_id, patient_data in results['patients'].items():
        for run_id, run_data in patient_data['runs'].items():
            if not run_data.get('ecg_exists') or run_data.get('seizures_annotated', 0) == 0:
                continue

            ecg_path = run_data['ecg_file']
            seizure_events = run_data.get('seizure_events', [])
            if not seizure_events:
                continue

            try:
                with pyedflib.EdfReader(ecg_path) as f:
                    labels = f.getSignalLabels()
                    ch_indices = []
                    for i, lab in enumerate(labels):
                        ll = lab.lower()
                        if any(k in ll for k in ['ecg', 'ekg', 'lead']):
                            ch_indices.append(i)
                    if not ch_indices:
                        continue

                    # Digital limits (optional; kann je Kanal unterschiedlich sein)
                    dig_min = [f.getDigitalMinimum(i) for i in range(f.signals_in_file)]
                    dig_max = [f.getDigitalMaximum(i) for i in range(f.signals_in_file)]

                    file_dur = float(f.file_duration)

                    for sz_idx, sz in enumerate(seizure_events):
                        onset = float(sz.get('onset', 0.0))
                        dur = float(sz.get('duration', 0.0))
                        label = str(sz.get('eventType', 'sz'))

                        # Fenster mit Padding, innerhalb Filegrenzen kappen
                        t0 = max(0.0, onset - pre_pad)
                        t1 = min(file_dur, onset + dur + post_pad)
                        if t1 <= t0:
                            continue

                        for ch in ch_indices:
                            fs = float(f.getSampleFrequency(ch))
                            start_sample = int(t0 * fs)
                            n_samples = int((t1 - t0) * fs)
                            if n_samples <= 0:
                                continue

                            try:
                                sig = f.readSignal(ch, start=start_sample, n=n_samples)
                            except Exception as e:
                                print(f"  readSignal failed ({patient_id} {run_id}, ch {labels[ch]}): {e}")
                                continue

                            saturated, metrics, reason = is_signal_saturated(
                                sig,
                                digital_min=dig_min[ch],
                                digital_max=dig_max[ch]
                            )
                            if not saturated:
                                continue  # nur gesättigte Segmente plotten

                            # Plot erstellen
                            t = np.arange(len(sig)) / fs + t0  # absolute Zeitachse (s)
                            fig, ax = plt.subplots(figsize=(14, 3.5))
                            ax.plot(t, sig, linewidth=0.5)
                            ax.set_xlabel("Time (s)")
                            ax.set_ylabel("Amplitude")
                            ax.set_title(
                                f"{patient_id} {run_id} | {label} #{sz_idx+1} | Channel {labels[ch]} | "
                                f"saturated: {reason} | uniq={metrics['uniq']}, "
                                f"ptp={metrics['ptp']:.4g}, clip={metrics['clip_frac']*100:.1f}%"
                            )
                            ax.grid(True, alpha=0.3)

                            # Seizure-Fenster markieren
                            ax.axvspan(onset, onset + dur, alpha=0.15, hatch='//')

                            # speichern
                            fname = out_dir / f"sat_{patient_id}_{run_id}_sz{sz_idx+1}_{labels[ch].replace(' ', '')}.png"
                            plt.tight_layout()
                            plt.savefig(fname, dpi=150, bbox_inches='tight')
                            plt.close(fig)

                            made_plots += 1
                            print(f"  ✓ Saved saturated seizure plot: {fname}")

                            if made_plots >= max_plots:
                                print(f"\nReached max_plots={max_plots}. Stopping.")
                                return

            except Exception as e:
                print(f"  ✗ Error processing {patient_id} {run_id} for seizure plots: {e}")

    print(f"\n✓ Done. Created {made_plots} saturated seizure plot(s) in '{out_dir}/'.")


def main():
    print("Analyzing ds005873-1.1.0_example data...")
    print("="*60)

    try:
        results = analyze_example_data()
        print_summary_report(results)

        # Save detailed results to JSON
        json_filename = 'example_data_analysis.json'
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Detailed results saved to '{json_filename}'")

        # Generate and save problems report
        print("\nGenerating problems report...")
        problems_report = generate_problems_report(results)

        report_filename = 'example_data_problems_report.txt'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(problems_report)

        print(f"✓ Problems report saved to '{report_filename}'")

        # Show quick preview of problems found
        problems_count = sum(1 for patient_data in results['patients'].values()
                           for run_data in patient_data['runs'].values()
                           if not run_data.get('usable'))
        print(f"\nTotal problems found: {problems_count}")
        print(f"See '{report_filename}' for detailed problem listing.")

        # Plot saturated signals for visual inspection
        plot_saturated_signals(results, max_plots=10)
        # Plot nur für SEIZURE-SEGMENTE, falls diese gesättigt sind
        plot_seizure_saturation(results, pre_pad=5.0, post_pad=3.0, max_plots=50)


    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()