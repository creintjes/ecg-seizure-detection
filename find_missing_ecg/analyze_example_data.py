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
                    sample_size = min(10000, int(f.getSampleFrequency(channel_idx) * 30))
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

def analyze_example_data():
    base_dir = Path("../ds005873-1.1.0_example")

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

            if problem['details'].get('signal_quality') != 'unknown':
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
    no_channels = sum(1 for p in problems if p['problem_type'] == 'No ECG Channels')
    if no_channels > 0:
        report_lines.append(f"2. NO ECG CHANNELS ({no_channels} files):")
        report_lines.append("   - Verify that ECG data was recorded for these sessions")
        report_lines.append("   - Check if different channel naming conventions are used")
        report_lines.append("")

    zero_signal = sum(1 for p in problems if p['problem_type'] == 'Zero/Empty Signal')
    if zero_signal > 0:
        report_lines.append(f"3. ZERO/EMPTY SIGNALS ({zero_signal} files):")
        report_lines.append("   - These files may be corrupted during recording or transfer")
        report_lines.append("   - Consider excluding from analysis")
        report_lines.append("")

    saturated = sum(1 for p in problems if p['problem_type'] == 'Saturated Signal')
    if saturated > 0:
        report_lines.append(f"4. SATURATED SIGNALS ({saturated} files):")
        report_lines.append("   - Check recording equipment calibration")
        report_lines.append("   - May indicate amplifier saturation or disconnected electrodes")
        report_lines.append("")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    return "\n".join(report_lines)

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

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()