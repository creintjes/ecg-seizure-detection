#!/usr/bin/env python3
"""
Analyze how many seizures would be lost by excluding test patients (sub-097 to sub-125)
with saturation >= threshold.

This script:
1. Loads saturation analysis results (JSON from analyze_saturation_by_segments.py)
2. Loads seizure annotations from event files
3. Determines which seizures would be lost at different saturation thresholds
4. Generates comprehensive report with statistics

Usage:
    python analyze_seizure_loss_from_saturation.py <saturation_json> [threshold]

Example:
    python analyze_seizure_loss_from_saturation.py saturation_analysis_300s_20251016.json 10
"""

import json
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from collections import defaultdict


def parse_run_id(name: str) -> str:
    """Extract run-ID as 'run-XX' (two digits)."""
    m = re.search(r"[_-]run-(\d+)(?=[_.]|$)", name)
    if m:
        return f"run-{int(m.group(1)):02d}"
    base = Path(name).stem
    return f"run-{hash(base) & 0xffff:04x}"


def count_seizures_in_events_file(events_file: Path) -> tuple:
    """
    Read *_events.tsv and count seizures in SeizeIT2 format.
    Returns (seizure_count, seizure_events_list)
    """
    try:
        df = pd.read_csv(events_file, sep="\t")
        if "eventType" not in df.columns:
            return 0, []

        # In SeizeIT2, seizures start with 'sz_' and exclude 'bckg' and 'impd'
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


def load_saturation_results(json_path: Path) -> dict:
    """Load the saturation analysis JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_seizure_annotations(base_dir: Path,
                            start_patient: int = 97,
                            end_patient: int = 125) -> dict:
    """
    Load seizure annotations for test patients from event files.

    Returns:
        dict: {patient_id: {run_id: {count, events}}}
    """
    seizure_data = {}

    if not base_dir.exists():
        print(f"Warning: Base directory {base_dir} not found")
        return seizure_data

    # Get test patients only
    subjects = [d for d in base_dir.iterdir()
                if d.is_dir() and d.name.startswith('sub-')]

    for subject in sorted(subjects):
        subject_id = subject.name

        # Extract patient number
        patient_num = int(subject_id.split('-')[1])

        # Only include test patients
        if not (start_patient <= patient_num <= end_patient):
            continue

        seizure_data[subject_id] = {}

        # Get sessions (typically ses-01)
        sessions = [d for d in subject.iterdir()
                   if d.is_dir() and d.name.startswith('ses-')]

        for session in sessions:
            eeg_dir = session / "eeg"

            if not eeg_dir.exists():
                continue

            # Get all event files
            event_files = list(eeg_dir.glob("*_events.tsv"))

            for event_file in sorted(event_files):
                run_id = parse_run_id(event_file.name)

                seizure_count, seizure_events = count_seizures_in_events_file(event_file)

                if seizure_count > 0:
                    seizure_data[subject_id][run_id] = {
                        'count': seizure_count,
                        'events': seizure_events,
                        'events_file': str(event_file)
                    }

    return seizure_data


def analyze_seizure_loss(saturation_results: dict,
                         seizure_data: dict,
                         threshold: float = 10.0,
                         start_patient: int = 97,
                         end_patient: int = 125) -> dict:
    """
    Analyze seizure loss at given threshold.

    Args:
        saturation_results: Saturation analysis results
        seizure_data: Seizure annotation data
        threshold: Saturation percentage threshold
        start_patient: First patient in test set
        end_patient: Last patient in test set

    Returns:
        dict: Analysis results with statistics
    """
    analysis = {
        'threshold': threshold,
        'total_test_patients': 0,
        'total_test_runs': 0,
        'total_seizures': 0,
        'excluded_runs': 0,
        'excluded_runs_with_seizures': 0,
        'lost_seizures': 0,
        'retained_seizures': 0,
        'excluded_patients': set(),
        'patients_losing_all_seizures': [],
        'detailed_losses': []
    }

    # Count total seizures in test set
    for patient_id, runs in seizure_data.items():
        patient_num = int(patient_id.split('-')[1])
        if not (start_patient <= patient_num <= end_patient):
            continue

        analysis['total_test_patients'] += 1

        for run_id, run_seizures in runs.items():
            analysis['total_test_runs'] += 1
            analysis['total_seizures'] += run_seizures['count']

    # Analyze losses
    for patient_id, patient_data in saturation_results['patients'].items():
        # Extract patient number
        patient_num = int(patient_id.split('-')[1])

        # Only include test patients
        if not (start_patient <= patient_num <= end_patient):
            continue

        patient_seizures_lost = 0
        patient_seizures_retained = 0

        for run_id, recording_data in patient_data['recordings'].items():
            # Skip runs with errors
            if recording_data.get('error'):
                continue

            sat_pct = recording_data['saturation_percentage']

            # Check if this run has seizures
            run_has_seizures = (patient_id in seizure_data and
                              run_id in seizure_data[patient_id])

            if run_has_seizures:
                seizure_count = seizure_data[patient_id][run_id]['count']
                seizure_events = seizure_data[patient_id][run_id]['events']

                if sat_pct >= threshold:
                    # This run would be excluded
                    analysis['excluded_runs'] += 1
                    analysis['excluded_runs_with_seizures'] += 1
                    analysis['lost_seizures'] += seizure_count
                    patient_seizures_lost += seizure_count
                    analysis['excluded_patients'].add(patient_id)

                    # Record detailed loss
                    analysis['detailed_losses'].append({
                        'patient_id': patient_id,
                        'run_id': run_id,
                        'saturation_percentage': sat_pct,
                        'seizure_count': seizure_count,
                        'seizure_events': seizure_events,
                        'total_segments': recording_data['total_segments'],
                        'saturated_segments': recording_data['overall_saturated_segments']
                    })
                else:
                    # This run would be retained
                    analysis['retained_seizures'] += seizure_count
                    patient_seizures_retained += seizure_count
            else:
                # Run has no seizures, but still count if excluded
                if sat_pct >= threshold:
                    analysis['excluded_runs'] += 1
                    analysis['excluded_patients'].add(patient_id)

        # Check if patient loses all their seizures
        if patient_id in seizure_data:
            patient_total_seizures = sum(r['count'] for r in seizure_data[patient_id].values())
            if patient_total_seizures > 0 and patient_seizures_retained == 0:
                analysis['patients_losing_all_seizures'].append({
                    'patient_id': patient_id,
                    'total_seizures': patient_total_seizures,
                    'lost_seizures': patient_seizures_lost
                })

    return analysis


def generate_report(analysis: dict, seizure_data: dict) -> str:
    """Generate comprehensive report about seizure losses."""
    lines = []

    lines.append("="*80)
    lines.append(f"SEIZURE LOSS ANALYSIS - TEST SET (sub-097 to sub-125)")
    lines.append("="*80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Saturation Threshold: {analysis['threshold']}%")
    lines.append("")

    # Summary statistics
    lines.append("OVERALL STATISTICS:")
    lines.append("-"*80)
    lines.append(f"  Test patients: {analysis['total_test_patients']}")
    lines.append(f"  Test runs: {analysis['total_test_runs']}")
    lines.append(f"  Total seizures in test set: {analysis['total_seizures']}")
    lines.append("")

    # Exclusion impact
    lines.append("EXCLUSION IMPACT:")
    lines.append("-"*80)
    lines.append(f"  Runs excluded (saturation ≥ {analysis['threshold']}%): "
                f"{analysis['excluded_runs']} "
                f"({analysis['excluded_runs']/analysis['total_test_runs']*100:.1f}% of test runs)")
    lines.append(f"  Excluded runs containing seizures: {analysis['excluded_runs_with_seizures']}")
    lines.append(f"  Patients affected by exclusion: {len(analysis['excluded_patients'])}")
    lines.append("")

    # Seizure losses
    lines.append("SEIZURE LOSSES:")
    lines.append("-"*80)
    lines.append(f"  Seizures LOST due to exclusion: {analysis['lost_seizures']} "
                f"({analysis['lost_seizures']/analysis['total_seizures']*100:.1f}% of total)")
    lines.append(f"  Seizures RETAINED: {analysis['retained_seizures']} "
                f"({analysis['retained_seizures']/analysis['total_seizures']*100:.1f}% of total)")
    lines.append("")

    # Patients losing all seizures
    if analysis['patients_losing_all_seizures']:
        lines.append(f"⚠️  CRITICAL: {len(analysis['patients_losing_all_seizures'])} patients "
                    f"would lose ALL their seizures!")
        lines.append("-"*80)
        for patient in sorted(analysis['patients_losing_all_seizures'],
                            key=lambda x: x['total_seizures'], reverse=True):
            lines.append(f"  {patient['patient_id']}: "
                        f"{patient['total_seizures']} seizure(s) lost")
        lines.append("")

    lines.append("")

    # Detailed breakdown
    if analysis['detailed_losses']:
        lines.append("DETAILED LOSS BREAKDOWN:")
        lines.append("-"*80)
        lines.append(f"{'Patient':<12} {'Run':<10} {'Saturation':<12} {'Seizures':<10} "
                    f"{'Segments':<20}")
        lines.append("-"*80)

        for loss in sorted(analysis['detailed_losses'],
                          key=lambda x: (x['patient_id'], x['run_id'])):
            lines.append(
                f"{loss['patient_id']:<12} {loss['run_id']:<10} "
                f"{loss['saturation_percentage']:>10.1f}%   "
                f"{loss['seizure_count']:>8}   "
                f"{loss['saturated_segments']:>6}/{loss['total_segments']:<10}"
            )

        lines.append("")

    lines.append("")

    # Seizure type breakdown (if available)
    lines.append("LOST SEIZURES BY TYPE:")
    lines.append("-"*80)

    seizure_types = defaultdict(int)
    for loss in analysis['detailed_losses']:
        for event in loss['seizure_events']:
            event_type = event.get('eventType', 'unknown')
            seizure_types[event_type] += 1

    if seizure_types:
        for sz_type, count in sorted(seizure_types.items(),
                                    key=lambda x: x[1], reverse=True):
            lines.append(f"  {sz_type}: {count}")
    else:
        lines.append("  No seizure type information available")

    lines.append("")
    lines.append("")

    # Patient-level summary
    lines.append("PATIENT-LEVEL SUMMARY:")
    lines.append("-"*80)

    patient_summary = defaultdict(lambda: {'total': 0, 'lost': 0, 'retained': 0})

    for loss in analysis['detailed_losses']:
        patient_id = loss['patient_id']
        patient_summary[patient_id]['lost'] += loss['seizure_count']

    # Add patients with retained seizures
    for patient_id, runs in seizure_data.items():
        patient_num = int(patient_id.split('-')[1])
        if not (97 <= patient_num <= 125):
            continue

        for run_id, run_data in runs.items():
            patient_summary[patient_id]['total'] += run_data['count']

    for patient_id in patient_summary:
        patient_summary[patient_id]['retained'] = (
            patient_summary[patient_id]['total'] -
            patient_summary[patient_id]['lost']
        )

    lines.append(f"{'Patient':<12} {'Total':<8} {'Lost':<8} {'Retained':<10} {'Loss Rate':<12}")
    lines.append("-"*80)

    for patient_id in sorted(patient_summary.keys()):
        stats = patient_summary[patient_id]
        loss_rate = (stats['lost'] / stats['total'] * 100) if stats['total'] > 0 else 0
        lines.append(
            f"{patient_id:<12} {stats['total']:>6}   {stats['lost']:>6}   "
            f"{stats['retained']:>8}   {loss_rate:>10.1f}%"
        )

    lines.append("")
    lines.append("="*80)
    lines.append("END OF REPORT")
    lines.append("="*80)

    return "\n".join(lines)


def generate_threshold_comparison(saturation_results: dict,
                                 seizure_data: dict,
                                 thresholds: list = None) -> str:
    """Generate comparison report across multiple thresholds."""
    if thresholds is None:
        thresholds = [5, 10, 15, 20, 25, 30, 40, 50]

    lines = []
    lines.append("="*80)
    lines.append("THRESHOLD COMPARISON: SEIZURE LOSS ANALYSIS")
    lines.append("="*80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Analyze at each threshold
    results = []
    for threshold in thresholds:
        analysis = analyze_seizure_loss(saturation_results, seizure_data, threshold)
        results.append(analysis)

    # Table header
    lines.append(f"{'Threshold':<12} {'Runs':<12} {'Runs w/':<12} {'Patients':<12} "
                f"{'Seizures':<12} {'Loss':<10}")
    lines.append(f"{'(%)':<12} {'Excluded':<12} {'Seizures':<12} {'Lose All':<12} "
                f"{'Lost':<12} {'Rate (%)':<10}")
    lines.append("-"*80)

    for analysis in results:
        threshold = analysis['threshold']
        excluded_runs = analysis['excluded_runs']
        excluded_with_sz = analysis['excluded_runs_with_seizures']
        patients_lose_all = len(analysis['patients_losing_all_seizures'])
        lost_seizures = analysis['lost_seizures']
        total_seizures = analysis['total_seizures']
        loss_rate = (lost_seizures / total_seizures * 100) if total_seizures > 0 else 0

        lines.append(
            f"{threshold:>10.0f}%  {excluded_runs:>10}   {excluded_with_sz:>10}   "
            f"{patients_lose_all:>10}   {lost_seizures:>10}   {loss_rate:>8.1f}%"
        )

    lines.append("")
    lines.append("")

    # Detailed statistics at each threshold
    lines.append("DETAILED STATISTICS BY THRESHOLD:")
    lines.append("="*80)

    for analysis in results:
        lines.append("")
        lines.append(f"Threshold: {analysis['threshold']}%")
        lines.append("-"*40)
        lines.append(f"  Excluded runs: {analysis['excluded_runs']}")
        lines.append(f"  Lost seizures: {analysis['lost_seizures']} / {analysis['total_seizures']} "
                    f"({analysis['lost_seizures']/analysis['total_seizures']*100:.1f}%)")
        lines.append(f"  Patients losing all seizures: {len(analysis['patients_losing_all_seizures'])}")

        if analysis['patients_losing_all_seizures']:
            patient_ids = [p['patient_id'] for p in analysis['patients_losing_all_seizures']]
            lines.append(f"    {', '.join(patient_ids)}")

    lines.append("")
    lines.append("="*80)
    lines.append("END OF COMPARISON")
    lines.append("="*80)

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_seizure_loss_from_saturation.py <saturation_json> [threshold]")
        print("\nExample:")
        print("  python analyze_seizure_loss_from_saturation.py saturation_analysis_300s_20251016.json 10")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    threshold = float(sys.argv[2]) if len(sys.argv) >= 3 else 10.0

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Base directory for dataset
    base_dir = Path("/home/swolf/asim_shared/raw_data/ds005873-1.1.0")

    print("="*80)
    print("ANALYZING SEIZURE LOSS FROM SATURATION-BASED EXCLUSION")
    print("="*80)
    print(f"Saturation analysis: {json_path}")
    print(f"Threshold: {threshold}%")
    print(f"Dataset: {base_dir}")
    print("")

    # Load saturation results
    print("Loading saturation analysis results...")
    saturation_results = load_saturation_results(json_path)
    print(f"  ✓ Loaded saturation data for {len(saturation_results['patients'])} patients")

    # Load seizure annotations
    print("\nLoading seizure annotations for test patients...")
    seizure_data = load_seizure_annotations(base_dir)

    total_test_patients = len(seizure_data)
    total_seizures = sum(
        sum(r['count'] for r in patient_runs.values())
        for patient_runs in seizure_data.values()
    )
    print(f"  ✓ Loaded seizure data for {total_test_patients} test patients")
    print(f"  ✓ Found {total_seizures} total seizures in test set")

    # Analyze seizure loss
    print(f"\nAnalyzing seizure loss at {threshold}% threshold...")
    analysis = analyze_seizure_loss(saturation_results, seizure_data, threshold)

    # Generate main report
    print("\nGenerating report...")
    report = generate_report(analysis, seizure_data)

    # Save report
    output_dir = json_path.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = output_dir / f'seizure_loss_analysis_threshold{int(threshold)}pct_{timestamp}.txt'

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Report saved to: {report_filename}")

    # Generate threshold comparison
    print("\nGenerating threshold comparison...")
    comparison = generate_threshold_comparison(saturation_results, seizure_data)

    comparison_filename = output_dir / f'seizure_loss_threshold_comparison_{timestamp}.txt'
    with open(comparison_filename, 'w', encoding='utf-8') as f:
        f.write(comparison)

    print(f"✓ Comparison saved to: {comparison_filename}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Threshold: {threshold}%")
    print(f"Total seizures in test set: {analysis['total_seizures']}")
    print(f"Seizures LOST: {analysis['lost_seizures']} "
          f"({analysis['lost_seizures']/analysis['total_seizures']*100:.1f}%)")
    print(f"Seizures RETAINED: {analysis['retained_seizures']} "
          f"({analysis['retained_seizures']/analysis['total_seizures']*100:.1f}%)")
    print(f"Patients losing all seizures: {len(analysis['patients_losing_all_seizures'])}")

    if analysis['patients_losing_all_seizures']:
        print("\nPatients losing all seizures:")
        for patient in analysis['patients_losing_all_seizures']:
            print(f"  - {patient['patient_id']}: {patient['total_seizures']} seizure(s)")

    print(f"\n✓ Analysis complete!")
    print(f"See '{report_filename}' for detailed report.")
    print(f"See '{comparison_filename}' for threshold comparison.")


if __name__ == "__main__":
    main()
