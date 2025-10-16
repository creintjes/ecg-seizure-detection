#!/usr/bin/env python3
"""
Analyze saturation results specifically for test patients (sub-097 to sub-125).
Reads the JSON output from analyze_saturation_by_segments.py and generates
a separate report for the test set.

Usage:
    python analyze_test_patients_saturation.py <path_to_json_file>
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys


def load_saturation_results(json_path: Path) -> dict:
    """Load the saturation analysis JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def filter_test_patients(results: dict, start_patient: int = 97, end_patient: int = 125) -> dict:
    """
    Filter results to only include test patients (sub-097 to sub-125).

    Args:
        results: Full analysis results
        start_patient: First patient number in test set (default: 97)
        end_patient: Last patient number in test set (default: 125)

    Returns:
        Filtered results containing only test patients
    """
    test_results = {
        'metadata': results['metadata'].copy(),
        'patients': {}
    }

    # Update metadata
    test_results['metadata']['filter'] = f'Test patients: sub-{start_patient:03d} to sub-{end_patient:03d}'
    test_results['metadata']['original_total_patients'] = results['metadata']['total_patients']

    # Filter patients
    for patient_id, patient_data in results['patients'].items():
        # Extract patient number (e.g., "sub-097" -> 97)
        patient_num = int(patient_id.split('-')[1])

        if start_patient <= patient_num <= end_patient:
            test_results['patients'][patient_id] = patient_data

    # Update counters
    test_results['metadata']['total_patients'] = len(test_results['patients'])
    test_results['metadata']['total_recordings'] = sum(
        len(p['recordings']) for p in test_results['patients'].values()
    )
    test_results['metadata']['recordings_analyzed'] = sum(
        1 for p in test_results['patients'].values()
        for r in p['recordings'].values()
        if not r.get('error')
    )
    test_results['metadata']['recordings_failed'] = sum(
        1 for p in test_results['patients'].values()
        for r in p['recordings'].values()
        if r.get('error')
    )

    return test_results


def generate_test_set_report(test_results: dict) -> str:
    """
    Generate a detailed report for test set patients.

    Args:
        test_results: Filtered results for test patients

    Returns:
        Formatted report text
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SATURATION ANALYSIS - TEST SET PATIENTS (sub-097 to sub-125)")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Original analysis: {test_results['metadata']['timestamp']}")
    report_lines.append(f"Segment Duration: {test_results['metadata']['segment_duration']}s")
    report_lines.append("")

    # Summary statistics
    report_lines.append("SUMMARY STATISTICS:")
    report_lines.append("-"*80)
    report_lines.append(f"  Test patients: {test_results['metadata']['total_patients']}")
    report_lines.append(f"  Total recordings: {test_results['metadata']['total_recordings']}")
    report_lines.append(f"  Successfully analyzed: {test_results['metadata']['recordings_analyzed']}")
    report_lines.append(f"  Failed: {test_results['metadata']['recordings_failed']}")
    report_lines.append("")

    # Collect all recordings with their saturation percentages
    recordings = []
    for patient_id, patient_data in test_results['patients'].items():
        for run_id, recording_data in patient_data['recordings'].items():
            if not recording_data.get('error'):
                recordings.append({
                    'patient_id': patient_id,
                    'run_id': run_id,
                    'saturation_percentage': recording_data['saturation_percentage'],
                    'total_segments': recording_data['total_segments'],
                    'saturated_segments': recording_data['overall_saturated_segments'],
                    'usable_segments': recording_data['overall_usable_segments'],
                    'file_duration': recording_data['file_duration']
                })

    # Distribution statistics
    if recordings:
        sat_percentages = [r['saturation_percentage'] for r in recordings]
        report_lines.append("SATURATION DISTRIBUTION (TEST SET):")
        report_lines.append("-"*80)
        report_lines.append(f"  Mean saturation: {np.mean(sat_percentages):.1f}%")
        report_lines.append(f"  Median saturation: {np.median(sat_percentages):.1f}%")
        report_lines.append(f"  Std deviation: {np.std(sat_percentages):.1f}%")
        report_lines.append(f"  Min saturation: {np.min(sat_percentages):.1f}%")
        report_lines.append(f"  Max saturation: {np.max(sat_percentages):.1f}%")
        report_lines.append("")

        # Percentiles
        report_lines.append("  Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            report_lines.append(f"    {p}th percentile: {np.percentile(sat_percentages, p):.1f}%")

    report_lines.append("")
    report_lines.append("")

    # Threshold analysis for test set
    thresholds = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    total_recordings = len(recordings)

    report_lines.append("THRESHOLD ANALYSIS (TEST SET):")
    report_lines.append("-"*80)
    report_lines.append(f"{'Threshold':<12} {'Excluded':<12} {'Remaining':<12} {'Exclusion Rate':<20}")
    report_lines.append("-"*80)

    for threshold in thresholds:
        excluded = sum(1 for r in recordings if r['saturation_percentage'] >= threshold)
        remaining = total_recordings - excluded
        exclusion_rate = (excluded / total_recordings * 100) if total_recordings > 0 else 0

        report_lines.append(f"{threshold:>10}% {excluded:>10}   {remaining:>10}   {exclusion_rate:>18.1f}%")

    report_lines.append("")
    report_lines.append("")

    # List all test set recordings with their saturation levels
    report_lines.append("DETAILED LISTING (TEST SET):")
    report_lines.append("-"*80)
    report_lines.append(f"{'Patient':<12} {'Run':<10} {'Duration':<12} {'Segments':<10} "
                       f"{'Saturated':<12} {'Saturation %':<15}")
    report_lines.append("-"*80)

    for rec in sorted(recordings, key=lambda x: (x['patient_id'], x['run_id'])):
        duration_hrs = rec['file_duration'] / 3600
        report_lines.append(
            f"{rec['patient_id']:<12} {rec['run_id']:<10} {duration_hrs:>9.2f}h   "
            f"{rec['total_segments']:>8}   {rec['saturated_segments']:>10}   "
            f"{rec['saturation_percentage']:>13.1f}%"
        )

    report_lines.append("")
    report_lines.append("")

    # Highly usable recordings (< 10% saturation)
    highly_usable = [r for r in recordings if r['saturation_percentage'] < 10]
    if highly_usable:
        report_lines.append(f"HIGHLY USABLE RECORDINGS (<10% saturation): {len(highly_usable)}")
        report_lines.append("-"*80)
        for rec in sorted(highly_usable, key=lambda x: x['saturation_percentage']):
            report_lines.append(f"  {rec['patient_id']} {rec['run_id']}: "
                              f"{rec['saturation_percentage']:.1f}% saturated "
                              f"({rec['usable_segments']}/{rec['total_segments']} usable segments)")

    report_lines.append("")
    report_lines.append("")

    # Moderately saturated recordings (10-50%)
    moderate_sat = [r for r in recordings if 10 <= r['saturation_percentage'] < 50]
    if moderate_sat:
        report_lines.append(f"MODERATELY SATURATED RECORDINGS (10-50%): {len(moderate_sat)}")
        report_lines.append("-"*80)
        for rec in sorted(moderate_sat, key=lambda x: x['saturation_percentage']):
            report_lines.append(f"  {rec['patient_id']} {rec['run_id']}: "
                              f"{rec['saturation_percentage']:.1f}% saturated "
                              f"({rec['usable_segments']}/{rec['total_segments']} usable segments)")

    report_lines.append("")
    report_lines.append("")

    # Heavily saturated recordings (>50% saturated)
    heavily_saturated = [r for r in recordings if r['saturation_percentage'] >= 50]
    if heavily_saturated:
        report_lines.append(f"HEAVILY SATURATED RECORDINGS (≥50%): {len(heavily_saturated)}")
        report_lines.append("-"*80)
        for rec in sorted(heavily_saturated, key=lambda x: x['saturation_percentage'], reverse=True):
            report_lines.append(f"  {rec['patient_id']} {rec['run_id']}: "
                              f"{rec['saturation_percentage']:.1f}% saturated "
                              f"({rec['saturated_segments']}/{rec['total_segments']} segments)")

    report_lines.append("")

    # Patient-level summary
    report_lines.append("")
    report_lines.append("PATIENT-LEVEL SUMMARY:")
    report_lines.append("-"*80)
    report_lines.append(f"{'Patient':<12} {'Recordings':<12} {'Avg Saturation':<18} {'Min-Max':<20}")
    report_lines.append("-"*80)

    patient_summaries = {}
    for rec in recordings:
        pid = rec['patient_id']
        if pid not in patient_summaries:
            patient_summaries[pid] = []
        patient_summaries[pid].append(rec['saturation_percentage'])

    for patient_id in sorted(patient_summaries.keys()):
        sat_values = patient_summaries[patient_id]
        avg_sat = np.mean(sat_values)
        min_sat = np.min(sat_values)
        max_sat = np.max(sat_values)
        num_recordings = len(sat_values)

        report_lines.append(
            f"{patient_id:<12} {num_recordings:>10}   {avg_sat:>15.1f}%   "
            f"{min_sat:>7.1f}% - {max_sat:<7.1f}%"
        )

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF TEST SET REPORT")
    report_lines.append("="*80)

    return "\n".join(report_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_test_patients_saturation.py <path_to_saturation_json>")
        print("\nExample:")
        print("  python analyze_test_patients_saturation.py find_missing_ecg/saturation_analysis_300s_20251016_220030.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print("="*80)
    print("TEST SET SATURATION ANALYSIS")
    print("="*80)
    print(f"Loading results from: {json_path}")

    # Load full results
    results = load_saturation_results(json_path)

    # Filter to test patients
    print("Filtering test patients (sub-097 to sub-125)...")
    test_results = filter_test_patients(results)

    print(f"Found {test_results['metadata']['total_patients']} test patients "
          f"with {test_results['metadata']['total_recordings']} recordings")

    # Generate report
    print("\nGenerating test set report...")
    report = generate_test_set_report(test_results)

    # Save report
    output_dir = json_path.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    segment_duration = int(test_results['metadata']['segment_duration'])
    report_filename = output_dir / f'test_set_saturation_report_{segment_duration}s_{timestamp}.txt'

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Report saved to: {report_filename}")

    # Save filtered JSON for test set only
    json_filename = output_dir / f'test_set_saturation_analysis_{segment_duration}s_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"✓ Test set JSON saved to: {json_filename}")

    # Print summary to console
    print("\n" + "="*80)
    print("TEST SET SUMMARY")
    print("="*80)
    print(f"Test patients: {test_results['metadata']['total_patients']}")
    print(f"Total recordings: {test_results['metadata']['total_recordings']}")
    print(f"Successfully analyzed: {test_results['metadata']['recordings_analyzed']}")
    print(f"Failed: {test_results['metadata']['recordings_failed']}")

    # Quick threshold preview
    recordings = []
    for patient_data in test_results['patients'].values():
        for recording_data in patient_data['recordings'].values():
            if not recording_data.get('error'):
                recordings.append(recording_data['saturation_percentage'])

    if recordings:
        print("\nQuick threshold preview for TEST SET:")
        for threshold in [10, 20, 30, 50]:
            excluded = sum(1 for s in recordings if s >= threshold)
            print(f"  At {threshold}% threshold: {excluded} runs excluded "
                  f"({excluded/len(recordings)*100:.1f}% of test set)")

    print("\n✓ Test set analysis complete!")
    print(f"See '{report_filename}' for detailed report.")


if __name__ == "__main__":
    main()
