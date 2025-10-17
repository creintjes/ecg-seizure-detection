#!/usr/bin/env python3
"""
Extract test patients (sub-097 to sub-125) with saturation >= threshold
and generate a regex-friendly list for filtering.

Usage:
    python extract_saturated_test_patients.py <json_file> [threshold]

Example:
    python extract_saturated_test_patients.py saturation_analysis_300s_20251016_220030.json 10
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_saturation_results(json_path: Path) -> dict:
    """Load the saturation analysis JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_saturated_runs(results: dict,
                          threshold: float = 10.0,
                          start_patient: int = 97,
                          end_patient: int = 125) -> list:
    """
    Extract test patient runs with saturation >= threshold.

    Args:
        results: Full analysis results
        threshold: Saturation percentage threshold (default: 10%)
        start_patient: First patient number in test set (default: 97)
        end_patient: Last patient number in test set (default: 125)

    Returns:
        List of dicts with patient_id, run_id, and saturation_percentage
    """
    saturated_runs = []

    for patient_id, patient_data in results['patients'].items():
        # Extract patient number
        patient_num = int(patient_id.split('-')[1])

        # Only include test patients
        if not (start_patient <= patient_num <= end_patient):
            continue

        # Check each run
        for run_id, recording_data in patient_data['recordings'].items():
            if recording_data.get('error'):
                continue

            sat_pct = recording_data['saturation_percentage']
            if sat_pct >= threshold:
                saturated_runs.append({
                    'patient_id': patient_id,
                    'run_id': run_id,
                    'saturation_percentage': sat_pct,
                    'total_segments': recording_data['total_segments'],
                    'saturated_segments': recording_data['overall_saturated_segments']
                })

    return saturated_runs


def generate_report(saturated_runs: list, threshold: float) -> str:
    """Generate a formatted report with multiple output formats."""
    lines = []

    lines.append("="*80)
    lines.append(f"SATURATED TEST PATIENTS (>={threshold}% saturation)")
    lines.append("="*80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Threshold: {threshold}%")
    lines.append(f"Total runs found: {len(saturated_runs)}")
    lines.append("")

    if not saturated_runs:
        lines.append("No runs found meeting the threshold criteria.")
        lines.append("="*80)
        return "\n".join(lines)

    # Sort by patient, then run
    saturated_runs = sorted(saturated_runs, key=lambda x: (x['patient_id'], x['run_id']))

    # ========================================================================
    # Format 1: Detailed List
    # ========================================================================
    lines.append("DETAILED LIST:")
    lines.append("-"*80)
    lines.append(f"{'Patient':<12} {'Run':<10} {'Saturation':<15} {'Segments':<20}")
    lines.append("-"*80)

    for run in saturated_runs:
        lines.append(
            f"{run['patient_id']:<12} {run['run_id']:<10} "
            f"{run['saturation_percentage']:>12.1f}%   "
            f"{run['saturated_segments']:>6}/{run['total_segments']:<6} saturated"
        )

    lines.append("")
    lines.append("")

    # ========================================================================
    # Format 2: Simple Patient/Run List (one per line)
    # ========================================================================
    lines.append("SIMPLE LIST (one per line):")
    lines.append("-"*80)
    for run in saturated_runs:
        lines.append(f"{run['patient_id']} {run['run_id']}")

    lines.append("")
    lines.append("")

    # ========================================================================
    # Format 3: Regex Pattern (for filtering)
    # ========================================================================
    lines.append("REGEX PATTERN FOR FILTERING:")
    lines.append("-"*80)
    lines.append("Use this pattern to match any of these patient/run combinations:")
    lines.append("")

    # Create regex pattern: (sub-097_run-01|sub-097_run-05|...)
    pattern_parts = [f"{run['patient_id']}_{run['run_id']}" for run in saturated_runs]
    regex_pattern = f"({'|'.join(pattern_parts)})"

    lines.append(regex_pattern)
    lines.append("")
    lines.append("")

    # ========================================================================
    # Format 4: Bash-friendly List (for grep -v, etc.)
    # ========================================================================
    lines.append("BASH GREP PATTERN (exclude these runs):")
    lines.append("-"*80)
    lines.append("Use with: grep -vE '<pattern>' file.txt")
    lines.append("")

    # For grep -E, we can use the same pattern
    lines.append(regex_pattern)
    lines.append("")
    lines.append("")

    # ========================================================================
    # Format 5: Python List
    # ========================================================================
    lines.append("PYTHON LIST:")
    lines.append("-"*80)
    lines.append("saturated_runs = [")
    for run in saturated_runs:
        lines.append(f'    ("{run["patient_id"]}", "{run["run_id"]}"),  # {run["saturation_percentage"]:.1f}%')
    lines.append("]")
    lines.append("")
    lines.append("")

    # ========================================================================
    # Format 6: Patient-level Summary
    # ========================================================================
    lines.append("PATIENT-LEVEL SUMMARY:")
    lines.append("-"*80)

    patient_runs = {}
    for run in saturated_runs:
        pid = run['patient_id']
        if pid not in patient_runs:
            patient_runs[pid] = []
        patient_runs[pid].append((run['run_id'], run['saturation_percentage']))

    for patient_id in sorted(patient_runs.keys()):
        runs = patient_runs[patient_id]
        lines.append(f"{patient_id}: {len(runs)} run(s)")
        for run_id, sat_pct in sorted(runs):
            lines.append(f"  - {run_id}: {sat_pct:.1f}%")
        lines.append("")

    # ========================================================================
    # Statistics
    # ========================================================================
    lines.append("")
    lines.append("STATISTICS:")
    lines.append("-"*80)
    lines.append(f"Total runs affected: {len(saturated_runs)}")
    lines.append(f"Total patients affected: {len(patient_runs)}")

    sat_values = [run['saturation_percentage'] for run in saturated_runs]
    lines.append(f"Average saturation: {sum(sat_values)/len(sat_values):.1f}%")
    lines.append(f"Min saturation: {min(sat_values):.1f}%")
    lines.append(f"Max saturation: {max(sat_values):.1f}%")

    lines.append("")
    lines.append("="*80)
    lines.append("END OF REPORT")
    lines.append("="*80)

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_saturated_test_patients.py <json_file> [threshold]")
        print("\nExample:")
        print("  python extract_saturated_test_patients.py saturation_analysis_300s_20251016.json 10")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    threshold = float(sys.argv[2]) if len(sys.argv) >= 3 else 10.0

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print("="*80)
    print("EXTRACTING SATURATED TEST PATIENTS")
    print("="*80)
    print(f"Input file: {json_path}")
    print(f"Threshold: {threshold}%")
    print("")

    # Load results
    print("Loading saturation analysis results...")
    results = load_saturation_results(json_path)

    # Extract saturated runs
    print(f"Extracting test patients (sub-097 to sub-125) with saturation >= {threshold}%...")
    saturated_runs = extract_saturated_runs(results, threshold=threshold)

    print(f"Found {len(saturated_runs)} runs meeting criteria")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(saturated_runs, threshold)

    # Save report
    output_dir = json_path.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = output_dir / f'saturated_test_patients_threshold{int(threshold)}pct_{timestamp}.txt'

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✓ Report saved to: {report_filename}")

    # Print summary to console
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)

    if saturated_runs:
        print(f"Total runs with >={threshold}% saturation: {len(saturated_runs)}")

        # Show first few runs
        print("\nFirst 10 runs:")
        for run in sorted(saturated_runs, key=lambda x: (x['patient_id'], x['run_id']))[:10]:
            print(f"  {run['patient_id']} {run['run_id']}: {run['saturation_percentage']:.1f}%")

        if len(saturated_runs) > 10:
            print(f"  ... and {len(saturated_runs) - 10} more")

        # Show regex pattern
        print("\nRegex pattern (first 5 entries):")
        pattern_parts = [f"{run['patient_id']}_{run['run_id']}" for run in saturated_runs[:5]]
        print(f"  ({('|'.join(pattern_parts))}|...)")
    else:
        print(f"No runs found with >={threshold}% saturation in test set")

    print(f"\n✓ Complete report available in: {report_filename}")


if __name__ == "__main__":
    main()
