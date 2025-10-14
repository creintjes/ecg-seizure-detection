#!/usr/bin/env python3
"""
Analyze Madrid windowed batch results from a directory.
Summarizes seizure information from all JSON result files.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse


def load_madrid_result(file_path):
    """
    Load a Madrid result JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        dict: Parsed JSON data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_seizure_info(result_data, file_path):
    """
    Extract seizure information from a Madrid result.

    Args:
        result_data: Parsed JSON data
        file_path: Path to the file (for reference)

    Returns:
        dict: Extracted seizure information
    """
    info = {
        'file_path': str(file_path),
        'file_name': Path(file_path).name,
        'subject_id': None,
        'run_id': None,
        'seizure_id': None,
        'seizure_present': False,
        'total_windows': 0,
        'seizure_windows_count': 0,
        'seizure_windows': [],
        'total_seizure_segments': 0,
        'total_seizure_duration_seconds': 0,
        'analysis_timestamp': None,
        'analysis_successful': False
    }

    # Extract metadata
    if 'analysis_metadata' in result_data:
        metadata = result_data['analysis_metadata']
        info['analysis_timestamp'] = metadata.get('timestamp', None)
        if 'computation_info' in metadata:
            info['analysis_successful'] = metadata['computation_info'].get('analysis_successful', False)

    # Extract input data
    if 'input_data' in result_data:
        input_data = result_data['input_data']
        info['subject_id'] = input_data.get('subject_id', None)
        info['run_id'] = input_data.get('run_id', None)
        info['seizure_id'] = input_data.get('seizure_id', None)

        if 'signal_metadata' in input_data:
            info['total_windows'] = input_data['signal_metadata'].get('num_windows', 0)

    # Extract validation/ground truth data
    if 'validation_data' in result_data:
        validation = result_data['validation_data']
        if 'ground_truth' in validation:
            ground_truth = validation['ground_truth']
            info['seizure_present'] = ground_truth.get('seizure_present', False)

            if 'seizure_windows' in ground_truth:
                seizure_windows = ground_truth['seizure_windows']
                info['seizure_windows_count'] = len(seizure_windows)
                info['seizure_windows'] = seizure_windows

                # Count segments and total duration
                total_duration = 0
                total_segments = 0

                for window in seizure_windows:
                    if 'seizure_segments' in window:
                        segments = window['seizure_segments']
                        total_segments += len(segments)

                        for segment in segments:
                            duration = segment.get('duration_seconds', 0)
                            total_duration += duration

                info['total_seizure_segments'] = total_segments
                info['total_seizure_duration_seconds'] = total_duration

    return info


def analyze_directory(directory_path, output_file=None):
    """
    Analyze all Madrid result JSON files in a directory.

    Args:
        directory_path: Path to directory containing JSON files
        output_file: Optional path to save summary report

    Returns:
        dict: Summary statistics
    """
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return None

    # Find all JSON files
    json_files = list(directory.glob("madrid_windowed_results_*.json"))

    if not json_files:
        print(f"No Madrid result JSON files found in {directory}")
        return None

    print(f"Found {len(json_files)} Madrid result files")
    print("="*80)

    # Process each file
    all_results = []

    for json_file in sorted(json_files):
        print(f"\nProcessing: {json_file.name}")

        result_data = load_madrid_result(json_file)
        if result_data is None:
            continue

        info = extract_seizure_info(result_data, json_file)
        all_results.append(info)

        # Print basic info
        print(f"  Subject: {info['subject_id']}, Run: {info['run_id']}")
        print(f"  Seizure present: {info['seizure_present']}")

        if info['seizure_present']:
            print(f"  Seizure windows: {info['seizure_windows_count']}/{info['total_windows']}")
            print(f"  Seizure segments: {info['total_seizure_segments']}")
            print(f"  Total seizure duration: {info['total_seizure_duration_seconds']:.1f} seconds")

    # Generate summary statistics
    summary = generate_summary(all_results)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nFiles analyzed: {summary['total_files']}")
    print(f"Successful analyses: {summary['successful_analyses']}")
    print(f"Failed analyses: {summary['failed_analyses']}")

    print(f"\nSEIZURE STATISTICS:")
    print(f"  Files with seizures: {summary['files_with_seizures']}")
    print(f"  Files without seizures: {summary['files_without_seizures']}")
    print(f"  Total seizure windows: {summary['total_seizure_windows']}")
    print(f"  Total seizure segments: {summary['total_seizure_segments']}")
    print(f"  Total seizure duration: {summary['total_seizure_duration_seconds']:.1f} seconds ({summary['total_seizure_duration_minutes']:.2f} minutes)")

    if summary['files_with_seizures'] > 0:
        print(f"\nAVERAGE PER FILE WITH SEIZURES:")
        print(f"  Avg seizure windows per file: {summary['avg_seizure_windows_per_file']:.2f}")
        print(f"  Avg seizure segments per file: {summary['avg_seizure_segments_per_file']:.2f}")
        print(f"  Avg seizure duration per file: {summary['avg_seizure_duration_per_file']:.2f} seconds")

    print(f"\nSUBJECTS AND RUNS:")
    print(f"  Unique subjects: {len(summary['unique_subjects'])}")
    print(f"  Unique runs: {len(summary['unique_runs'])}")

    print(f"\nPER-SUBJECT BREAKDOWN:")
    for subject_id, count in sorted(summary['subjects_with_seizures'].items()):
        print(f"  {subject_id}: {count} file(s) with seizures")

    # Save detailed report if requested
    if output_file:
        save_report(summary, all_results, output_file)
        print(f"\n✓ Detailed report saved to: {output_file}")

    return summary


def generate_summary(all_results):
    """
    Generate summary statistics from all results.

    Args:
        all_results: List of extracted info dictionaries

    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_files': len(all_results),
        'successful_analyses': sum(1 for r in all_results if r['analysis_successful']),
        'failed_analyses': sum(1 for r in all_results if not r['analysis_successful']),
        'files_with_seizures': sum(1 for r in all_results if r['seizure_present']),
        'files_without_seizures': sum(1 for r in all_results if not r['seizure_present']),
        'total_seizure_windows': sum(r['seizure_windows_count'] for r in all_results),
        'total_seizure_segments': sum(r['total_seizure_segments'] for r in all_results),
        'total_seizure_duration_seconds': sum(r['total_seizure_duration_seconds'] for r in all_results),
        'total_seizure_duration_minutes': sum(r['total_seizure_duration_seconds'] for r in all_results) / 60.0,
        'unique_subjects': set(r['subject_id'] for r in all_results if r['subject_id']),
        'unique_runs': set((r['subject_id'], r['run_id']) for r in all_results if r['subject_id'] and r['run_id']),
        'subjects_with_seizures': defaultdict(int),
        'avg_seizure_windows_per_file': 0,
        'avg_seizure_segments_per_file': 0,
        'avg_seizure_duration_per_file': 0
    }

    # Count seizures per subject
    for result in all_results:
        if result['seizure_present'] and result['subject_id']:
            summary['subjects_with_seizures'][result['subject_id']] += 1

    # Calculate averages
    if summary['files_with_seizures'] > 0:
        summary['avg_seizure_windows_per_file'] = summary['total_seizure_windows'] / summary['files_with_seizures']
        summary['avg_seizure_segments_per_file'] = summary['total_seizure_segments'] / summary['files_with_seizures']
        summary['avg_seizure_duration_per_file'] = summary['total_seizure_duration_seconds'] / summary['files_with_seizures']

    return summary


def save_report(summary, all_results, output_file):
    """
    Save detailed report to file.

    Args:
        summary: Summary statistics dictionary
        all_results: List of all extracted results
        output_file: Path to output file
    """
    output_path = Path(output_file)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MADRID RESULTS ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files analyzed: {summary['total_files']}\n")
        f.write("\n")

        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Successful analyses: {summary['successful_analyses']}\n")
        f.write(f"Failed analyses: {summary['failed_analyses']}\n")
        f.write(f"\n")
        f.write(f"Files with seizures: {summary['files_with_seizures']}\n")
        f.write(f"Files without seizures: {summary['files_without_seizures']}\n")
        f.write(f"Total seizure windows: {summary['total_seizure_windows']}\n")
        f.write(f"Total seizure segments: {summary['total_seizure_segments']}\n")
        f.write(f"Total seizure duration: {summary['total_seizure_duration_seconds']:.1f} seconds ({summary['total_seizure_duration_minutes']:.2f} minutes)\n")
        f.write(f"\n")

        if summary['files_with_seizures'] > 0:
            f.write(f"Average seizure windows per file: {summary['avg_seizure_windows_per_file']:.2f}\n")
            f.write(f"Average seizure segments per file: {summary['avg_seizure_segments_per_file']:.2f}\n")
            f.write(f"Average seizure duration per file: {summary['avg_seizure_duration_per_file']:.2f} seconds\n")

        f.write(f"\n")
        f.write(f"Unique subjects: {len(summary['unique_subjects'])}\n")
        f.write(f"Unique runs: {len(summary['unique_runs'])}\n")
        f.write("\n")

        # Per-subject breakdown
        f.write("PER-SUBJECT BREAKDOWN\n")
        f.write("-"*80 + "\n")
        for subject_id, count in sorted(summary['subjects_with_seizures'].items()):
            f.write(f"{subject_id}: {count} file(s) with seizures\n")
        f.write("\n")

        # Detailed file listing
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("DETAILED FILE LISTING\n")
        f.write("="*80 + "\n")
        f.write("\n")

        for result in all_results:
            f.write(f"File: {result['file_name']}\n")
            f.write(f"  Subject: {result['subject_id']}, Run: {result['run_id']}\n")
            f.write(f"  Analysis successful: {result['analysis_successful']}\n")
            f.write(f"  Seizure present: {result['seizure_present']}\n")

            if result['seizure_present']:
                f.write(f"  Total windows: {result['total_windows']}\n")
                f.write(f"  Seizure windows: {result['seizure_windows_count']}\n")
                f.write(f"  Seizure segments: {result['total_seizure_segments']}\n")
                f.write(f"  Total seizure duration: {result['total_seizure_duration_seconds']:.1f} seconds\n")

                # List each seizure window
                if result['seizure_windows']:
                    f.write(f"  Seizure window details:\n")
                    for sw in result['seizure_windows']:
                        f.write(f"    Window {sw['window_index']}: {len(sw.get('seizure_segments', []))} segment(s), "
                               f"ratio: {sw.get('seizure_ratio', 0):.4f}\n")
                        for seg in sw.get('seizure_segments', []):
                            f.write(f"      Segment: {seg['start_time_absolute']:.1f}s - {seg['end_time_absolute']:.1f}s "
                                   f"(duration: {seg['duration_seconds']:.1f}s)\n")

            f.write("\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze Madrid windowed batch result files and summarize seizure information.'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing Madrid result JSON files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file for detailed report (optional)'
    )

    args = parser.parse_args()

    # If no output file specified, create default name
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'madrid_results_summary_{timestamp}.txt'

    # Analyze directory
    analyze_directory(args.directory, output_file)


if __name__ == "__main__":
    main()
