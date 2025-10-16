#!/usr/bin/env python3
"""
Analyze ECG recordings by dividing them into 5-minute segments and checking
each segment for saturation. This allows dynamic threshold analysis to determine
how many runs would be excluded at different saturation percentage thresholds.

Output: JSON file with saturation statistics per recording
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pyedflib
from typing import Dict, List, Tuple
import re


def parse_run_id(name: str) -> str:
    """
    Extract run-ID as 'run-XX' (two digits).
    Accepts ..._run-05_events.tsv, ..._run-05_ecg.edf, ..._run-5.edf etc.
    """
    m = re.search(r"[_-]run-(\d+)(?=[_.]|$)", name)
    if m:
        return f"run-{int(m.group(1)):02d}"
    base = Path(name).stem
    return f"run-{hash(base) & 0xffff:04x}"


def is_segment_saturated(signal_data: np.ndarray,
                         digital_min: float = None,
                         digital_max: float = None) -> Tuple[bool, Dict, str]:
    """
    Determines if an ECG segment is 'saturated' based on multiple criteria:
    1. Very few unique values (flat/hard quantization)
    2. Very small dynamic range (peak-to-peak)
    3. (Optional) Significant fraction of samples at digital min/max (clipping)

    Returns:
        tuple: (is_saturated: bool, metrics: dict, reason: str)
    """
    if signal_data is None or len(signal_data) == 0:
        return True, {"unique_values": 0, "range": 0.0, "clip_fraction": 0.0}, "empty segment"

    unique_values = len(np.unique(signal_data))
    signal_range = float(np.ptp(signal_data))

    reason_parts = []
    saturated = False

    # Check 1: Less than 10 unique values suggests saturation
    if unique_values < 10:
        saturated = True
        reason_parts.append(f"low unique values ({unique_values})")

    # Check 2: Extremely small dynamic range (< 1 microV)
    if signal_range < 0.001:
        saturated = True
        reason_parts.append(f"low range ({signal_range:.3g})")

    # Check 3: Clipping at digital boundaries (if known)
    clip_fraction = 0.0
    if digital_min is not None and digital_max is not None and digital_max > digital_min:
        clip_mask = (signal_data <= digital_min) | (signal_data >= digital_max)
        clip_fraction = float(np.mean(clip_mask))
        if clip_fraction > 0.05:  # >5% at boundaries indicates strong clipping
            saturated = True
            reason_parts.append(f"clipping {clip_fraction*100:.1f}%")

    reason = ", ".join(reason_parts) if reason_parts else "not saturated"
    metrics = {
        "unique_values": unique_values,
        "range": signal_range,
        "clip_fraction": clip_fraction
    }

    return saturated, metrics, reason


def analyze_recording_saturation(ecg_file_path: str,
                                 segment_duration: float = 300.0) -> Dict:
    """
    Analyze an ECG recording by dividing it into segments and checking each for saturation.

    Args:
        ecg_file_path: Path to the EDF file
        segment_duration: Duration of each segment in seconds (default: 300s = 5 minutes)

    Returns:
        dict: Analysis results containing saturation statistics per channel
    """
    results = {
        'file_path': ecg_file_path,
        'file_duration': 0,
        'segment_duration': segment_duration,
        'total_segments': 0,
        'channels': {},
        'overall_saturated_segments': 0,
        'overall_usable_segments': 0,
        'error': None
    }

    try:
        with pyedflib.EdfReader(ecg_file_path) as f:
            # Basic file info
            file_duration = f.file_duration
            results['file_duration'] = file_duration
            signal_labels = f.getSignalLabels()

            # Find ECG channels
            ecg_channel_indices = []
            for i, label in enumerate(signal_labels):
                label_lower = label.lower()
                if any(ecg_keyword in label_lower for ecg_keyword in ['ecg', 'ekg', 'lead']):
                    ecg_channel_indices.append(i)

            if not ecg_channel_indices:
                results['error'] = f"No ECG channels found (available: {signal_labels})"
                return results

            # Calculate number of segments
            num_segments = int(np.ceil(file_duration / segment_duration))
            results['total_segments'] = num_segments

            # Analyze each channel
            for channel_idx in ecg_channel_indices:
                channel_label = signal_labels[channel_idx]
                sampling_rate = f.getSampleFrequency(channel_idx)
                total_samples = f.getNSamples()[channel_idx]

                # Get digital boundaries for this channel
                digital_min = f.getDigitalMinimum(channel_idx)
                digital_max = f.getDigitalMaximum(channel_idx)

                channel_results = {
                    'label': channel_label,
                    'sampling_rate': sampling_rate,
                    'total_samples': total_samples,
                    'segments': [],
                    'saturated_segments': 0,
                    'usable_segments': 0,
                    'empty_segments': 0
                }

                # Analyze each segment
                for seg_idx in range(num_segments):
                    seg_start_time = seg_idx * segment_duration
                    seg_end_time = min((seg_idx + 1) * segment_duration, file_duration)
                    seg_duration_actual = seg_end_time - seg_start_time

                    # Calculate sample indices
                    start_sample = int(seg_start_time * sampling_rate)
                    n_samples = int(seg_duration_actual * sampling_rate)

                    # Ensure we don't read beyond file boundaries
                    if start_sample >= total_samples:
                        break
                    if start_sample + n_samples > total_samples:
                        n_samples = total_samples - start_sample

                    if n_samples <= 0:
                        segment_info = {
                            'segment_index': seg_idx,
                            'start_time': seg_start_time,
                            'end_time': seg_end_time,
                            'duration': 0,
                            'saturated': True,
                            'reason': 'empty segment',
                            'metrics': {}
                        }
                        channel_results['segments'].append(segment_info)
                        channel_results['empty_segments'] += 1
                        continue

                    # Read signal segment
                    try:
                        signal_data = f.readSignal(channel_idx, start=start_sample, n=n_samples)

                        # Check for saturation
                        is_saturated, metrics, reason = is_segment_saturated(
                            signal_data,
                            digital_min=digital_min,
                            digital_max=digital_max
                        )

                        segment_info = {
                            'segment_index': seg_idx,
                            'start_time': seg_start_time,
                            'end_time': seg_end_time,
                            'duration': seg_duration_actual,
                            'saturated': is_saturated,
                            'reason': reason,
                            'metrics': metrics
                        }

                        channel_results['segments'].append(segment_info)

                        if is_saturated:
                            channel_results['saturated_segments'] += 1
                        else:
                            channel_results['usable_segments'] += 1

                    except Exception as e:
                        segment_info = {
                            'segment_index': seg_idx,
                            'start_time': seg_start_time,
                            'end_time': seg_end_time,
                            'duration': seg_duration_actual,
                            'saturated': True,
                            'reason': f'read error: {str(e)}',
                            'metrics': {}
                        }
                        channel_results['segments'].append(segment_info)
                        channel_results['empty_segments'] += 1

                results['channels'][channel_label] = channel_results

            # Calculate overall statistics (worst case across all channels)
            # A segment is considered saturated if ANY channel is saturated
            if results['channels']:
                segment_saturation = {}  # segment_idx -> is_saturated
                for channel_data in results['channels'].values():
                    for seg in channel_data['segments']:
                        seg_idx = seg['segment_index']
                        if seg_idx not in segment_saturation:
                            segment_saturation[seg_idx] = False
                        if seg['saturated']:
                            segment_saturation[seg_idx] = True

                results['overall_saturated_segments'] = sum(1 for s in segment_saturation.values() if s)
                results['overall_usable_segments'] = sum(1 for s in segment_saturation.values() if not s)

    except Exception as e:
        results['error'] = f"File read error: {str(e)}"

    return results


def analyze_all_recordings(base_dir: Path,
                           segment_duration: float = 300.0) -> Dict:
    """
    Analyze all ECG recordings in the dataset.

    Args:
        base_dir: Base directory of the dataset (e.g., ds005873-1.1.0)
        segment_duration: Duration of each segment in seconds

    Returns:
        dict: Complete analysis results
    """
    analysis_results = {
        'metadata': {
            'base_directory': str(base_dir),
            'segment_duration': segment_duration,
            'timestamp': datetime.now().isoformat(),
            'total_patients': 0,
            'total_recordings': 0,
            'recordings_analyzed': 0,
            'recordings_failed': 0
        },
        'patients': {}
    }

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found")
        return analysis_results

    # Get all subjects
    subjects = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    analysis_results['metadata']['total_patients'] = len(subjects)

    print(f"Found {len(subjects)} subjects")
    print(f"Segment duration: {segment_duration}s ({segment_duration/60:.1f} minutes)")
    print(f"\nAnalyzing recordings...\n")

    for subject in sorted(subjects):
        subject_id = subject.name
        print(f"\nProcessing {subject_id}...")

        analysis_results['patients'][subject_id] = {
            'recordings': {}
        }

        # Get sessions (typically ses-01)
        sessions = [d for d in subject.iterdir() if d.is_dir() and d.name.startswith('ses-')]

        for session in sessions:
            ecg_dir = session / "ecg"

            if not ecg_dir.exists():
                print(f"  Warning: No ECG directory for {subject_id}")
                continue

            # Get all ECG files for this subject
            ecg_files = list(ecg_dir.glob("*_ecg.edf"))

            for ecg_file in sorted(ecg_files):
                run_id = parse_run_id(ecg_file.name)
                analysis_results['metadata']['total_recordings'] += 1

                print(f"  Analyzing {run_id}...")

                # Analyze this recording
                recording_results = analyze_recording_saturation(
                    str(ecg_file),
                    segment_duration=segment_duration
                )

                # Add summary statistics
                recording_summary = {
                    'file_path': recording_results['file_path'],
                    'file_duration': recording_results['file_duration'],
                    'total_segments': recording_results['total_segments'],
                    'overall_saturated_segments': recording_results['overall_saturated_segments'],
                    'overall_usable_segments': recording_results['overall_usable_segments'],
                    'saturation_percentage': 0.0,
                    'error': recording_results.get('error'),
                    'channels': recording_results.get('channels', {})
                }

                # Calculate saturation percentage
                if recording_results['total_segments'] > 0:
                    recording_summary['saturation_percentage'] = (
                        recording_results['overall_saturated_segments'] /
                        recording_results['total_segments'] * 100
                    )

                analysis_results['patients'][subject_id]['recordings'][run_id] = recording_summary

                # Update counters
                if recording_results.get('error'):
                    analysis_results['metadata']['recordings_failed'] += 1
                    print(f"    ✗ Failed: {recording_results['error']}")
                else:
                    analysis_results['metadata']['recordings_analyzed'] += 1
                    sat_pct = recording_summary['saturation_percentage']
                    print(f"    ✓ {recording_results['total_segments']} segments, "
                          f"{recording_results['overall_saturated_segments']} saturated "
                          f"({sat_pct:.1f}%)")

    return analysis_results


def generate_threshold_analysis(analysis_results: Dict) -> str:
    """
    Generate a threshold analysis report showing how many runs would be excluded
    at different saturation percentage thresholds.

    Args:
        analysis_results: Complete analysis results from analyze_all_recordings

    Returns:
        str: Formatted report text
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SATURATION THRESHOLD ANALYSIS")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Segment Duration: {analysis_results['metadata']['segment_duration']}s")
    report_lines.append("")

    # Collect all recordings with their saturation percentages
    recordings = []
    for patient_id, patient_data in analysis_results['patients'].items():
        for run_id, recording_data in patient_data['recordings'].items():
            if not recording_data.get('error'):
                recordings.append({
                    'patient_id': patient_id,
                    'run_id': run_id,
                    'saturation_percentage': recording_data['saturation_percentage'],
                    'total_segments': recording_data['total_segments'],
                    'saturated_segments': recording_data['overall_saturated_segments']
                })

    total_recordings = len(recordings)
    report_lines.append(f"Total recordings analyzed: {total_recordings}")
    report_lines.append("")

    # Threshold analysis
    thresholds = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

    report_lines.append("THRESHOLD ANALYSIS:")
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

    # Distribution statistics
    if recordings:
        sat_percentages = [r['saturation_percentage'] for r in recordings]
        report_lines.append("SATURATION DISTRIBUTION:")
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

    # List of heavily saturated recordings (>50% saturated)
    heavily_saturated = [r for r in recordings if r['saturation_percentage'] > 50]
    if heavily_saturated:
        report_lines.append("HEAVILY SATURATED RECORDINGS (>50%):")
        report_lines.append("-"*80)
        for rec in sorted(heavily_saturated, key=lambda x: x['saturation_percentage'], reverse=True):
            report_lines.append(f"  {rec['patient_id']} {rec['run_id']}: "
                              f"{rec['saturation_percentage']:.1f}% "
                              f"({rec['saturated_segments']}/{rec['total_segments']} segments)")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    return "\n".join(report_lines)


def main():
    # Configuration
    base_dir = Path("/home/swolf/asim_shared/raw_data/ds005873-1.1.0")
    segment_duration = 300.0  # 5 minutes in seconds
    output_dir = Path("find_missing_ecg")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("ECG SATURATION ANALYSIS BY SEGMENTS")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Segment duration: {segment_duration}s ({segment_duration/60:.1f} minutes)")
    print("="*80)

    # Analyze all recordings
    results = analyze_all_recordings(base_dir, segment_duration)

    # Save results to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = output_dir / f'saturation_analysis_{int(segment_duration)}s_{timestamp}.json'

    print(f"\nSaving results to {json_filename}...")
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to '{json_filename}'")

    # Generate threshold analysis report
    print("\nGenerating threshold analysis report...")
    threshold_report = generate_threshold_analysis(results)

    report_filename = output_dir / f'saturation_threshold_analysis_{int(segment_duration)}s_{timestamp}.txt'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(threshold_report)
    print(f"✓ Report saved to '{report_filename}'")

    # Print summary to console
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total patients: {results['metadata']['total_patients']}")
    print(f"Total recordings: {results['metadata']['total_recordings']}")
    print(f"Successfully analyzed: {results['metadata']['recordings_analyzed']}")
    print(f"Failed: {results['metadata']['recordings_failed']}")
    print("")

    # Quick threshold preview
    print("Quick threshold preview (how many runs excluded at each threshold):")
    recordings = []
    for patient_data in results['patients'].values():
        for recording_data in patient_data['recordings'].values():
            if not recording_data.get('error'):
                recordings.append(recording_data['saturation_percentage'])

    if recordings:
        for threshold in [10, 20, 30, 50]:
            excluded = sum(1 for s in recordings if s >= threshold)
            print(f"  At {threshold}% threshold: {excluded} runs excluded "
                  f"({excluded/len(recordings)*100:.1f}% of total)")

    print("\n✓ Analysis complete!")
    print(f"See '{report_filename}' for detailed threshold analysis.")


if __name__ == "__main__":
    main()
