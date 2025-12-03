#!/usr/bin/env python3
"""
Count ground truth anomalies in TimeVQVAE-AD pickle files.
Reports per-file and total statistics.
"""

import pickle
import numpy as np
import torch
import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd


def load_pkl(fname: str):
    """Load pickle file."""
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_events(mask) -> List[Tuple[int, int]]:
    """Extract events (contiguous True regions) from boolean mask."""
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()
    elif hasattr(mask, 'numpy'):
        mask = mask.numpy()
    mask = np.asarray(mask).flatten()

    events = []
    in_event = False
    for i, v in enumerate(mask):
        if v and not in_event:
            start = i
            in_event = True
        elif not v and in_event:
            end = i - 1
            events.append((start, end))
            in_event = False
    if in_event:
        events.append((start, len(mask) - 1))
    return events


def count_ground_truths_in_file(pkl_path: Path, fs: float = 8.0) -> dict:
    """
    Count ground truth anomalies in a single file.

    Returns:
        Dictionary with ground truth statistics, or None if failed
    """
    try:
        data = load_pkl(pkl_path)

        # Extract ground truth labels
        Y = data.get('Y', None)
        if Y is None:
            return None

        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()
        Y = Y.astype(bool)

        # Get events
        events = get_events(Y)

        # Calculate statistics
        total_samples = len(Y)
        anomaly_samples = int(np.sum(Y))
        normal_samples = int(np.sum(~Y))
        anomaly_percentage = (anomaly_samples / total_samples) * 100

        duration_seconds = total_samples / fs
        duration_hours = duration_seconds / 3600.0

        # Calculate event durations
        event_durations = []
        for start, end in events:
            duration_sec = (end - start + 1) / fs
            event_durations.append(duration_sec)

        total_anomaly_duration = sum(event_durations)
        mean_event_duration = np.mean(event_durations) if event_durations else 0
        median_event_duration = np.median(event_durations) if event_durations else 0
        min_event_duration = np.min(event_durations) if event_durations else 0
        max_event_duration = np.max(event_durations) if event_durations else 0

        # Get dataset info
        dataset_index = data.get('dataset_index', pkl_path.stem.split('_')[0])
        subject_id = data.get('subject_id', f'sub-{dataset_index}')
        run_id = data.get('run_id', 'unknown')

        return {
            'file': pkl_path.name,
            'dataset_index': dataset_index,
            'subject_id': subject_id,
            'run_id': run_id,
            'total_samples': total_samples,
            'duration_seconds': duration_seconds,
            'duration_hours': duration_hours,
            'anomaly_samples': anomaly_samples,
            'normal_samples': normal_samples,
            'anomaly_percentage': anomaly_percentage,
            'n_events': len(events),
            'has_anomalies': len(events) > 0,
            'total_anomaly_duration_seconds': total_anomaly_duration,
            'mean_event_duration_seconds': mean_event_duration,
            'median_event_duration_seconds': median_event_duration,
            'min_event_duration_seconds': min_event_duration,
            'max_event_duration_seconds': max_event_duration,
            'events': events  # Store for detailed output if needed
        }

    except Exception as e:
        print(f"⚠ Error processing {pkl_path.name}: {e}")
        return None


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(text)
    print("="*80)


def print_section(text):
    """Print formatted section."""
    print("\n" + "-"*80)
    print(text)
    print("-"*80)


def main():
    parser = argparse.ArgumentParser(description='Count ground truth anomalies in TimeVQVAE-AD pickle files')
    parser.add_argument('input_dir', type=str, help='Directory containing pkl files')
    parser.add_argument('--pattern', type=str, default='*joint_anomaly_score.pkl',
                       help='File pattern (default: *joint_anomaly_score.pkl)')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Save results to CSV (default: None)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed event information for each file')
    parser.add_argument('--fs', type=float, default=8.0,
                       help='Sampling frequency in Hz (default: 8.0)')

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    # Find all matching files
    pkl_files = sorted(input_dir.rglob(args.pattern))

    if not pkl_files:
        print(f"ERROR: No files matching pattern '{args.pattern}' found in {input_dir}")
        return

    print_header("GROUND TRUTH COUNTER - TIMEVQVAE-AD")
    print(f"Input directory: {input_dir}")
    print(f"Files found: {len(pkl_files)}")
    print(f"Pattern: {args.pattern}")
    print(f"Sampling frequency: {args.fs} Hz")

    # Process files
    results = []
    failed_files = []

    for pkl_file in tqdm(pkl_files, desc="Counting ground truths"):
        result = count_ground_truths_in_file(pkl_file, fs=args.fs)

        if result is not None:
            results.append(result)
        else:
            failed_files.append(pkl_file.name)

    if not results:
        print("\nERROR: No files could be processed successfully")
        return

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Summary statistics
    print_header("SUMMARY STATISTICS")

    print(f"\nFiles processed: {len(results)}/{len(pkl_files)}")
    if failed_files:
        print(f"Failed files: {len(failed_files)}")

    print(f"\nTotal recording duration: {results_df['duration_hours'].sum():.2f} hours")
    print(f"Mean recording duration: {results_df['duration_hours'].mean():.2f} hours")

    print(f"\n--- Ground Truth Samples ---")
    print(f"Total samples (all files): {results_df['total_samples'].sum():,}")
    print(f"Total anomaly samples: {results_df['anomaly_samples'].sum():,}")
    print(f"Total normal samples: {results_df['normal_samples'].sum():,}")
    print(f"Overall anomaly percentage: {(results_df['anomaly_samples'].sum() / results_df['total_samples'].sum() * 100):.3f}%")

    print(f"\nMean anomaly percentage per file: {results_df['anomaly_percentage'].mean():.3f}%")
    print(f"Median anomaly percentage per file: {results_df['anomaly_percentage'].median():.3f}%")
    print(f"Min anomaly percentage: {results_df['anomaly_percentage'].min():.3f}%")
    print(f"Max anomaly percentage: {results_df['anomaly_percentage'].max():.3f}%")

    print(f"\n--- Ground Truth Events ---")
    print(f"Total events (all files): {results_df['n_events'].sum()}")
    print(f"Files with anomalies: {results_df['has_anomalies'].sum()}/{len(results_df)}")
    print(f"Files without anomalies: {(~results_df['has_anomalies']).sum()}/{len(results_df)}")

    print(f"\nMean events per file: {results_df['n_events'].mean():.2f}")
    print(f"Median events per file: {results_df['n_events'].median():.0f}")
    print(f"Min events per file: {results_df['n_events'].min()}")
    print(f"Max events per file: {results_df['n_events'].max()}")

    # Event duration statistics (only for files with events)
    files_with_events = results_df[results_df['n_events'] > 0]
    if len(files_with_events) > 0:
        print(f"\n--- Event Duration Statistics (files with events) ---")
        print(f"Total anomaly duration: {files_with_events['total_anomaly_duration_seconds'].sum():.2f} seconds = {files_with_events['total_anomaly_duration_seconds'].sum()/60:.2f} minutes")
        print(f"\nMean event duration: {files_with_events['mean_event_duration_seconds'].mean():.2f} seconds")
        print(f"Median event duration: {files_with_events['median_event_duration_seconds'].median():.2f} seconds")
        print(f"Shortest event (overall): {files_with_events['min_event_duration_seconds'].min():.2f} seconds")
        print(f"Longest event (overall): {files_with_events['max_event_duration_seconds'].max():.2f} seconds")

    # Top files by number of events
    print_section("TOP 10 FILES BY NUMBER OF EVENTS")
    top10 = results_df.nlargest(10, 'n_events')[['dataset_index', 'n_events', 'anomaly_samples', 'anomaly_percentage', 'duration_hours']]
    for idx, row in top10.iterrows():
        print(f"  {row['dataset_index']}: {row['n_events']} events, {row['anomaly_samples']:,} samples ({row['anomaly_percentage']:.2f}%), {row['duration_hours']:.2f}h")

    # Files with highest anomaly percentage
    print_section("TOP 10 FILES BY ANOMALY PERCENTAGE")
    top10_pct = results_df.nlargest(10, 'anomaly_percentage')[['dataset_index', 'n_events', 'anomaly_samples', 'anomaly_percentage', 'duration_hours']]
    for idx, row in top10_pct.iterrows():
        print(f"  {row['dataset_index']}: {row['anomaly_percentage']:.2f}%, {row['n_events']} events, {row['anomaly_samples']:,} samples, {row['duration_hours']:.2f}h")

    # Distribution of events
    print_section("DISTRIBUTION OF EVENTS PER FILE")
    event_counts = results_df['n_events'].value_counts().sort_index()
    print(f"\n{'Events':<10} {'Files':<10} {'Percentage'}")
    print("-" * 40)
    for n_events, count in event_counts.items():
        pct = (count / len(results_df)) * 100
        print(f"{n_events:<10} {count:<10} {pct:>6.1f}%")

    # Detailed output if requested
    if args.detailed:
        print_section("DETAILED EVENT INFORMATION")
        for idx, row in results_df.iterrows():
            if row['n_events'] > 0:
                print(f"\n{row['dataset_index']} ({row['file']}):")
                print(f"  Events: {row['n_events']}")
                print(f"  Duration: {row['duration_hours']:.2f} hours")
                print(f"  Anomaly samples: {row['anomaly_samples']:,} ({row['anomaly_percentage']:.2f}%)")

                events = row['events']
                if len(events) <= 10:
                    print(f"  Event details:")
                    for i, (start, end) in enumerate(events, 1):
                        duration = (end - start + 1) / args.fs
                        start_time = start / args.fs / 60.0  # minutes
                        print(f"    {i}. {duration:.2f}s (at {start_time:.2f} min, samples {start}-{end})")
                else:
                    print(f"  Event details (first 10 of {len(events)}):")
                    for i, (start, end) in enumerate(events[:10], 1):
                        duration = (end - start + 1) / args.fs
                        start_time = start / args.fs / 60.0
                        print(f"    {i}. {duration:.2f}s (at {start_time:.2f} min, samples {start}-{end})")

    # Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        # Remove 'events' column for CSV (can't serialize tuples easily)
        results_df_export = results_df.drop(columns=['events'])
        results_df_export.to_csv(output_path, index=False)
        print(f"\n✓ Saved results to: {output_path}")

    # Failed files
    if failed_files:
        print_section("FAILED FILES")
        print(f"\nCould not process {len(failed_files)} files:")
        for f in failed_files[:20]:
            print(f"  - {f}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files)-20} more")

    print("\n" + "="*80)
    print("COUNTING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
