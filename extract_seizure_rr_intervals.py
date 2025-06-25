#!/usr/bin/env python3
"""
Script to extract RR intervals from 5 random seizures from the SeizeIT2 dataset.
For each seizure: 5 minutes before to 5 minutes after the seizure event.
Saves RR intervals in ms and seizure indices to a text file.
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
import argparse
import matplotlib.pyplot as plt

# Add the seizeit2 classes to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Information', 'Data', 'seizeit2-main'))
from classes.data import Data
from classes.annotation import Annotation


def detect_r_peaks(ecg_signal, fs, min_distance_ms=200):
    """
    Detect R peaks in ECG signal using peak detection.
    
    Args:
        ecg_signal: ECG signal array
        fs: Sampling frequency
        min_distance_ms: Minimum distance between R peaks in milliseconds
        
    Returns:
        Array of R peak indices
    """
    # Convert min distance to samples
    min_distance_samples = int(min_distance_ms * fs / 1000)
    
    # Find peaks with minimum distance constraint
    peaks, _ = find_peaks(ecg_signal, distance=min_distance_samples, height=np.std(ecg_signal) * 0.5)
    
    return peaks


def calculate_rr_intervals(r_peaks, fs):
    """
    Calculate RR intervals from R peak positions.
    
    Args:
        r_peaks: Array of R peak indices
        fs: Sampling frequency
        
    Returns:
        Array of RR intervals in milliseconds
    """
    if len(r_peaks) < 2:
        return np.array([])
    
    # Calculate intervals between consecutive R peaks
    rr_intervals_samples = np.diff(r_peaks)
    
    # Convert to milliseconds
    rr_intervals_ms = (rr_intervals_samples / fs) * 1000
    
    return rr_intervals_ms


def plot_r_peaks(ecg_segment, r_peaks, fs, seizure_start_idx, seizure_end_idx, 
                 recording_info, seizure_id, output_dir="."):
    """
    Create a plot showing ECG signal with detected R peaks and seizure boundaries.
    
    Args:
        ecg_segment: ECG signal array
        r_peaks: Array of R peak indices
        fs: Sampling frequency
        seizure_start_idx: Index where seizure starts in segment
        seizure_end_idx: Index where seizure ends in segment
        recording_info: Recording information tuple (subject, run)
        seizure_id: Seizure identifier for filename
        output_dir: Directory to save plot
    """
    # Create time axis in seconds
    time_axis = np.arange(len(ecg_segment)) / fs
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full segment overview
    ax1.plot(time_axis, ecg_segment, 'b-', linewidth=0.8, alpha=0.7, label='ECG Signal')
    ax1.plot(time_axis[r_peaks], ecg_segment[r_peaks], 'ro', markersize=4, 
             label=f'R Peaks (n={len(r_peaks)})')
    
    # Mark seizure period
    if seizure_start_idx < len(ecg_segment) and seizure_end_idx < len(ecg_segment):
        ax1.axvspan(seizure_start_idx/fs, seizure_end_idx/fs, 
                   alpha=0.3, color='red', label='Seizure Period')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('ECG Amplitude')
    ax1.set_title(f'ECG Signal with R Peak Detection - {recording_info[0]} {recording_info[1]} Seizure {seizure_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view of seizure period (±30 seconds around seizure)
    if seizure_start_idx < len(ecg_segment) and seizure_end_idx < len(ecg_segment):
        seizure_center = (seizure_start_idx + seizure_end_idx) // 2
        zoom_start = max(0, seizure_center - int(30 * fs))  # 30 seconds before center
        zoom_end = min(len(ecg_segment), seizure_center + int(30 * fs))  # 30 seconds after center
        
        zoom_time = time_axis[zoom_start:zoom_end]
        zoom_ecg = ecg_segment[zoom_start:zoom_end]
        
        # Find R peaks in zoom window
        zoom_peaks = r_peaks[(r_peaks >= zoom_start) & (r_peaks < zoom_end)] - zoom_start
        
        ax2.plot(zoom_time, zoom_ecg, 'b-', linewidth=1.2, label='ECG Signal')
        if len(zoom_peaks) > 0:
            ax2.plot(zoom_time[zoom_peaks], zoom_ecg[zoom_peaks], 'ro', markersize=6, 
                    label=f'R Peaks (n={len(zoom_peaks)})')
        
        # Mark seizure boundaries in zoom
        seizure_start_time = seizure_start_idx / fs
        seizure_end_time = seizure_end_idx / fs
        if zoom_time[0] <= seizure_start_time <= zoom_time[-1]:
            ax2.axvline(seizure_start_time, color='red', linestyle='--', 
                       label='Seizure Start', linewidth=2)
        if zoom_time[0] <= seizure_end_time <= zoom_time[-1]:
            ax2.axvline(seizure_end_time, color='red', linestyle='--', 
                       label='Seizure End', linewidth=2)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('ECG Amplitude')
        ax2.set_title('Zoomed View: Seizure Period (±30 seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Seizure boundaries not available for zoom view', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('Zoomed View: Not Available')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"r_peaks_{recording_info[0]}_{recording_info[1]}_seizure_{seizure_id:02d}.png"
    filepath = output_path / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"  - Plot saved: {filepath}")


def extract_seizure_segment(ecg_data, annotations, seizure_idx, fs, pre_minutes=5, post_minutes=5):
    """
    Extract ECG segment around a seizure event.
    
    Args:
        ecg_data: ECG signal array
        annotations: Annotation object
        seizure_idx: Index of the seizure to extract
        fs: Sampling frequency
        pre_minutes: Minutes before seizure onset
        post_minutes: Minutes after seizure offset
        
    Returns:
        Tuple of (ecg_segment, seizure_start_in_segment, seizure_end_in_segment)
    """
    seizure_start_sec = annotations.events[seizure_idx][0]
    seizure_end_sec = annotations.events[seizure_idx][1]
    
    # Calculate segment boundaries
    segment_start_sec = max(0, seizure_start_sec - pre_minutes * 60)
    segment_end_sec = min(len(ecg_data) / fs, seizure_end_sec + post_minutes * 60)
    
    # Convert to sample indices
    segment_start_idx = int(segment_start_sec * fs)
    segment_end_idx = int(segment_end_sec * fs)
    
    # Extract segment
    ecg_segment = ecg_data[segment_start_idx:segment_end_idx]
    
    # Calculate seizure position within segment
    seizure_start_in_segment = int((seizure_start_sec - segment_start_sec) * fs)
    seizure_end_in_segment = int((seizure_end_sec - segment_start_sec) * fs)
    
    return ecg_segment, seizure_start_in_segment, seizure_end_in_segment


def process_seizures(data_path, output_file, num_seizures=5, plot_dir="r_peak_plots"):
    """
    Process random seizures and extract RR intervals.
    
    Args:
        data_path: Path to SeizeIT2 dataset
        output_file: Output file path
        num_seizures: Number of seizures to process
        plot_dir: Directory to save R peak plots
    """
    data_path = Path(data_path)
    
    # Build recordings list
    sub_list = [x for x in data_path.glob("sub-*") if x.is_dir()]
    recordings = [[x.name, xx.name.split('_')[-2]] for x in sub_list for xx in (x / 'ses-01' / 'ecg').glob("*_ecg.edf")]
    
    print(f"Found {len(recordings)} recordings")
    
    # Collect all seizures from all recordings
    all_seizures = []
    
    for rec in recordings:
        try:
            # Load annotations
            rec_annotations = Annotation.loadAnnotation(data_path.as_posix(), rec)
            
            # Skip if no seizures
            if len(rec_annotations.events) == 0:
                continue
                
            # Add seizures with recording info
            for i, event in enumerate(rec_annotations.events):
                all_seizures.append({
                    'recording': rec,
                    'seizure_idx': i,
                    'annotations': rec_annotations,
                    'start_time': event[0],
                    'duration': event[1] - event[0]
                })
                
        except Exception as e:
            print(f"Error processing {rec}: {e}")
            continue
    
    print(f"Found {len(all_seizures)} total seizures")
    
    if len(all_seizures) < num_seizures:
        print(f"Warning: Only {len(all_seizures)} seizures available, processing all of them")
        num_seizures = len(all_seizures)
    
    # Randomly select seizures
    selected_seizures = random.sample(all_seizures, num_seizures)
    
    results = []
    
    for i, seizure_info in enumerate(selected_seizures):
        print(f"Processing seizure {i+1}/{num_seizures}: {seizure_info['recording'][0]} {seizure_info['recording'][1]}")
        
        try:
            # Load ECG data
            rec_data = Data.loadData(data_path.as_posix(), seizure_info['recording'], modalities=['ecg'])
            
            # Find ECG channel
            ecg_channel_idx = None
            for j, channel in enumerate(rec_data.channels):
                if 'ecg' in channel.lower() or 'ekg' in channel.lower():
                    ecg_channel_idx = j
                    break
            
            if ecg_channel_idx is None:
                print(f"No ECG channel found in {seizure_info['recording']}")
                continue
            
            ecg_data = rec_data.data[ecg_channel_idx]
            fs = rec_data.fs[ecg_channel_idx]
            
            # Extract seizure segment
            ecg_segment, seizure_start_idx, seizure_end_idx = extract_seizure_segment(
                ecg_data, seizure_info['annotations'], seizure_info['seizure_idx'], fs
            )
            
            # Detect R peaks
            r_peaks = detect_r_peaks(ecg_segment, fs)
            
            # Calculate RR intervals
            rr_intervals = calculate_rr_intervals(r_peaks, fs)
            
            if len(rr_intervals) == 0:
                print(f"No RR intervals found for seizure {i+1}")
                continue
            
            # Create plot showing R peaks
            plot_r_peaks(ecg_segment, r_peaks, fs, seizure_start_idx, seizure_end_idx,
                        seizure_info['recording'], i + 1, output_dir=plot_dir)
            
            # Find which RR intervals correspond to seizure period
            seizure_rr_indices = []
            for j, peak_idx in enumerate(r_peaks[:-1]):  # -1 because RR intervals are between peaks
                if seizure_start_idx <= peak_idx <= seizure_end_idx:
                    seizure_rr_indices.append(j)
            
            results.append({
                'seizure_id': i + 1,
                'recording': f"{seizure_info['recording'][0]}_{seizure_info['recording'][1]}",
                'rr_intervals': rr_intervals,
                'seizure_indices': seizure_rr_indices,
                'seizure_duration': seizure_info['duration'],
                'total_rr_intervals': len(rr_intervals)
            })
            
            print(f"  - Extracted {len(rr_intervals)} RR intervals")
            print(f"  - Seizure spans RR intervals {seizure_rr_indices[0] if seizure_rr_indices else 'N/A'} to {seizure_rr_indices[-1] if seizure_rr_indices else 'N/A'}")
            
        except Exception as e:
            print(f"Error processing seizure {i+1}: {e}")
            continue
    
    # Save results to file
    with open(output_file, 'w') as f:
        f.write("# RR Intervals from Random Seizures (SeizeIT2 Dataset)\n")
        f.write("# Format: Seizure_ID | Recording | RR_Intervals_ms | Seizure_RR_Indices\n")
        f.write("# RR intervals are in milliseconds\n")
        f.write("# Seizure indices indicate which RR intervals occur during seizure\n\n")
        
        for result in results:
            f.write(f"Seizure_{result['seizure_id']}:\n")
            f.write(f"Recording: {result['recording']}\n")
            f.write(f"Total_RR_Intervals: {result['total_rr_intervals']}\n")
            f.write(f"Seizure_Duration_sec: {result['seizure_duration']:.2f}\n")
            
            # Write RR intervals
            f.write("RR_Intervals_ms: ")
            f.write(", ".join([f"{interval:.2f}" for interval in result['rr_intervals']]))
            f.write("\n")
            
            # Write seizure indices
            f.write("Seizure_RR_Indices: ")
            f.write(", ".join([str(idx) for idx in result['seizure_indices']]))
            f.write("\n\n")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Processed {len(results)} seizures successfully")


def main():
    parser = argparse.ArgumentParser(description='Extract RR intervals from random seizures')
    parser.add_argument('data_path', help='Path to SeizeIT2 dataset')
    parser.add_argument('--output', '-o', default='seizure_rr_intervals.txt', 
                       help='Output file name (default: seizure_rr_intervals.txt)')
    parser.add_argument('--num_seizures', '-n', type=int, default=5,
                       help='Number of seizures to process (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--plot-dir', default='r_peak_plots',
                       help='Directory to save R peak plots (default: r_peak_plots)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        sys.exit(1)
    
    process_seizures(args.data_path, args.output, args.num_seizures, args.plot_dir)


if __name__ == "__main__":
    main()