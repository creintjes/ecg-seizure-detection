#!/usr/bin/env python3
"""
Plot ECG examples from test set (sub-097 to sub-125):
- 3 saturated signals (only raw)
- 3 normal signals (before and after preprocessing)
- 3 seizure signals (before and after preprocessing)

Each plot shows 10 seconds of ECG signal.
Seizure plots start at seizure onset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Information', 'Data', 'seizeit2_main'))

import numpy as np
import matplotlib.pyplot as plt
import pyedflib
from pathlib import Path
from datetime import datetime
from scipy.signal import butter, filtfilt, decimate
import pandas as pd
import random

from classes.data import Data
from classes.annotation import Annotation
from config import RAW_DATA_PATH


class ECGPlotter:
    """Plot ECG examples with preprocessing comparison."""

    def __init__(self, base_dir: Path, output_dir: Path):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)

        # Madrid preprocessing parameters
        self.filter_params = {
            'low_freq': 0.5,
            'high_freq': 40.0,
            'order': 4
        }
        self.downsample_freq = 8

        # Create output directories
        self.saturated_dir = self.output_dir / 'saturated'
        self.normal_dir = self.output_dir / 'normal'
        self.seizure_dir = self.output_dir / 'seizure'

        for dir_path in [self.saturated_dir, self.normal_dir, self.seizure_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Output directories created:")
        print(f"  - {self.saturated_dir}")
        print(f"  - {self.normal_dir}")
        print(f"  - {self.seizure_dir}")

    def apply_bandpass_filter(self, signal_data: np.ndarray, fs: int) -> np.ndarray:
        """Apply bandpass filter to ECG signal (from Madrid preprocessing)."""
        nyquist = fs / 2
        low_norm = self.filter_params['low_freq'] / nyquist
        high_norm = self.filter_params['high_freq'] / nyquist

        if high_norm >= 1.0:
            high_norm = 0.99

        b, a = butter(
            self.filter_params['order'],
            [low_norm, high_norm],
            btype='band'
        )

        filtered_signal = filtfilt(b, a, signal_data)
        return filtered_signal

    def downsample_signal(self, signal_data: np.ndarray, original_fs: int) -> np.ndarray:
        """Downsample signal to target frequency (from Madrid preprocessing)."""
        if self.downsample_freq >= original_fs:
            return signal_data

        decimation_factor = original_fs // self.downsample_freq

        if decimation_factor < 2:
            from scipy import signal
            n_samples = int(len(signal_data) * self.downsample_freq / original_fs)
            return signal.resample(signal_data, n_samples)

        downsampled_signal = decimate(signal_data, decimation_factor, ftype='iir')
        return downsampled_signal

    def preprocess_signal(self, signal_data: np.ndarray, fs: int) -> tuple:
        """
        Apply Madrid preprocessing pipeline.

        Returns:
            (filtered_signal, downsampled_signal, new_fs)
        """
        # Apply bandpass filter
        filtered_signal = self.apply_bandpass_filter(signal_data, fs)

        # Downsample
        downsampled_signal = self.downsample_signal(filtered_signal, fs)
        new_fs = min(self.downsample_freq, fs)

        return filtered_signal, downsampled_signal, new_fs

    def load_ecg_segment(self, subject_id: str, run_id: str,
                         start_time: float = 0.0, duration: float = 10.0) -> dict:
        """
        Load ECG segment from EDF file.

        Args:
            subject_id: Patient ID (e.g., 'sub-097')
            run_id: Run ID (e.g., 'run-01')
            start_time: Start time in seconds
            duration: Duration in seconds

        Returns:
            dict with signal data, sampling rate, channel name
        """
        try:
            # Load using SeizeIT2 classes
            recording = [subject_id, run_id]
            data = Data.loadData(str(self.base_dir), recording, modalities=['ecg'])

            if not data.data:
                return None

            # Get first ECG channel
            channel_data = data.data[0]
            channel_name = data.channels[0]
            fs = int(data.fs[0])

            # Extract segment
            start_sample = int(start_time * fs)
            n_samples = int(duration * fs)
            end_sample = start_sample + n_samples

            if end_sample > len(channel_data):
                end_sample = len(channel_data)
                n_samples = end_sample - start_sample

            segment = channel_data[start_sample:end_sample]

            return {
                'signal': segment,
                'fs': fs,
                'channel_name': channel_name,
                'start_time': start_time,
                'duration': len(segment) / fs,
                'subject_id': subject_id,
                'run_id': run_id
            }

        except Exception as e:
            print(f"Error loading {subject_id} {run_id}: {e}")
            return None

    def plot_raw_only(self, segment: dict, output_path: Path, title: str = None):
        """Plot only raw signal (for saturated examples)."""
        fig, ax = plt.subplots(figsize=(5, 4))

        signal = segment['signal']
        fs = segment['fs']
        time_axis = np.arange(len(signal)) / fs + segment['start_time']

        ax.plot(time_axis, signal, linewidth=2.5, color='blue')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude (mV)', fontsize=12)
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"Raw ECG: {segment['subject_id']} {segment['run_id']}",
                        fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=3000, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

    def plot_before_after(self, segment: dict, output_path: Path, title: str = None):
        """Plot before and after preprocessing."""
        # Preprocess
        filtered, downsampled, new_fs = self.preprocess_signal(segment['signal'], segment['fs'])

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(5, 8))

        # Time axes
        time_raw = np.arange(len(segment['signal'])) / segment['fs'] + segment['start_time']
        time_processed = np.arange(len(downsampled)) / new_fs + segment['start_time']

        # Plot 1: Raw signal
        axes[0].plot(time_raw, segment['signal'], linewidth=0.5, color='blue')
        axes[0].set_ylabel('Amplitude (mV)', fontsize=11)
        axes[0].set_title('Before Preprocessing (Raw)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Processed signal
        axes[1].plot(time_processed, downsampled, linewidth=0.8, color='green')
        axes[1].set_xlabel('Time (seconds)', fontsize=11)
        axes[1].set_ylabel('Amplitude (mV)', fontsize=11)
        axes[1].set_title(f'After Preprocessing (Bandpass 0.5-40 Hz, Downsampled to {new_fs} Hz)',
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Overall title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        else:
            fig.suptitle(f"{segment['subject_id']} {segment['run_id']}",
                        fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(output_path, dpi=3000, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

    def is_segment_saturated(self, signal: np.ndarray) -> tuple:
        """
        Check if a signal segment is saturated.

        Returns:
            (is_saturated, reason)
        """
        if signal is None or len(signal) == 0:
            return True, "empty segment"

        uniq = len(np.unique(signal))
        ptp = float(np.ptp(signal))

        # Criteria from analyze_data.py
        if uniq < 10:
            return True, f"low unique values ({uniq})"

        if ptp < 1e-3:
            return True, f"low range (ptp={ptp:.3g})"

        return False, "not saturated"

    def find_saturated_segment(self, subject_id: str, run_id: str,
                               segment_duration: float = 10.0) -> dict:
        """
        Find a saturated segment in the recording.

        Scans through the recording in 10-second windows to find
        a saturated segment.

        Returns:
            dict with segment info or None if no saturated segment found
        """
        try:
            # Load full recording
            recording = [subject_id, run_id]
            data = Data.loadData(str(self.base_dir), recording, modalities=['ecg'])

            if not data.data:
                return None

            channel_data = data.data[0]
            fs = int(data.fs[0])

            # Scan through recording in 10-second windows
            window_samples = int(segment_duration * fs)
            step_samples = int(5.0 * fs)  # 5-second steps (50% overlap)

            for start_sample in range(0, len(channel_data) - window_samples, step_samples):
                segment = channel_data[start_sample:start_sample + window_samples]

                is_saturated, reason = self.is_segment_saturated(segment)

                if is_saturated:
                    start_time = start_sample / fs
                    print(f"    Found saturated segment at t={start_time:.1f}s ({reason})")

                    return {
                        'signal': segment,
                        'fs': fs,
                        'channel_name': data.channels[0],
                        'start_time': start_time,
                        'duration': len(segment) / fs,
                        'subject_id': subject_id,
                        'run_id': run_id,
                        'saturation_reason': reason
                    }

            print(f"    Warning: No saturated segment found, using start of recording")
            # Fallback: use beginning
            segment = channel_data[:window_samples]
            return {
                'signal': segment,
                'fs': fs,
                'channel_name': data.channels[0],
                'start_time': 0.0,
                'duration': len(segment) / fs,
                'subject_id': subject_id,
                'run_id': run_id,
                'saturation_reason': 'fallback'
            }

        except Exception as e:
            print(f"    Error finding saturated segment: {e}")
            return None

    def find_saturated_examples(self, n_examples: int = 3) -> list:
        """Find saturated signal examples from test set."""
        # Known saturated runs from test set (from analysis results)
        saturated_candidates = [
            ('sub-099', 'run-01'),
            ('sub-114', 'run-03'),
            ('sub-115', 'run-11'),
            ('sub-115', 'run-32'),
            ('sub-117', 'run-13'),
            ('sub-118', 'run-07'),
            ('sub-119', 'run-24'),
            ('sub-119', 'run-36'),
        ]

        # Shuffle and select
        random.shuffle(saturated_candidates)
        selected = saturated_candidates[:n_examples]

        print(f"\nSelected saturated examples: {selected}")
        return selected

    def find_normal_examples(self, n_examples: int = 3) -> list:
        """Find normal (non-seizure) signal examples from test set."""
        examples = []

        # Test patients
        test_patients = [f'sub-{i:03d}' for i in range(97, 126)]
        random.shuffle(test_patients)

        for subject_id in test_patients:
            if len(examples) >= n_examples:
                break

            subject_dir = self.base_dir / subject_id / 'ses-01'
            ecg_dir = subject_dir / 'ecg'
            eeg_dir = subject_dir / 'eeg'

            if not ecg_dir.exists():
                continue

            # Get all ECG runs
            ecg_files = list(ecg_dir.glob('*_ecg.edf'))

            for ecg_file in ecg_files:
                if len(examples) >= n_examples:
                    break

                # Extract run ID
                run_parts = ecg_file.stem.split('_')
                run_id = [p for p in run_parts if p.startswith('run-')][0]

                # Check if this run has seizures
                events_file = eeg_dir / f"{subject_id}_ses-01_task-szMonitoring_{run_id}_events.tsv"

                if events_file.exists():
                    try:
                        events_df = pd.read_csv(events_file, sep='\t')
                        if 'eventType' in events_df.columns:
                            seizures = events_df[
                                (events_df['eventType'].str.startswith('sz_', na=False)) &
                                (events_df['eventType'] != 'bckg') &
                                (events_df['eventType'] != 'impd')
                            ]
                            if len(seizures) > 0:
                                continue  # Skip runs with seizures
                    except:
                        continue

                examples.append((subject_id, run_id))

        print(f"\nSelected normal examples: {examples}")
        return examples

    def find_seizure_examples(self, n_examples: int = 3) -> list:
        """Find seizure signal examples from test set."""
        examples = []

        # Test patients
        test_patients = [f'sub-{i:03d}' for i in range(97, 126)]
        random.shuffle(test_patients)

        for subject_id in test_patients:
            if len(examples) >= n_examples:
                break

            subject_dir = self.base_dir / subject_id / 'ses-01'
            eeg_dir = subject_dir / 'eeg'
            ecg_dir = subject_dir / 'ecg'

            if not eeg_dir.exists() or not ecg_dir.exists():
                continue

            # Get event files
            event_files = list(eeg_dir.glob('*_events.tsv'))

            for event_file in event_files:
                if len(examples) >= n_examples:
                    break

                # Extract run ID
                run_parts = event_file.stem.split('_')
                run_id = [p for p in run_parts if p.startswith('run-')][0]

                # Check if ECG file exists
                ecg_file = ecg_dir / f"{subject_id}_ses-01_task-szMonitoring_{run_id}_ecg.edf"
                if not ecg_file.exists():
                    continue

                # Check for seizures
                try:
                    events_df = pd.read_csv(event_file, sep='\t')
                    if 'eventType' not in events_df.columns:
                        continue

                    seizures = events_df[
                        (events_df['eventType'].str.startswith('sz_', na=False)) &
                        (events_df['eventType'] != 'bckg') &
                        (events_df['eventType'] != 'impd')
                    ]

                    if len(seizures) > 0:
                        # Get first seizure
                        first_seizure = seizures.iloc[0]
                        onset = float(first_seizure['onset'])
                        duration = float(first_seizure['duration'])

                        examples.append({
                            'subject_id': subject_id,
                            'run_id': run_id,
                            'seizure_onset': onset,
                            'seizure_duration': duration
                        })
                except Exception as e:
                    print(f"  Warning: Error reading {event_file}: {e}")
                    continue

        print(f"\nSelected seizure examples:")
        for ex in examples:
            print(f"  {ex['subject_id']} {ex['run_id']}: onset={ex['seizure_onset']:.1f}s, duration={ex['seizure_duration']:.1f}s")

        return examples

    def generate_all_plots(self):
        """Generate all example plots."""
        print("="*80)
        print("GENERATING ECG EXAMPLE PLOTS")
        print("="*80)
        print(f"Parameters:")
        print(f"  - Plot duration: 10 seconds")
        print(f"  - Preprocessing: Bandpass 0.5-40 Hz, Downsample 250→8 Hz")
        print(f"  - Test set only: sub-097 to sub-125")
        print(f"  - Output format: PNG, 300 DPI")
        print()

        # 1. Saturated signals (only raw)
        print("\n" + "="*80)
        print("1. SATURATED SIGNALS (raw only)")
        print("="*80)

        saturated_examples = self.find_saturated_examples(n_examples=3)

        for idx, (subject_id, run_id) in enumerate(saturated_examples, 1):
            print(f"\nProcessing saturated example {idx}/3: {subject_id} {run_id}")

            # Find saturated segment automatically
            segment = self.find_saturated_segment(subject_id, run_id, segment_duration=10.0)

            if segment is None:
                print(f"  ✗ Failed to load segment")
                continue

            # Plot raw only
            output_path = self.saturated_dir / f"saturated_{idx}_{subject_id}_{run_id}.png"

            title = f"Saturated ECG Signal #{idx}: {subject_id} {run_id}"
            if 'saturation_reason' in segment and segment['saturation_reason'] != 'fallback':
                title += f" ({segment['saturation_reason']})"

            self.plot_raw_only(segment, output_path, title=title)

        # 2. Normal signals (before and after)
        print("\n" + "="*80)
        print("2. NORMAL SIGNALS (before and after preprocessing)")
        print("="*80)

        normal_examples = self.find_normal_examples(n_examples=3)

        for idx, (subject_id, run_id) in enumerate(normal_examples, 1):
            print(f"\nProcessing normal example {idx}/3: {subject_id} {run_id}")

            # Load random 10-second segment
            segment = self.load_ecg_segment(subject_id, run_id,
                                           start_time=60.0,  # Start at 1 minute
                                           duration=5.0)

            if segment is None:
                print(f"  ✗ Failed to load segment")
                continue

            # Plot before and after
            output_path = self.normal_dir / f"normal_{idx}_{subject_id}_{run_id}.png"
            #self.plot_before_after(segment, output_path,
                                  #title=f"Normal ECG Signal #{idx}: {subject_id} {run_id}")

        # 3. Seizure signals (before and after)
        print("\n" + "="*80)
        print("3. SEIZURE SIGNALS (before and after preprocessing)")
        print("="*80)

        seizure_examples = self.find_seizure_examples(n_examples=3)

        for idx, example in enumerate(seizure_examples, 1):
            subject_id = example['subject_id']
            run_id = example['run_id']
            onset = example['seizure_onset']
            duration = example['seizure_duration']

            print(f"\nProcessing seizure example {idx}/3: {subject_id} {run_id}")
            print(f"  Seizure onset: {onset:.1f}s, duration: {duration:.1f}s")

            # Load 10-second segment starting at seizure onset
            segment = self.load_ecg_segment(subject_id, run_id,
                                           start_time=onset,
                                           duration=5.0)

            if segment is None:
                print(f"  ✗ Failed to load segment")
                continue

            # Plot before and after
            output_path = self.seizure_dir / f"seizure_{idx}_{subject_id}_{run_id}.png"
            #self.plot_before_after(segment, output_path,
                                  #title=f"Seizure ECG Signal #{idx}: {subject_id} {run_id} (onset at t={onset:.1f}s)")

        print("\n" + "="*80)
        print("PLOTTING COMPLETE!")
        print("="*80)
        print(f"\nPlots saved to:")
        print(f"  - Saturated: {self.saturated_dir}")
        print(f"  - Normal: {self.normal_dir}")
        print(f"  - Seizure: {self.seizure_dir}")


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Paths
    base_dir = Path(RAW_DATA_PATH)
    output_dir = Path(__file__).parent / 'results' / 'ecg_plots'

    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        print("Please check the RAW_DATA_PATH in config.py")
        return

    # Create plotter
    plotter = ECGPlotter(base_dir, output_dir)

    # Generate all plots
    plotter.generate_all_plots()

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
