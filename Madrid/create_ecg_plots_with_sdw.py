#!/usr/bin/env python3
"""
ECG Plot Generator with SDW Visualization
Creates 20 ECG plots showing seizure detection with and without Seizure Detection Window (SDW).
Each plot includes main view and 4 zoom subplots for anomalies.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import random
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Information', 'Data', 'seizeit2_main'))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import PREPROCESSED_DATA_PATH


class ECGPlotGenerator:
    """Generate ECG plots with SDW visualization and anomaly detection."""

    def __init__(
        self,
        preprocessed_dir: str,
        output_dir: str = None,
        pre_seizure_minutes: float = 5.0,
        post_seizure_minutes: float = 3.0
    ):
        """
        Initialize plot generator.

        Args:
            preprocessed_dir: Directory containing preprocessed ECG data
            output_dir: Directory to save plots (default: ./ecg_plots_sdw)
            pre_seizure_minutes: Minutes before seizure for SDW
            post_seizure_minutes: Minutes after seizure for SDW
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("./ecg_plots_sdw")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.pre_seizure_seconds = pre_seizure_minutes * 60.0
        self.post_seizure_seconds = post_seizure_minutes * 60.0

        # Color scheme
        self.colors = {
            'pre': '#FFEB3B',      # Yellow
            'ictal': '#FF5252',     # Red
            'post': '#64B5F6',      # Blue
            'normal': '#FFFFFF'     # White
        }

        # Fixed anomaly positions (relative to seizure)
        self.anomaly_positions = {
            'outside': -420,   # -7 minutes (outside SDW)
            'pre': -180,       # -3 minutes (within SDW)
            'ictal': 0,        # 0 (middle of seizure, relative position)
            'post': 90         # +1.5 minutes (within SDW)
        }

    def is_test_subject(self, subject_id: str) -> bool:
        """Check if subject is in test set (sub097-sub125)."""
        match = re.search(r'sub-(\d{3})', subject_id)
        if match:
            subject_num = int(match.group(1))
            return 97 <= subject_num <= 125
        return False

    def find_preprocessed_files(self) -> List[Path]:
        """Find all preprocessed files in the test set."""
        all_files = list(self.preprocessed_dir.glob("sub-*_run-*_preprocessed.pkl"))
        test_files = [f for f in all_files if self.is_test_subject(f.name)]
        return test_files

    def load_preprocessed_data(self, filepath: Path) -> Optional[Dict]:
        """Load preprocessed ECG data."""
        try:
            return pd.read_pickle(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def select_seizure_files(self, num_files: int = 20) -> List[Tuple[Path, Dict]]:
        """
        Select random files from test set that contain seizures.

        Args:
            num_files: Number of files to select

        Returns:
            List of (filepath, data) tuples
        """
        test_files = self.find_preprocessed_files()
        print(f"Found {len(test_files)} test set files")

        # Load and filter for files with seizures
        seizure_files = []
        for filepath in test_files:
            data = self.load_preprocessed_data(filepath)
            if data and data.get('total_seizures', 0) > 0:
                seizure_files.append((filepath, data))

        print(f"Found {len(seizure_files)} files with seizures in test set")

        # Randomly select
        if len(seizure_files) > num_files:
            selected = random.sample(seizure_files, num_files)
        else:
            selected = seizure_files
            print(f"Warning: Only {len(seizure_files)} seizure files available, using all")

        return selected

    def get_seizure_timing(self, data: Dict) -> Optional[Tuple[float, float]]:
        """
        Extract first seizure timing from preprocessed data.

        Returns:
            Tuple of (seizure_start, seizure_end) in seconds, or None
        """
        try:
            channels = data.get('channels', [])
            if not channels:
                return None

            # Get first channel's metadata
            first_channel = channels[0]
            metadata = first_channel.get('metadata', [])

            # Find first window with seizure
            for meta in metadata:
                seizure_segments = meta.get('seizure_segments', [])
                if seizure_segments:
                    # Get first seizure segment
                    first_seizure = seizure_segments[0]
                    return (
                        first_seizure['start_time_absolute'],
                        first_seizure['end_time_absolute']
                    )

            return None
        except Exception as e:
            print(f"Error extracting seizure timing: {e}")
            return None

    def get_ecg_signal(self, data: Dict) -> Tuple[np.ndarray, int]:
        """
        Extract ECG signal from preprocessed data.

        Returns:
            Tuple of (signal, sampling_rate)
        """
        channels = data.get('channels', [])
        if not channels:
            raise ValueError("No channels found")

        first_channel = channels[0]
        windows = first_channel.get('windows', [])

        if not windows:
            raise ValueError("No windows found")

        # Concatenate all windows to get full signal
        signal = np.concatenate(windows)
        fs = first_channel.get('processed_fs', 125)

        return signal, int(fs)

    def place_anomalies(
        self,
        seizure_start: float,
        seizure_end: float,
        signal_duration: float
    ) -> Dict[str, float]:
        """
        Calculate absolute time positions for anomalies.

        Args:
            seizure_start: Seizure start time in seconds
            seizure_end: Seizure end time in seconds
            signal_duration: Total signal duration in seconds

        Returns:
            Dict with anomaly positions {'outside': time, 'pre': time, 'ictal': time, 'post': time}
        """
        seizure_middle = (seizure_start + seizure_end) / 2

        positions = {}
        for name, offset in self.anomaly_positions.items():
            if name == 'ictal':
                # Place in middle of seizure
                positions[name] = seizure_middle
            else:
                # Offset from seizure start
                positions[name] = seizure_start + offset

        # Ensure all positions are within signal bounds
        for name in positions:
            positions[name] = max(0, min(positions[name], signal_duration - 1))

        return positions

    def classify_anomaly(
        self,
        anomaly_time: float,
        seizure_start: float,
        seizure_end: float,
        use_sdw: bool
    ) -> str:
        """
        Classify anomaly as TP or FP.

        Args:
            anomaly_time: Time of anomaly in seconds
            seizure_start: Seizure start time
            seizure_end: Seizure end time
            use_sdw: If True, use extended SDW window; if False, use only ictal period

        Returns:
            'TP' or 'FP'
        """
        if use_sdw:
            # With SDW: TP if within [-5min, +3min] around seizure
            sdw_start = seizure_start - self.pre_seizure_seconds
            sdw_end = seizure_end + self.post_seizure_seconds

            if sdw_start <= anomaly_time <= sdw_end:
                return 'TP'
            else:
                return 'FP'
        else:
            # Without SDW: TP only during ictal period
            if seizure_start <= anomaly_time <= seizure_end:
                return 'TP'
            else:
                return 'FP'

    def create_plot(
        self,
        signal: np.ndarray,
        fs: int,
        seizure_start: float,
        seizure_end: float,
        anomaly_positions: Dict[str, float],
        use_sdw: bool,
        subject_id: str,
        run_id: str,
        plot_index: int
    ):
        """
        Create ECG plot with main view and 4 zoom subplots.

        Args:
            signal: ECG signal
            fs: Sampling frequency
            seizure_start: Seizure start time in seconds
            seizure_end: Seizure end time in seconds
            anomaly_positions: Dict with anomaly positions
            use_sdw: Whether to use SDW visualization
            subject_id: Subject identifier
            run_id: Run identifier
            plot_index: Index of the plot (1-20)
        """
        # Create time array
        time = np.arange(len(signal)) / fs
        signal_duration = len(signal) / fs

        # Calculate plot window (full context)
        plot_start = max(0, seizure_start - self.pre_seizure_seconds - 60)  # Extra 1 min before
        plot_end = min(signal_duration, seizure_end + self.post_seizure_seconds + 60)  # Extra 1 min after

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

        # Main plot (top, spanning all columns)
        ax_main = fig.add_subplot(gs[0:2, :])

        # Zoom subplots (bottom row)
        ax_zoom = [
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2]),
            fig.add_subplot(gs[2, 3])
        ]

        # Plot main ECG
        mask = (time >= plot_start) & (time <= plot_end)
        ax_main.plot(time[mask], signal[mask], 'k-', linewidth=0.5, alpha=0.8)

        # Add colored background regions
        sdw_start = seizure_start - self.pre_seizure_seconds
        sdw_end = seizure_end + self.post_seizure_seconds

        if use_sdw:
            # Pre-ictal region (SDW)
            ax_main.axvspan(sdw_start, seizure_start, alpha=0.3, color=self.colors['pre'],
                          label=f'Pre-ictal (SDW: -{self.pre_seizure_seconds/60:.0f}min)')
        else:
            # Show pre-ictal without SDW label
            if seizure_start > plot_start:
                ax_main.axvspan(plot_start, seizure_start, alpha=0.15, color=self.colors['pre'],
                              label='Pre-ictal')

        # Ictal region
        ax_main.axvspan(seizure_start, seizure_end, alpha=0.3, color=self.colors['ictal'],
                       label='Ictal (Seizure)')

        if use_sdw:
            # Post-ictal region (SDW)
            ax_main.axvspan(seizure_end, sdw_end, alpha=0.3, color=self.colors['post'],
                          label=f'Post-ictal (SDW: +{self.post_seizure_seconds/60:.0f}min)')
        else:
            # Show post-ictal without SDW label
            if seizure_end < plot_end:
                ax_main.axvspan(seizure_end, plot_end, alpha=0.15, color=self.colors['post'],
                              label='Post-ictal')

        # Plot anomalies on main plot
        for name, anom_time in anomaly_positions.items():
            label = self.classify_anomaly(anom_time, seizure_start, seizure_end, use_sdw)
            color = 'red' if label == 'FP' else 'green'

            # Get signal value at anomaly time
            idx = int(anom_time * fs)
            if 0 <= idx < len(signal):
                y_val = signal[idx]
                ax_main.scatter(anom_time, y_val, s=200, c=color, marker='o',
                              edgecolors='black', linewidths=2, zorder=10,
                              label=f'{name.capitalize()}: {label}')

        ax_main.set_xlabel('Time (seconds)', fontsize=12)
        ax_main.set_ylabel('ECG Amplitude (normalized)', fontsize=12)
        ax_main.set_xlim(plot_start, plot_end)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='upper right', fontsize=10)

        sdw_text = "WITH SDW" if use_sdw else "WITHOUT SDW"
        ax_main.set_title(
            f'ECG Plot #{plot_index}: {subject_id} {run_id} - {sdw_text}\n'
            f'Seizure: {seizure_start:.1f}s - {seizure_end:.1f}s (Duration: {seizure_end-seizure_start:.1f}s)',
            fontsize=14, fontweight='bold'
        )

        # Create zoom subplots for each anomaly
        anomaly_names = ['outside', 'pre', 'ictal', 'post']
        zoom_window = 30  # ±30 seconds around anomaly

        for i, name in enumerate(anomaly_names):
            anom_time = anomaly_positions[name]
            label = self.classify_anomaly(anom_time, seizure_start, seizure_end, use_sdw)

            # Define zoom window
            zoom_start = max(0, anom_time - zoom_window)
            zoom_end = min(signal_duration, anom_time + zoom_window)

            # Plot zoomed ECG
            zoom_mask = (time >= zoom_start) & (time <= zoom_end)
            ax_zoom[i].plot(time[zoom_mask], signal[zoom_mask], 'k-', linewidth=1)

            # Add background color based on region
            if use_sdw:
                # Color based on SDW regions
                if sdw_start <= anom_time <= seizure_start:
                    bg_color = self.colors['pre']
                    region_name = 'Pre (SDW)'
                elif seizure_start <= anom_time <= seizure_end:
                    bg_color = self.colors['ictal']
                    region_name = 'Ictal'
                elif seizure_end <= anom_time <= sdw_end:
                    bg_color = self.colors['post']
                    region_name = 'Post (SDW)'
                else:
                    bg_color = self.colors['normal']
                    region_name = 'Outside SDW'
            else:
                # Color only ictal
                if seizure_start <= anom_time <= seizure_end:
                    bg_color = self.colors['ictal']
                    region_name = 'Ictal'
                else:
                    bg_color = self.colors['normal']
                    region_name = 'Outside Ictal'

            ax_zoom[i].axvspan(zoom_start, zoom_end, alpha=0.2, color=bg_color)

            # Mark anomaly
            idx = int(anom_time * fs)
            if 0 <= idx < len(signal):
                y_val = signal[idx]
                color = 'red' if label == 'FP' else 'green'
                ax_zoom[i].scatter(anom_time, y_val, s=150, c=color, marker='o',
                                 edgecolors='black', linewidths=2, zorder=10)

            ax_zoom[i].set_xlabel('Time (s)', fontsize=9)
            ax_zoom[i].set_ylabel('Amplitude', fontsize=9)
            ax_zoom[i].set_title(f'{name.capitalize()}: {label}\n{region_name}',
                                fontsize=10, fontweight='bold')
            ax_zoom[i].grid(True, alpha=0.3)
            ax_zoom[i].tick_params(labelsize=8)

        # Save plot
        sdw_suffix = "with_sdw" if use_sdw else "without_sdw"
        filename = f"plot_{plot_index:02d}_{subject_id}_{run_id}_{sdw_suffix}.png"
        filepath = self.output_dir / filename

        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)

    def generate_plots(self, num_plots: int = 20):
        """
        Generate all ECG plots.

        Args:
            num_plots: Number of different seizures to plot (default: 20)
        """
        print(f"Generating {num_plots} ECG plots with SDW visualization...")
        print(f"Preprocessed data directory: {self.preprocessed_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"SDW window: -{self.pre_seizure_seconds/60:.0f} min to +{self.post_seizure_seconds/60:.0f} min")

        # Select seizure files
        seizure_files = self.select_seizure_files(num_plots)

        if not seizure_files:
            print("No seizure files found!")
            return

        print(f"\nGenerating plots for {len(seizure_files)} seizures...")
        print(f"Total plots to create: {len(seizure_files) * 2} (without SDW + with SDW)")

        successful_plots = 0

        for i, (filepath, data) in enumerate(seizure_files, 1):
            try:
                print(f"\n[{i}/{len(seizure_files)}] Processing: {filepath.name}")

                # Extract subject and run IDs
                subject_id = data.get('subject_id', 'unknown')
                run_id = data.get('run_id', 'unknown')

                # Get seizure timing
                seizure_timing = self.get_seizure_timing(data)
                if not seizure_timing:
                    print(f"  Skipping: No seizure timing found")
                    continue

                seizure_start, seizure_end = seizure_timing
                print(f"  Seizure: {seizure_start:.1f}s - {seizure_end:.1f}s")

                # Get ECG signal
                signal, fs = self.get_ecg_signal(data)
                signal_duration = len(signal) / fs
                print(f"  Signal: {signal_duration:.1f}s @ {fs}Hz")

                # Place anomalies
                anomaly_positions = self.place_anomalies(seizure_start, seizure_end, signal_duration)
                print(f"  Anomaly positions: {anomaly_positions}")

                # Create plot WITHOUT SDW
                print(f"  Creating plot WITHOUT SDW...")
                self.create_plot(
                    signal, fs, seizure_start, seizure_end, anomaly_positions,
                    use_sdw=False, subject_id=subject_id, run_id=run_id, plot_index=i
                )

                # Create plot WITH SDW
                print(f"  Creating plot WITH SDW...")
                self.create_plot(
                    signal, fs, seizure_start, seizure_end, anomaly_positions,
                    use_sdw=True, subject_id=subject_id, run_id=run_id, plot_index=i
                )

                successful_plots += 1

            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*60}")
        print(f"PLOT GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully created plots for {successful_plots}/{len(seizure_files)} seizures")
        print(f"Total plots created: {successful_plots * 2}")
        print(f"Output directory: {self.output_dir}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ECG plots with SDW visualization"
    )
    parser.add_argument(
        "-d", "--data-dir",
        default=PREPROCESSED_DATA_PATH,
        help=f"Directory with preprocessed data (default: {PREPROCESSED_DATA_PATH})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./ecg_plots_sdw",
        help="Output directory for plots (default: ./ecg_plots_sdw)"
    )
    parser.add_argument(
        "-n", "--num-plots",
        type=int,
        default=20,
        help="Number of seizures to plot (default: 20)"
    )
    parser.add_argument(
        "--pre-seizure-minutes",
        type=float,
        default=5.0,
        help="Minutes before seizure for SDW (default: 5.0)"
    )
    parser.add_argument(
        "--post-seizure-minutes",
        type=float,
        default=3.0,
        help="Minutes after seizure for SDW (default: 3.0)"
    )

    args = parser.parse_args()

    # Create generator
    generator = ECGPlotGenerator(
        preprocessed_dir=args.data_dir,
        output_dir=args.output_dir,
        pre_seizure_minutes=args.pre_seizure_minutes,
        post_seizure_minutes=args.post_seizure_minutes
    )

    # Generate plots
    generator.generate_plots(args.num_plots)


if __name__ == "__main__":
    main()
