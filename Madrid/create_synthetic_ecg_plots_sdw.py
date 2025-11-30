#!/usr/bin/env python3
"""
Synthetic ECG Plot Generator with SDW Demonstration
Creates synthetic ECG signals with controlled anomalies to demonstrate the effectiveness of SDW.
Shows how SDW helps detect pre-ictal and post-ictal anomalies that would be missed otherwise.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime


class SyntheticECGGenerator:
    """Generate synthetic ECG signals with controlled anomalies."""

    def __init__(
        self,
        output_dir: str = None,
        pre_seizure_minutes: float = 5.0,
        post_seizure_minutes: float = 3.0,
        fs: int = 125
    ):
        """
        Initialize synthetic ECG generator.

        Args:
            output_dir: Directory to save plots
            pre_seizure_minutes: Minutes before seizure for SDW
            post_seizure_minutes: Minutes after seizure for SDW
            fs: Sampling frequency in Hz
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./synthetic_ecg_plots_sdw")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.pre_seizure_seconds = pre_seizure_minutes * 60.0
        self.post_seizure_seconds = post_seizure_minutes * 60.0
        self.fs = fs

        # Color scheme
        self.colors = {
            'pre': '#FFEB3B',      # Yellow
            'ictal': '#FF5252',     # Red
            'post': '#64B5F6',      # Blue
            'normal': '#FFFFFF'     # White
        }

    def generate_qrs_complex(self, duration: float, hr: float = 75.0) -> np.ndarray:
        """
        Generate realistic QRS complexes (simplified ECG).

        Args:
            duration: Duration in seconds
            hr: Heart rate in BPM

        Returns:
            Synthetic ECG signal
        """
        n_samples = int(duration * self.fs)
        signal = np.zeros(n_samples)
        time = np.arange(n_samples) / self.fs

        # Calculate inter-beat interval
        ibi = 60.0 / hr  # seconds per beat

        # Generate QRS complexes
        for beat_time in np.arange(0, duration, ibi):
            beat_idx = int(beat_time * self.fs)

            # P wave (small positive deflection)
            p_start = beat_idx - int(0.15 * self.fs)
            p_duration = int(0.08 * self.fs)
            if p_start >= 0 and p_start + p_duration < n_samples:
                p_wave = 0.2 * np.sin(np.linspace(0, np.pi, p_duration))
                signal[p_start:p_start + p_duration] += p_wave

            # QRS complex (sharp spike)
            qrs_start = beat_idx - int(0.04 * self.fs)
            qrs_duration = int(0.08 * self.fs)
            if qrs_start >= 0 and qrs_start + qrs_duration < n_samples:
                # Q wave (small negative)
                q_samples = int(0.02 * self.fs)
                signal[qrs_start:qrs_start + q_samples] -= 0.3

                # R wave (large positive spike)
                r_samples = int(0.04 * self.fs)
                r_wave = 1.5 * np.sin(np.linspace(0, np.pi, r_samples))
                signal[qrs_start + q_samples:qrs_start + q_samples + r_samples] += r_wave

                # S wave (small negative)
                s_samples = int(0.02 * self.fs)
                signal[qrs_start + q_samples + r_samples:qrs_start + qrs_duration] -= 0.2

            # T wave (broader positive deflection)
            t_start = beat_idx + int(0.15 * self.fs)
            t_duration = int(0.15 * self.fs)
            if t_start >= 0 and t_start + t_duration < n_samples:
                t_wave = 0.4 * np.sin(np.linspace(0, np.pi, t_duration))
                signal[t_start:t_start + t_duration] += t_wave

        # Removed baseline wander and noise for perfect ECG
        # Only anomalies should be visible

        return signal

    def add_anomaly(
        self,
        signal: np.ndarray,
        anomaly_type: str,
        center_time: float,
        duration: float = 10.0
    ) -> np.ndarray:
        """
        Add anomaly/disruption to the signal.

        Args:
            signal: ECG signal
            anomaly_type: Type of anomaly ('amplitude', 'frequency', 'noise', 'arrhythmia')
            center_time: Center time of anomaly in seconds
            duration: Duration of anomaly in seconds

        Returns:
            Modified signal with anomaly
        """
        modified_signal = signal.copy()
        center_idx = int(center_time * self.fs)
        half_duration = int((duration / 2) * self.fs)

        start_idx = max(0, center_idx - half_duration)
        end_idx = min(len(signal), center_idx + half_duration)

        if anomaly_type == 'amplitude':
            # Increased amplitude (sign of autonomic changes)
            modified_signal[start_idx:end_idx] *= 1.8

        elif anomaly_type == 'frequency':
            # Increased heart rate (tachycardia)
            segment_duration = (end_idx - start_idx) / self.fs
            fast_hr = 120.0  # Faster heart rate
            anomaly_segment = self.generate_qrs_complex(segment_duration, hr=fast_hr)
            modified_signal[start_idx:end_idx] = anomaly_segment

        elif anomaly_type == 'noise':
            # High-frequency noise/artifact (random noise, not sinusoidal)
            noise_amplitude = 0.6
            noise = np.random.normal(0, noise_amplitude, end_idx - start_idx)
            modified_signal[start_idx:end_idx] += noise

        elif anomaly_type == 'arrhythmia':
            # Irregular rhythm
            segment_duration = (end_idx - start_idx) / self.fs
            # Create irregular beats
            irregular_signal = np.zeros(end_idx - start_idx)
            beat_times = [0]
            while beat_times[-1] < segment_duration:
                # Random IBI variation
                ibi = np.random.uniform(0.4, 1.0)
                beat_times.append(beat_times[-1] + ibi)

            for beat_time in beat_times[:-1]:
                beat_idx = int(beat_time * self.fs)
                qrs_duration = int(0.08 * self.fs)
                if beat_idx + qrs_duration < len(irregular_signal):
                    # Variable amplitude
                    amplitude = np.random.uniform(0.8, 1.5)
                    r_wave = amplitude * np.sin(np.linspace(0, np.pi, qrs_duration))
                    irregular_signal[beat_idx:beat_idx + qrs_duration] += r_wave

            modified_signal[start_idx:end_idx] = irregular_signal

        return modified_signal

    def create_demo_signal(
        self,
        scenario: int
    ) -> Tuple[np.ndarray, float, float, Dict[str, Dict]]:
        """
        Create a complete demo signal with seizure and anomalies.

        Args:
            scenario: Scenario number (1-5) for different anomaly patterns

        Returns:
            Tuple of (signal, seizure_start, seizure_end, anomaly_info)
        """
        # Total duration: 12 minutes (shorter for better visualization)
        total_duration = 12 * 60  # seconds

        # Seizure timing (middle of recording)
        seizure_start = 7 * 60  # 7 minutes in
        seizure_duration = 10  # 10 seconds (short seizure)
        seizure_end = seizure_start + seizure_duration

        # Generate base signal
        signal = self.generate_qrs_complex(total_duration, hr=75.0)

        # Define anomaly scenarios
        # Pre-ictal anomalies start 10 seconds BEFORE seizure (early warning signs)
        anomaly_configs = {
            1: {  # Amplitude changes
                'pre': ('amplitude', seizure_start - 5, 3),
                'ictal': ('amplitude', seizure_start + 5, 8),
                'post': ('amplitude', seizure_end + 10, 8)
            },
            2: {  # Frequency changes (tachycardia)
                'pre': ('frequency', seizure_start - 5, 3),
                'ictal': ('frequency', seizure_start + 5, 10),
                'post': ('frequency', seizure_end + 10, 10)
            },
            3: {  # High-frequency noise
                'pre': ('noise', seizure_start - 5, 3),
                'ictal': ('noise', seizure_start + 5, 8),
                'post': ('noise', seizure_end + 5, 3)
            },
            4: {  # Arrhythmia
                'pre': ('arrhythmia', seizure_start - 5, 3),
                'ictal': ('arrhythmia', seizure_start + 5, 10),
                'post': ('arrhythmia', seizure_end + 5, 3)
            },
            5: {  # Mixed anomalies
                'pre': ('amplitude', seizure_start - 5, 3),
                'ictal': ('frequency', seizure_start + 5, 9),
                'post': ('noise', seizure_end + 5, 3)
            }
        }

        config = anomaly_configs.get(scenario, anomaly_configs[1])
        anomaly_info = {}

        # Add anomalies
        for region, (anom_type, center_time, duration) in config.items():
            signal = self.add_anomaly(signal, anom_type, center_time, duration)
            anomaly_info[region] = {
                'type': anom_type,
                'time': center_time,
                'duration': duration
            }

        # Also add one anomaly outside SDW window
        # SDW is [-5min, +3min], so place outside anomaly at -6 minutes
        outside_time = seizure_start - 360  # -6 minutes (outside SDW)
        signal = self.add_anomaly(signal, 'noise', outside_time, 8)
        anomaly_info['outside'] = {
            'type': 'noise',
            'time': outside_time,
            'duration': 8
        }

        return signal, seizure_start, seizure_end, anomaly_info

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
            use_sdw: If True, use extended SDW window

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
        seizure_start: float,
        seizure_end: float,
        anomaly_info: Dict[str, Dict],
        use_sdw: bool,
        scenario: int
    ):
        """
        Create ECG plot with main view and 4 zoom subplots.

        Args:
            signal: ECG signal
            seizure_start: Seizure start time in seconds
            seizure_end: Seizure end time in seconds
            anomaly_info: Dict with anomaly information
            use_sdw: Whether to use SDW visualization
            scenario: Scenario number
        """
        # Apply scientific plot style from figures.ipynb (larger fonts)
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 150
        })

        # Create time array
        time = np.arange(len(signal)) / self.fs
        signal_duration = len(signal) / self.fs

        # Calculate plot window - only 30 seconds total
        # Show 10s before (pre-ictal anomaly), 10s seizure, 10s after (post-ictal anomaly)
        plot_start = max(0, seizure_start - 10)  # 10 seconds before (shows pre-ictal anomaly)
        plot_end = min(signal_duration, seizure_end + 10)  # 10 seconds after (shows post-ictal anomaly)

        # Create figure with only main plot (no subplots)
        fig, ax_main = plt.subplots(figsize=(12, 6))

        # Plot main ECG
        mask = (time >= plot_start) & (time <= plot_end)
        ax_main.plot(time[mask], signal[mask], 'k-', linewidth=0.5, alpha=0.8)

        # Add colored background regions
        sdw_start = seizure_start - self.pre_seizure_seconds
        sdw_end = seizure_end + self.post_seizure_seconds

        if use_sdw:
            # Pre-ictal region (SDW)
            ax_main.axvspan(sdw_start, seizure_start, alpha=0.3, color=self.colors['pre'],
                          label=f'Pre-ictal SDW (-{self.pre_seizure_seconds/60:.0f}min)')
        else:
            # Show pre-ictal without SDW
            if seizure_start > plot_start:
                ax_main.axvspan(plot_start, seizure_start, alpha=0.15, color=self.colors['pre'],
                              label='Pre-ictal (no SDW)')

        # Ictal region
        ax_main.axvspan(seizure_start, seizure_end, alpha=0.3, color=self.colors['ictal'],
                       label='Ictal (Seizure)')

        if use_sdw:
            # Post-ictal region (SDW)
            ax_main.axvspan(seizure_end, sdw_end, alpha=0.3, color=self.colors['post'],
                          label=f'Post-ictal SDW (+{self.post_seizure_seconds/60:.0f}min)')
        else:
            # Show post-ictal without SDW
            if seizure_end < plot_end:
                ax_main.axvspan(seizure_end, plot_end, alpha=0.15, color=self.colors['post'],
                              label='Post-ictal (no SDW)')

        # Count TP and FP
        tp_count = 0
        fp_count = 0

        # Track which labels have been added to legend
        added_labels = set()

        # Plot anomalies on main plot
        for region, info in anomaly_info.items():
            anom_time = info['time']
            anom_type = info['type']
            label = self.classify_anomaly(anom_time, seizure_start, seizure_end, use_sdw)

            if label == 'TP':
                tp_count += 1
            else:
                fp_count += 1

            color = 'red' if label == 'FP' else 'green'

            # Only add label to legend if not already added
            legend_label = 'True Positive' if label == 'TP' else 'False Positive'
            if legend_label not in added_labels:
                display_label = legend_label
                added_labels.add(legend_label)
            else:
                display_label = None  # Don't add to legend again

            # Get signal value at anomaly time
            idx = int(anom_time * self.fs)
            if 0 <= idx < len(signal):
                y_val = signal[idx]
                ax_main.scatter(anom_time, y_val, s=200, c=color, marker='o',
                              edgecolors='black', linewidths=2, zorder=10,
                              label=display_label)

        ax_main.set_xlabel('Time (seconds)')
        ax_main.set_ylabel('ECG Amplitude (normalized)')
        ax_main.set_xlim(plot_start, plot_end)

        # Add subtle gridlines (scientific style from figures.ipynb)
        ax_main.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax_main.set_axisbelow(True)  # Grid behind data

        ax_main.legend(loc='upper right', ncol=2, framealpha=0.9)

        sdw_text = "WITH SDW" if use_sdw else "WITHOUT SDW"
        sensitivity = (tp_count / len(anomaly_info)) * 100 if anomaly_info else 0

        ax_main.set_title(
            f'Synthetic ECG - Scenario {scenario} - {sdw_text}',
            pad=15  # Padding like in figures.ipynb
        )

        # Apply tight layout (scientific style from figures.ipynb)
        plt.tight_layout()

        # Save plot with high resolution
        sdw_suffix = "with_sdw" if use_sdw else "without_sdw"

        # Save as PNG (high DPI like figures.ipynb)
        filename_png = f"synthetic_scenario{scenario}_{sdw_suffix}.png"
        filepath_png = self.output_dir / filename_png
        plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
        print(f"Saved PNG: {filepath_png}")

        # Also save as PDF (vector format for publications)
        filename_pdf = f"synthetic_scenario{scenario}_{sdw_suffix}.pdf"
        filepath_pdf = self.output_dir / filename_pdf
        plt.savefig(filepath_pdf, bbox_inches='tight')
        print(f"Saved PDF: {filepath_pdf}")

        plt.close(fig)

    def generate_all_scenarios(self, num_scenarios: int = 5):
        """
        Generate plots for all scenarios.

        Args:
            num_scenarios: Number of scenarios to generate (default: 5)
        """
        print(f"Generating synthetic ECG plots with SDW demonstration...")
        print(f"Output directory: {self.output_dir}")
        print(f"SDW window: -{self.pre_seizure_seconds/60:.0f} min to +{self.post_seizure_seconds/60:.0f} min")
        print(f"Sampling rate: {self.fs} Hz")
        print(f"\nScenarios:")
        print("  1: Amplitude changes (autonomic)")
        print("  2: Frequency changes (tachycardia)")
        print("  3: High-frequency noise")
        print("  4: Arrhythmia (irregular rhythm)")
        print("  5: Mixed anomalies")
        print(f"\nTotal plots to create: {num_scenarios * 2} (without SDW + with SDW)\n")

        for scenario in range(1, num_scenarios + 1):
            print(f"[Scenario {scenario}/{num_scenarios}]")

            # Generate signal
            signal, seizure_start, seizure_end, anomaly_info = self.create_demo_signal(scenario)

            print(f"  Signal duration: {len(signal)/self.fs:.0f}s")
            print(f"  Seizure: {seizure_start:.0f}s - {seizure_end:.0f}s")
            print(f"  Anomalies: {list(anomaly_info.keys())}")

            # Create plot WITHOUT SDW
            print(f"  Creating plot WITHOUT SDW...")
            self.create_plot(signal, seizure_start, seizure_end, anomaly_info,
                           use_sdw=False, scenario=scenario)

            # Create plot WITH SDW
            print(f"  Creating plot WITH SDW...")
            self.create_plot(signal, seizure_start, seizure_end, anomaly_info,
                           use_sdw=True, scenario=scenario)

            print()

        print(f"\n{'='*60}")
        print(f"PLOT GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total plots created: {num_scenarios * 2}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nKey findings:")
        print(f"  • WITHOUT SDW: Only ictal anomalies detected as TP")
        print(f"  • WITH SDW: Pre- and post-ictal anomalies also detected as TP")
        print(f"  • SDW increases sensitivity by detecting seizure-related changes")
        print(f"    that occur before and after the visible seizure period")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic ECG plots demonstrating SDW effectiveness"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./synthetic_ecg_plots_sdw",
        help="Output directory for plots (default: ./synthetic_ecg_plots_sdw)"
    )
    parser.add_argument(
        "-n", "--num-scenarios",
        type=int,
        default=5,
        help="Number of scenarios to generate (default: 5)"
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
    parser.add_argument(
        "--fs",
        type=int,
        default=125,
        help="Sampling frequency in Hz (default: 125)"
    )

    args = parser.parse_args()

    # Create generator
    generator = SyntheticECGGenerator(
        output_dir=args.output_dir,
        pre_seizure_minutes=args.pre_seizure_minutes,
        post_seizure_minutes=args.post_seizure_minutes,
        fs=args.fs
    )

    # Generate plots
    generator.generate_all_scenarios(args.num_scenarios)


if __name__ == "__main__":
    main()
