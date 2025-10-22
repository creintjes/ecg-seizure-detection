#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Responder Detection with Elgendi R-Peak Detection - First VALID Seizure
========================================================================

This script classifies responders based on the first VALID (analyzable) seizure
per patient. If the chronologically first seizure is too short to compute metrics,
it automatically falls back to the next seizure, and so on.

Key differences from find_responder_elgendi_first_seizure.py:
- Takes the first VALID seizure (with sufficient data for HR change calculation)
- If first seizure is too short (< 100 RR intervals), uses the next one
- Continues until a valid seizure is found or all seizures are exhausted
- Parallel processing support

Reference:
Elgendi, M. (2013). Fast QRS Detection with an Optimized Knowledge-Based Method.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal
import traceback
import sys
from typing import List, Tuple, Optional, Dict
from scipy.signal import butter, filtfilt, medfilt
import warnings
import os
import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel processing
import matplotlib.pyplot as plt
import random


# ==================== CONFIGURATION ====================

DATA_PATH = "/home/swolf/asim_shared/raw_data/ds005873-1.1.0"

# Output files
OUTPUT_TXT = "responders_and_non_responders_elgendi_first_valid_seizure.txt"
OUTPUT_CSV = "patient_responder_summary_elgendi_first_valid_seizure.csv"
PLOT_DIR = "responder_plots_first_valid_seizure"
NUM_PLOTS = 10  # Number of responders to plot

# Analysis parameters
WINDOW_RR = 100          # Rolling window for Jeppesen HR-diff calculation
BPM_THRESHOLD = 50       # Threshold for responder classification
MIN_RR_SECONDS = 0.25    # Minimum RR interval (max ~240 bpm)
MAX_RR_SECONDS = 2.0     # Maximum RR interval (min ~30 bpm)

# Elgendi method parameters (from supervisor's code)
ELGENDI_LOW = 8          # Hz - low cutoff for bandpass
ELGENDI_HIGH = 20        # Hz - high cutoff for bandpass
ELGENDI_ORDER = 3        # Filter order
ELGENDI_W1_FACTOR = 0.12 # Window factor for QRS detection
ELGENDI_W2_FACTOR = 0.65 # Window factor for beat detection
ELGENDI_BETA = 0.08      # Threshold scaling factor

# Signal quality parameters
MIN_SIGNAL_STD = 0.01    # Minimum std deviation for valid signal
MAX_SATURATION = 0.10    # Max fraction of saturated samples

# ==================== SETUP PATHS ====================

# Add seizeit2-main to path for Data/Annotation classes
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Information" / "Data" / "seizeit2-main"))

# Import preprocessing module
import importlib.util
spec = importlib.util.spec_from_file_location("preprocessing_module",
                                               Path(__file__).parent / "preprocessing.py")
pre_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre_mod)

ECGPreprocessor = getattr(pre_mod, "ECGPreprocessor", None)
if ECGPreprocessor is None:
    raise RuntimeError("ECGPreprocessor not found in preprocessing.py")

pre = ECGPreprocessor()

# ==================== PARALLEL WORKER INIT ====================

PRE = None  # Set per process

def init_worker():
    """
    Initializer for each worker process:
    - Limits BLAS threads (prevents oversubscription)
    - Initializes ECGPreprocessor (per process)
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    global PRE, ECGPreprocessor
    try:
        PRE = ECGPreprocessor()
    except Exception as e:
        PRE = None


# ==================== ELGENDI R-PEAK DETECTION ====================

def peak_detection_elgendi(ecg_data, sampling_rate,
                          low=ELGENDI_LOW,
                          high=ELGENDI_HIGH,
                          order=ELGENDI_ORDER,
                          w1factor=ELGENDI_W1_FACTOR,
                          w2factor=ELGENDI_W2_FACTOR,
                          beta=ELGENDI_BETA):
    """
    Detects R-peaks in ECG data using the Elgendi method.

    This is the implementation provided by the supervisor.

    Parameters:
        ecg_data (list or np.array): The ECG signal data.
        sampling_rate (int): The sampling rate of the ECG data in Hz.
        low (float, optional): Low cutoff frequency for bandpass filter. Default is 8 Hz.
        high (float, optional): High cutoff frequency for bandpass filter. Default is 20 Hz.
        order (int, optional): Order of the Butterworth filter. Default is 3.
        w1factor (float, optional): Factor for window size w1. Default is 0.12.
        w2factor (float, optional): Factor for window size w2. Default is 0.65.
        beta (float, optional): Scaling factor for threshold calculation. Default is 0.08.

    Returns:
        list: A list of indices representing the detected R-peaks.
    """
    def _filter_peaks(ecg_data, foundpeaks, sampling_rate, min_rr_distance=0.25):
        """
        Filters detected peaks in ECG data based on minimum RR interval distance.

        Parameters:
            ecg_data (list or np.array): The ECG signal data.
            foundpeaks (list or np.array): Indices of detected peaks in the ECG data.
            sampling_rate (int): The sampling rate of the ECG data in Hz.
            min_rr_distance (float, optional): The minimum RR interval distance in seconds.
                Peaks closer than this distance will be filtered out. Default is 0.25 seconds.

        Returns:
            list: A list of indices representing the filtered peaks.
        """
        filtered_peaks = []
        jumpnextone = False
        min_rr_samples = int(min_rr_distance * sampling_rate)

        for i in range(len(foundpeaks) - 1):
            if jumpnextone:
                jumpnextone = False
                continue

            dist = foundpeaks[i + 1] - foundpeaks[i]

            # Forwards block proximity filter
            if dist > min_rr_samples:
                # Backwards block proximity filter
                if len(filtered_peaks) == 0 or (foundpeaks[i] - filtered_peaks[-1]) > min_rr_samples:
                    filtered_peaks.append(foundpeaks[i])
            else:
                if ecg_data[foundpeaks[i]] > ecg_data[foundpeaks[i + 1]]:
                    # Backwards block proximity filter
                    if len(filtered_peaks) == 0 or (foundpeaks[i] - filtered_peaks[-1]) > min_rr_samples:
                        filtered_peaks.append(foundpeaks[i])
                    jumpnextone = True
                else:
                    # Backwards block proximity filter
                    if len(filtered_peaks) == 0 or (foundpeaks[i + 1] - filtered_peaks[-1]) > min_rr_samples:
                        filtered_peaks.append(foundpeaks[i + 1])
                    jumpnextone = True

        # Check the last peak
        if len(foundpeaks) > 0 and (len(filtered_peaks) == 0 or (foundpeaks[-1] - filtered_peaks[-1]) > min_rr_samples):
            filtered_peaks.append(foundpeaks[-1])

        return filtered_peaks

    # Bandpass filter
    nyquist = 0.5 * sampling_rate
    low_norm = low / nyquist
    high_norm = high / nyquist
    coeffs = scipy.signal.butter(order, [low_norm, high_norm], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], ecg_data)

    # First Derivative (QRS enhancement)
    diff = np.diff(filtered)
    diff = np.append(diff, diff[-1])

    # Squaring (QRS enhancement)
    squared = diff ** 2

    # Normalization
    filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
    peaks = np.zeros(len(filtered))

    w1 = int(w1factor * sampling_rate)
    w2 = int(w2factor * sampling_rate)

    # Moving average QRS
    maqrs = np.convolve(squared, np.ones(w1), mode="same") / w1

    # Moving average beat
    mabeat = np.convolve(squared, np.ones(w2), mode="same") / w2

    # Threshold calculation
    alpha = beta * np.mean(squared)
    thr1 = mabeat + alpha

    # Determination of Blocks of Interest
    blocksofinterest = maqrs > thr1
    blocksofinterest = np.append(blocksofinterest, False)

    boi = False
    for i, boi_val in enumerate(blocksofinterest):
        if boi_val and not boi:
            boi = True
            boiarea = i
        elif not boi_val and boi:
            boi = False
            # Block width filter
            if (i - boiarea) >= w1:
                peak = boiarea + np.argmax(filtered[boiarea:i])
                peaks[peak] = 1

    foundpeaks = np.where(peaks == 1)[0]

    return _filter_peaks(ecg_data, foundpeaks, sampling_rate)


def detect_r_peaks_elgendi(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Wrapper for Elgendi R-peak detection with error handling.

    Args:
        signal: ECG signal (can be raw or preprocessed)
        fs: Sampling frequency

    Returns:
        Array of R-peak indices
    """
    if signal is None or len(signal) < 10:
        return np.array([], dtype=int)

    try:
        peaks = peak_detection_elgendi(signal, int(fs))
        return np.array(peaks, dtype=int)
    except Exception as e:
        warnings.warn(f"Elgendi peak detection failed: {e}")
        return np.array([], dtype=int)


# ==================== SIGNAL QUALITY FUNCTIONS ====================

def check_signal_quality(signal: np.ndarray, fs: float) -> Tuple[bool, str]:
    """
    Assess ECG signal quality.

    Args:
        signal: ECG signal
        fs: Sampling frequency

    Returns:
        Tuple of (is_valid, reason)
    """
    if signal is None or len(signal) < fs * 5:  # At least 5 seconds
        return False, "Signal too short"

    # Check for flat line
    if np.std(signal) < MIN_SIGNAL_STD:
        return False, "Flat line detected"

    # Check for saturation
    signal_range = np.max(signal) - np.min(signal)
    if signal_range == 0:
        return False, "Zero range signal"

    # Count samples near min/max (saturation)
    threshold = 0.95 * signal_range
    saturated = np.sum((signal > (np.min(signal) + threshold)) |
                       (signal < (np.min(signal) + 0.05 * signal_range)))
    saturation_fraction = saturated / len(signal)

    if saturation_fraction > MAX_SATURATION:
        return False, f"High saturation: {saturation_fraction:.2%}"

    return True, "OK"


# ==================== PREPROCESSING FUNCTIONS ====================

def remove_baseline_wander(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Remove baseline wander using median filtering.

    Args:
        signal: Input signal
        fs: Sampling frequency

    Returns:
        Signal with baseline removed
    """
    kernel_size = int(0.2 * fs)  # 200ms window
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd

    baseline = medfilt(signal, kernel_size=kernel_size)
    return signal - baseline


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalization.

    Args:
        signal: Input signal

    Returns:
        Normalized signal (mean=0, std=1)
    """
    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-10:
        warnings.warn("Signal std too low for normalization")
        return signal - mean

    return (signal - mean) / std


def preprocess_for_elgendi(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Minimal preprocessing for Elgendi method.

    The Elgendi method includes its own bandpass filter (8-20 Hz),
    so we only apply minimal preprocessing:
    1. Remove baseline wander
    2. Normalize signal

    Args:
        signal: Raw ECG signal
        fs: Sampling frequency

    Returns:
        Preprocessed signal ready for Elgendi R-peak detection
    """
    # 1. Remove baseline wander
    signal_corrected = remove_baseline_wander(signal, fs)

    # 2. Normalize (helps with amplitude variations)
    signal_normalized = normalize_signal(signal_corrected)

    return signal_normalized


# ==================== RR INTERVAL FUNCTIONS ====================

def rr_intervals_ms_from_peaks(peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    Calculate RR intervals in milliseconds from peak indices.

    Args:
        peaks: Array of peak indices
        fs: Sampling frequency

    Returns:
        Array of RR intervals in milliseconds
    """
    if peaks.size < 2:
        return np.array([], dtype=float)

    diffs = np.diff(peaks).astype(float)
    rr_ms = (diffs / fs) * 1000.0

    # Filter physiologically implausible intervals
    valid_mask = (rr_ms >= MIN_RR_SECONDS * 1000) & (rr_ms <= MAX_RR_SECONDS * 1000)

    return rr_ms[valid_mask]


# ==================== HEART RATE ANALYSIS ====================

def calculate_hr_diff_jeppesen(rr_intervals_ms: np.ndarray,
                               window_size: int = WINDOW_RR) -> np.ndarray:
    """
    Calculate heart rate difference using Jeppesen method.

    Uses central differences and rolling sum over specified window.

    Args:
        rr_intervals_ms: RR intervals in milliseconds
        window_size: Rolling window size (default: 100)

    Returns:
        Array of HR difference sums
    """
    rr = np.asarray(rr_intervals_ms, dtype=float)

    if rr.size < 3:
        return np.full(rr.shape, np.nan)

    # Central difference: (RR[i+1] - RR[i-1]) / 2
    central_diff = 0.5 * (rr[2:] - rr[:-2])
    central_diff = np.concatenate(([np.nan], central_diff, [np.nan]))

    # Rolling sum (full window required as per updated specification)
    hr_diff_sum = pd.Series(central_diff).rolling(
        window=window_size,
        min_periods=window_size
    ).sum().to_numpy()

    return hr_diff_sum


def compute_max_hr_change_from_rr(rr_ms: np.ndarray,
                                  window_rr: int = WINDOW_RR) -> Optional[float]:
    """
    Compute maximum heart rate change from RR intervals.

    Args:
        rr_ms: RR intervals in milliseconds
        window_rr: Window size for rolling calculation

    Returns:
        Maximum absolute HR change in BPM, or None if insufficient data
    """
    if rr_ms is None or rr_ms.size < window_rr:
        return None

    # Calculate HR difference sum
    hr_diff_sum = calculate_hr_diff_jeppesen(rr_ms, window_size=window_rr)

    # Calculate mean RR over rolling window
    mean_rr = pd.Series(rr_ms).rolling(
        window=window_rr,
        min_periods=window_rr
    ).mean().to_numpy()

    # Convert to BPM change: -60000 / (mean_RR^2) * HR_diff_sum
    with np.errstate(divide='ignore', invalid='ignore'):
        hr_diff_bpm = -60000.0 / (mean_rr ** 2) * hr_diff_sum

    # Take absolute values and remove NaN
    hr_abs = np.abs(hr_diff_bpm)
    hr_abs = hr_abs[~np.isnan(hr_abs)]

    if hr_abs.size == 0:
        return None

    return float(np.max(hr_abs))


# ==================== RECORDING DISCOVERY ====================

def discover_all_recordings(data_path: str) -> List[Tuple[str, str]]:
    """
    Discover all available recordings in the SeizeIT2 dataset.

    Args:
        data_path: Path to SeizeIT2 dataset root

    Returns:
        List of (subject_id, run_id) tuples
    """
    data_path = Path(data_path)
    recordings = []

    if not data_path.exists():
        print(f"Warning: Data path {data_path} does not exist")
        return recordings

    # Find all subjects
    subjects = sorted([x for x in data_path.glob("sub-*") if x.is_dir()])
    print(f"Found {len(subjects)} subjects")

    for subject_dir in subjects:
        subject_id = subject_dir.name

        # Look for ECG sessions
        ecg_dir = subject_dir / 'ses-01' / 'ecg'
        if ecg_dir.exists():
            # Find all runs for this subject
            edf_files = sorted(list(ecg_dir.glob("*_ecg.edf")))

            for edf_file in edf_files:
                # Extract run ID from filename
                parts = edf_file.stem.split('_')
                run_part = [p for p in parts if p.startswith('run-')]

                if run_part:
                    run_id = run_part[0]
                    recordings.append((subject_id, run_id))

    print(f"Found {len(recordings)} total recordings")
    return recordings


# ==================== PLOTTING FUNCTIONS ====================

def plot_seizure_with_rpeaks(subject_id: str, run_id: str, seizure_info: Dict,
                              patient_id: str, output_dir: str) -> bool:
    """
    Create a plot showing the first valid seizure with detected R-peaks.

    Args:
        subject_id: Subject identifier
        run_id: Run identifier
        seizure_info: Dictionary with seizure information (start_time, duration, max_hr_change, seizure_index)
        patient_id: Patient ID for filename
        output_dir: Directory to save plot

    Returns:
        True if plot was created successfully, False otherwise
    """
    try:
        # Load data
        global PRE
        if PRE is None:
            PRE = ECGPreprocessor()

        data_obj, annotations = PRE.load_data(DATA_PATH, subject_id, run_id)

        if not data_obj or not data_obj.data:
            return False

        # Select ECG channel
        for ch_data, ch_name, fs in zip(data_obj.data, data_obj.channels, data_obj.fs):
            if 'ecg' in ch_name.lower() or 'ekg' in ch_name.lower():
                raw_signal = np.asarray(ch_data, dtype=float)
                fs = float(fs)
                break
        else:
            raw_signal = np.asarray(data_obj.data[0], dtype=float)
            fs = float(data_obj.fs[0])

        # Preprocess
        preprocessed_signal = preprocess_for_elgendi(raw_signal, fs)

        # Detect R-peaks
        peaks = detect_r_peaks_elgendi(preprocessed_signal, fs)

        if peaks.size < 2:
            return False

        # Extract seizure timing
        start_s = seizure_info['start_time']
        end_s = start_s + seizure_info['duration']

        # Define plot window: 2 minutes before to 2 minutes after seizure
        plot_start_s = max(0, start_s - 120)
        plot_end_s = min(len(raw_signal) / fs, end_s + 120)

        plot_start_sample = int(plot_start_s * fs)
        plot_end_sample = int(plot_end_s * fs)

        # Extract segment for plotting
        plot_signal = preprocessed_signal[plot_start_sample:plot_end_sample]
        time_axis = np.arange(len(plot_signal)) / fs + plot_start_s

        # Find peaks in plot window
        peaks_in_window = peaks[(peaks >= plot_start_sample) & (peaks < plot_end_sample)]
        peaks_relative = peaks_in_window - plot_start_sample

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

        # ===== Plot 1: Full overview =====
        ax1.plot(time_axis, plot_signal, 'b-', linewidth=0.8, alpha=0.7, label='ECG Signal (preprocessed)')
        ax1.plot(time_axis[peaks_relative], plot_signal[peaks_relative],
                'ro', markersize=4, label=f'R-Peaks (n={len(peaks_relative)})', alpha=0.6)

        # Mark seizure period
        ax1.axvspan(start_s, end_s, alpha=0.3, color='red', label='Seizure Period')
        ax1.axvline(start_s, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax1.axvline(end_s, color='red', linestyle='--', linewidth=2, alpha=0.8)

        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('ECG Amplitude (normalized)', fontsize=12)
        seizure_idx = seizure_info.get('seizure_index', 1)
        ax1.set_title(f'First Valid Seizure (#{seizure_idx}) - {patient_id} ({subject_id}_{run_id})\n'
                     f'Max HR Change: {seizure_info["max_hr_change"]:.1f} bpm | '
                     f'Seizure Duration: {seizure_info["duration"]:.1f}s',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ===== Plot 2: Zoomed view of seizure ±30 seconds =====
        zoom_start_s = max(plot_start_s, start_s - 30)
        zoom_end_s = min(plot_end_s, end_s + 30)

        zoom_start_sample = int((zoom_start_s - plot_start_s) * fs)
        zoom_end_sample = int((zoom_end_s - plot_start_s) * fs)

        zoom_signal = plot_signal[zoom_start_sample:zoom_end_sample]
        zoom_time = time_axis[zoom_start_sample:zoom_end_sample]

        # Find peaks in zoom window
        peaks_in_zoom = peaks_relative[(peaks_relative >= zoom_start_sample) &
                                       (peaks_relative < zoom_end_sample)]
        peaks_in_zoom_adjusted = peaks_in_zoom - zoom_start_sample

        ax2.plot(zoom_time, zoom_signal, 'b-', linewidth=1.5, label='ECG Signal')
        if len(peaks_in_zoom_adjusted) > 0:
            ax2.plot(zoom_time[peaks_in_zoom_adjusted], zoom_signal[peaks_in_zoom_adjusted],
                    'ro', markersize=8, label=f'R-Peaks (n={len(peaks_in_zoom_adjusted)})')

        # Mark seizure boundaries
        ax2.axvline(start_s, color='red', linestyle='--', linewidth=2,
                   label='Seizure Start', alpha=0.8)
        ax2.axvline(end_s, color='red', linestyle='--', linewidth=2,
                   label='Seizure End', alpha=0.8)
        ax2.axvspan(start_s, end_s, alpha=0.2, color='red')

        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('ECG Amplitude (normalized)', fontsize=12)
        ax2.set_title('Zoomed View: Seizure Period (±30 seconds)', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        filename = f"responder_{patient_id}_{subject_id}_{run_id}_seizure{seizure_idx}.png"
        filepath = output_path / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"    Plot saved: {filepath}")
        return True

    except Exception as e:
        print(f"    Plot failed: {e}")
        traceback.print_exc()
        return False


# ==================== MAIN PROCESSING ====================

def process_recording(subject_id: str, run_id: str) -> Optional[Dict]:
    """
    Process a single recording to extract ALL seizures with timing info.

    Returns dict with:
    - subject_id
    - run_id
    - seizures: list of dicts with {start_time, duration, max_hr_change, seizure_index, is_valid}

    Args:
        subject_id: Subject identifier
        run_id: Run identifier

    Returns:
        Dictionary with seizure data, or None if processing failed
    """
    global PRE
    try:
        # Lazy-Init if not set via init_worker
        if PRE is None:
            PRE = ECGPreprocessor()

        # Load raw data
        data_obj, annotations = PRE.load_data(DATA_PATH, subject_id, run_id)

        if not data_obj or not data_obj.data:
            return None

        # Select ECG channel(s)
        selected_signals = []
        selected_fs = []

        for ch_data, ch_name, fs in zip(data_obj.data, data_obj.channels, data_obj.fs):
            if 'ecg' in ch_name.lower() or 'ekg' in ch_name.lower():
                selected_signals.append(np.asarray(ch_data, dtype=float))
                selected_fs.append(float(fs))

        # Fallback: use first channel if no ECG label found
        if not selected_signals:
            selected_signals.append(np.asarray(data_obj.data[0], dtype=float))
            selected_fs.append(float(data_obj.fs[0]))

        # Use first ECG channel
        raw_signal = selected_signals[0]
        fs = selected_fs[0]

        # Check signal quality
        is_valid, reason = check_signal_quality(raw_signal, fs)
        if not is_valid:
            return None

        # Minimal preprocessing for Elgendi method
        preprocessed_signal = preprocess_for_elgendi(raw_signal, fs)

        # Detect R-peaks using Elgendi method
        peaks = detect_r_peaks_elgendi(preprocessed_signal, fs)

        if peaks.size < 2:
            return None

        # Calculate RR intervals
        rr_ms = rr_intervals_ms_from_peaks(peaks, fs)

        if rr_ms.size < 10:
            return None

        # Extract seizure events
        seizure_events = annotations.events if hasattr(annotations, "events") else []
        events_secs = []

        for e in seizure_events:
            try:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    events_secs.append((float(e[0]), float(e[1])))
                elif isinstance(e, dict):
                    s = e.get("start", e.get("onset"))
                    en = e.get("end", e.get("offset", s + 10.0))
                    events_secs.append((float(s), float(en)))
            except Exception:
                continue

        if not events_secs:
            return None

        # Process ALL seizures and collect timing info
        # IMPORTANT: Track both valid and invalid seizures with their index
        seizure_results = []

        for seizure_idx, (start_s, end_s) in enumerate(events_secs, start=1):
            start_sample = int(start_s * fs)
            end_sample = int(end_s * fs)

            # Find peaks within seizure interval
            mask = (peaks >= start_sample) & (peaks <= end_sample)
            local_peaks = peaks[mask]

            # Track why seizure is invalid
            invalid_reason = None

            if local_peaks.size < 2:
                invalid_reason = "Too few R-peaks"
            else:
                # Calculate local RR intervals
                local_rr = rr_intervals_ms_from_peaks(local_peaks, fs)

                if local_rr.size < 3:
                    invalid_reason = "Too few RR intervals"
                else:
                    # Compute max HR change
                    max_change = compute_max_hr_change_from_rr(local_rr, WINDOW_RR)

                    if max_change is None:
                        invalid_reason = f"Insufficient data for window (need {WINDOW_RR} RR intervals, have {local_rr.size})"
                    else:
                        # Valid seizure!
                        seizure_results.append({
                            'start_time': start_s,
                            'duration': end_s - start_s,
                            'max_hr_change': max_change,
                            'seizure_index': seizure_idx,
                            'is_valid': True,
                            'invalid_reason': None
                        })
                        continue

            # Add invalid seizure to results for tracking
            seizure_results.append({
                'start_time': start_s,
                'duration': end_s - start_s,
                'max_hr_change': None,
                'seizure_index': seizure_idx,
                'is_valid': False,
                'invalid_reason': invalid_reason
            })

        if not seizure_results:
            return None

        return {
            'subject_id': subject_id,
            'run_id': run_id,
            'seizures': seizure_results
        }

    except Exception as e:
        return None


def main():
    """Main processing pipeline."""
    print("=" * 70)
    print("Responder Detection - FIRST VALID SEIZURE")
    print("Using Elgendi R-Peak Detection")
    print("=" * 70)
    print()
    print("Classification criterion:")
    print("  Uses the first VALID (analyzable) seizure per patient.")
    print("  If the first seizure is too short, automatically uses the next one.")
    print("  Responder: First valid seizure has HR change > 50 bpm")
    print()
    print(f"Parameters:")
    print(f"  - Bandpass: {ELGENDI_LOW}-{ELGENDI_HIGH} Hz")
    print(f"  - BPM threshold: {BPM_THRESHOLD}")
    print(f"  - RR window: {WINDOW_RR} intervals")
    print("=" * 70)
    print()

    # CLI arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel processes (Default: CPU count)")
    args, _ = parser.parse_known_args()
    workers = max(1, int(args.workers or 1))

    # Discover recordings
    print("Discovering recordings...")
    recordings = discover_all_recordings(DATA_PATH)

    if not recordings:
        print("No recordings found!")
        return

    print(f"\nProcessing {len(recordings)} recordings with {workers} workers...\n")

    # Parallel: process all recordings
    recording_results = []
    processed_count = 0
    failed_count = 0

    total = len(recordings)

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        future_to_rec = {
            executor.submit(process_recording, subject_id, run_id): (subject_id, run_id)
            for (subject_id, run_id) in recordings
        }

        for idx, future in enumerate(as_completed(future_to_rec), 1):
            subject_id, run_id = future_to_rec[future]
            try:
                result = future.result()
            except Exception as e:
                failed_count += 1
                print(f"[{idx}/{total}] {subject_id} {run_id} -> ERROR: {e}")
                continue

            if result is not None:
                recording_results.append(result)
                processed_count += 1
                valid_count = sum(1 for s in result['seizures'] if s['is_valid'])
                total_count = len(result['seizures'])
                print(f"[{idx}/{total}] {subject_id} {run_id} -> OK ({valid_count}/{total_count} valid seizures)")
            else:
                failed_count += 1
                print(f"[{idx}/{total}] {subject_id} {run_id} -> NO DATA")

    print(f"\n{'=' * 70}")
    print(f"Processing complete: {processed_count} successful, {failed_count} failed")
    print(f"{'=' * 70}\n")

    # Group by patient and find FIRST VALID seizure
    print("Identifying first VALID seizure per patient...")

    patient_first_valid_seizure = {}  # patient_id -> {recording, seizure_info}
    patient_stats = {}  # Track statistics

    for rec_result in recording_results:
        patient_id = Path(rec_result['subject_id']).name

        # Sort seizures by start time (chronologically)
        sorted_seizures = sorted(rec_result['seizures'], key=lambda x: x['start_time'])

        # Find first VALID seizure
        first_valid = None
        for seizure in sorted_seizures:
            if seizure['is_valid']:
                first_valid = seizure
                break

        if first_valid is None:
            # No valid seizures in this recording
            if patient_id not in patient_stats:
                patient_stats[patient_id] = {
                    'total_seizures': 0,
                    'valid_seizures': 0,
                    'invalid_seizures': 0,
                    'invalid_reasons': []
                }
            patient_stats[patient_id]['total_seizures'] += len(sorted_seizures)
            patient_stats[patient_id]['invalid_seizures'] += len(sorted_seizures)
            for s in sorted_seizures:
                if s['invalid_reason']:
                    patient_stats[patient_id]['invalid_reasons'].append(s['invalid_reason'])
            continue

        # Update patient statistics
        if patient_id not in patient_stats:
            patient_stats[patient_id] = {
                'total_seizures': 0,
                'valid_seizures': 0,
                'invalid_seizures': 0,
                'invalid_reasons': []
            }

        patient_stats[patient_id]['total_seizures'] += len(sorted_seizures)
        patient_stats[patient_id]['valid_seizures'] += sum(1 for s in sorted_seizures if s['is_valid'])
        patient_stats[patient_id]['invalid_seizures'] += sum(1 for s in sorted_seizures if not s['is_valid'])

        # Check if this is the earliest valid seizure for this patient across all recordings
        if patient_id not in patient_first_valid_seizure:
            patient_first_valid_seizure[patient_id] = {
                'recording': f"{rec_result['subject_id']}_{rec_result['run_id']}",
                'seizure': first_valid
            }
        else:
            # Compare with existing first valid seizure
            # Use alphabetically first recording as tiebreaker
            existing_rec = patient_first_valid_seizure[patient_id]['recording']
            new_rec = f"{rec_result['subject_id']}_{rec_result['run_id']}"

            if new_rec < existing_rec:
                patient_first_valid_seizure[patient_id] = {
                    'recording': new_rec,
                    'seizure': first_valid
                }

    # Classify patients based on first VALID seizure only
    responders = []
    non_responders = []
    no_valid_seizures = []

    for patient_id, stats in patient_stats.items():
        if patient_id in patient_first_valid_seizure:
            max_hr_change = patient_first_valid_seizure[patient_id]['seizure']['max_hr_change']

            if max_hr_change > BPM_THRESHOLD:
                responders.append(patient_id)
            else:
                non_responders.append(patient_id)
        else:
            # Patient has no valid seizures
            no_valid_seizures.append(patient_id)

    # Sort lists
    responders.sort()
    non_responders.sort()
    no_valid_seizures.sort()

    # Write results to text file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Responders and non-responders - FIRST VALID SEIZURE\n")
        f.write("# Using Elgendi R-peak detection\n")
        f.write("# Classification: First VALID seizure with HR change >50 bpm (100 RR intervals)\n")
        f.write("# If first seizure is too short, uses next valid seizure\n")
        f.write(f"# BPM threshold: {BPM_THRESHOLD}\n")
        f.write(f"# Window size: {WINDOW_RR} RR intervals\n\n")
        f.write("responders = " + repr(responders) + "\n\n")
        f.write("non_responders = " + repr(non_responders) + "\n\n")
        f.write("no_valid_seizures = " + repr(no_valid_seizures) + "\n")

    # Write detailed CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["subject_id", "recording", "seizure_index", "seizure_start_time",
                        "seizure_duration", "max_hr_change_bpm", "label",
                        "total_seizures", "valid_seizures", "invalid_seizures"])

        for pid in sorted(patient_stats.keys()):
            stats = patient_stats[pid]

            if pid in patient_first_valid_seizure:
                data = patient_first_valid_seizure[pid]
                seizure = data['seizure']

                max_hr = seizure['max_hr_change']
                label = "responder" if max_hr > BPM_THRESHOLD else "non_responder"

                writer.writerow([
                    pid,
                    data['recording'],
                    seizure['seizure_index'],
                    f"{seizure['start_time']:.2f}",
                    f"{seizure['duration']:.2f}",
                    f"{max_hr:.2f}",
                    label,
                    stats['total_seizures'],
                    stats['valid_seizures'],
                    stats['invalid_seizures']
                ])
            else:
                # No valid seizures
                writer.writerow([
                    pid,
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "no_valid_seizures",
                    stats['total_seizures'],
                    stats['valid_seizures'],
                    stats['invalid_seizures']
                ])

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - FIRST VALID SEIZURE")
    print("=" * 70)
    print(f"Responders:         {len(responders):3d}")
    print(f"Non-responders:     {len(non_responders):3d}")
    print(f"No valid seizures:  {len(no_valid_seizures):3d}")
    print(f"Total patients:     {len(responders) + len(non_responders) + len(no_valid_seizures):3d}")
    print("=" * 70)

    # Statistics on first valid seizure index
    if patient_first_valid_seizure:
        seizure_indices = [data['seizure']['seizure_index'] for data in patient_first_valid_seizure.values()]
        print(f"\nFirst valid seizure statistics:")
        print(f"  - Used seizure #1: {sum(1 for i in seizure_indices if i == 1):3d} patients")
        print(f"  - Used seizure #2: {sum(1 for i in seizure_indices if i == 2):3d} patients")
        print(f"  - Used seizure #3+: {sum(1 for i in seizure_indices if i >= 3):3d} patients")
        print(f"  - Mean seizure index: {np.mean(seizure_indices):.2f}")

    print(f"\nResults written to:")
    print(f"  - {OUTPUT_TXT}")
    print(f"  - {OUTPUT_CSV}")
    print()

    # ==================== PLOTTING SECTION ====================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS FOR RANDOM RESPONDERS")
    print("=" * 70)
    print()

    if len(responders) == 0:
        print("No responders found - skipping plotting.")
    else:
        # Select random responders to plot
        num_to_plot = min(NUM_PLOTS, len(responders))
        selected_responders = random.sample(responders, num_to_plot)

        print(f"Randomly selected {num_to_plot} responders to plot:")
        for pid in selected_responders:
            seizure_idx = patient_first_valid_seizure[pid]['seizure']['seizure_index']
            print(f"  - {pid} (seizure #{seizure_idx})")
        print()

        # Create plots
        plot_count = 0
        failed_count = 0

        for idx, patient_id in enumerate(selected_responders, 1):
            print(f"[{idx}/{num_to_plot}] Plotting {patient_id}...")

            # Get recording info
            data = patient_first_valid_seizure[patient_id]
            recording_parts = data['recording'].split('_')
            subject_id = '_'.join(recording_parts[:-1])  # Everything except last part
            run_id = recording_parts[-1]  # Last part is run_id

            # Create plot
            success = plot_seizure_with_rpeaks(
                subject_id=subject_id,
                run_id=run_id,
                seizure_info=data['seizure'],
                patient_id=patient_id,
                output_dir=PLOT_DIR
            )

            if success:
                plot_count += 1
            else:
                failed_count += 1

        print()
        print("=" * 70)
        print(f"Plotting complete: {plot_count} successful, {failed_count} failed")
        print(f"Plots saved to: {PLOT_DIR}/")
        print("=" * 70)
    print()


if __name__ == "__main__":
    main()
