#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Responder Detection with Robust R-Peak Detection
=========================================================

This script improves upon find_responder_jh.py by implementing:
1. Proper ECG preprocessing before R-peak detection
2. Robust R-peak detection algorithm (Pan-Tompkins inspired)
3. Signal quality assessment
4. Enhanced artifact handling

Uses raw data loading from preprocessing.py but applies targeted filtering
optimized for R-peak detection before computing heart rate changes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import traceback
import sys
from typing import List, Tuple, Optional
from scipy.signal import butter, filtfilt, find_peaks, medfilt, iirnotch
from scipy import signal as sp_signal
import warnings

# ==================== CONFIGURATION ====================

DATA_PATH = "/home/swolf/asim_shared/raw_data/ds005873-1.1.0"

# Output files
OUTPUT_TXT = "responders_and_non_responders_improved.txt"
OUTPUT_CSV = "patient_responder_summary_improved.csv"

# Analysis parameters
WINDOW_RR = 100          # Rolling window for Jeppesen HR-diff calculation
BPM_THRESHOLD = 50       # Threshold for responder classification
MIN_RR_SECONDS = 0.25    # Minimum RR interval (max ~240 bpm)
MAX_RR_SECONDS = 2.0     # Maximum RR interval (min ~30 bpm)

# Preprocessing parameters for R-peak detection
BANDPASS_LOW = 5.0       # Hz - optimized for QRS complex
BANDPASS_HIGH = 15.0     # Hz - optimized for QRS complex
FILTER_ORDER = 4         # Butterworth filter order
POWERLINE_FREQ = 50.0    # Hz - 50 Hz (Europe) or 60 Hz (USA)

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

def apply_bandpass_filter(signal: np.ndarray, fs: float,
                          low_freq: float = BANDPASS_LOW,
                          high_freq: float = BANDPASS_HIGH,
                          order: int = FILTER_ORDER) -> np.ndarray:
    """
    Apply Butterworth bandpass filter optimized for QRS complex detection.

    Args:
        signal: Input ECG signal
        fs: Sampling frequency
        low_freq: Low cutoff frequency (default: 5 Hz)
        high_freq: High cutoff frequency (default: 15 Hz)
        order: Filter order (default: 4)

    Returns:
        Filtered signal
    """
    nyquist = fs / 2.0
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist

    # Ensure valid frequency range
    low_norm = max(0.001, min(low_norm, 0.99))
    high_norm = max(low_norm + 0.01, min(high_norm, 0.99))

    b, a = butter(order, [low_norm, high_norm], btype='band')
    filtered = filtfilt(b, a, signal)

    return filtered


def remove_baseline_wander(signal: np.ndarray, fs: float,
                           cutoff: float = 0.5) -> np.ndarray:
    """
    Remove baseline wander using median filtering.

    Args:
        signal: Input signal
        fs: Sampling frequency
        cutoff: Cutoff frequency for baseline (default: 0.5 Hz)

    Returns:
        Signal with baseline removed
    """
    # Kernel size based on cutoff frequency
    kernel_size = int(0.2 * fs)  # 200ms window
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd

    baseline = medfilt(signal, kernel_size=kernel_size)
    return signal - baseline


def apply_notch_filter(signal: np.ndarray, fs: float,
                       freq: float = POWERLINE_FREQ,
                       quality: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference.

    Args:
        signal: Input signal
        fs: Sampling frequency
        freq: Frequency to remove (50 or 60 Hz)
        quality: Quality factor (higher = narrower notch)

    Returns:
        Filtered signal
    """
    if freq >= fs / 2:
        warnings.warn(f"Notch frequency {freq} >= Nyquist, skipping")
        return signal

    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal)


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


def preprocess_for_rpeak_detection(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Complete preprocessing pipeline optimized for R-peak detection.

    Pipeline:
    1. Bandpass filter (5-15 Hz) - QRS complex
    2. Remove baseline wander
    3. Notch filter (50/60 Hz) - powerline interference
    4. Normalize signal

    Args:
        signal: Raw ECG signal
        fs: Sampling frequency

    Returns:
        Preprocessed signal ready for R-peak detection
    """
    # 1. Bandpass filter
    signal_filtered = apply_bandpass_filter(signal, fs)

    # 2. Remove baseline wander
    signal_corrected = remove_baseline_wander(signal_filtered, fs)

    # 3. Notch filter for powerline interference
    signal_notched = apply_notch_filter(signal_corrected, fs)

    # 4. Normalize
    signal_normalized = normalize_signal(signal_notched)

    return signal_normalized


# ==================== R-PEAK DETECTION ====================

def detect_r_peaks_robust(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Robust R-peak detection using Pan-Tompkins inspired algorithm.

    Steps:
    1. Square signal to emphasize peaks
    2. Moving window integration
    3. Adaptive thresholding
    4. Peak detection with validation

    Args:
        signal: Preprocessed ECG signal
        fs: Sampling frequency

    Returns:
        Array of R-peak indices
    """
    if signal is None or len(signal) < 10:
        return np.array([], dtype=int)

    # 1. Derivative (emphasizes QRS slope)
    derivative = np.diff(signal)
    derivative = np.concatenate(([0], derivative))

    # 2. Square (emphasizes larger peaks)
    signal_squared = derivative ** 2

    # 3. Moving window integration (smooth)
    window_size = int(0.15 * fs)  # 150ms - typical QRS duration
    if window_size < 1:
        window_size = 1

    window = np.ones(window_size) / window_size
    signal_integrated = np.convolve(signal_squared, window, mode='same')

    # 4. Adaptive thresholding
    # Calculate threshold as fraction of signal statistics
    signal_mean = np.mean(signal_integrated)
    signal_std = np.std(signal_integrated)
    threshold = signal_mean + 0.3 * signal_std

    # 5. Peak detection
    min_distance = int(MIN_RR_SECONDS * fs)
    max_distance = int(MAX_RR_SECONDS * fs)

    peaks, properties = find_peaks(
        signal_integrated,
        height=threshold,
        distance=min_distance,
        prominence=0.1 * np.max(signal_integrated)
    )

    # 6. Validation: check if peaks are physiologically plausible
    if len(peaks) > 1:
        # Remove peaks with invalid RR intervals
        rr_intervals = np.diff(peaks) / fs
        valid_mask = np.concatenate(([True],
                                     (rr_intervals >= MIN_RR_SECONDS) &
                                     (rr_intervals <= MAX_RR_SECONDS)))
        peaks = peaks[valid_mask]

    return peaks


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

    # Rolling sum
    hr_diff_sum = pd.Series(central_diff).rolling(
        window=window_size,
        min_periods=max(1, window_size // 2)  # More robust for short seizures
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
    if rr_ms is None or rr_ms.size < 3:
        return None

    # Calculate HR difference sum
    hr_diff_sum = calculate_hr_diff_jeppesen(rr_ms, window_size=window_rr)

    # Calculate mean RR over rolling window
    mean_rr = pd.Series(rr_ms).rolling(
        window=window_rr,
        min_periods=max(1, window_rr // 2)
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
                # Format: sub-XXX_ses-01_task-szMonitoring_run-XX_ecg.edf
                parts = edf_file.stem.split('_')
                run_part = [p for p in parts if p.startswith('run-')]

                if run_part:
                    run_id = run_part[0]
                    recordings.append((subject_id, run_id))

    print(f"Found {len(recordings)} total recordings")
    return recordings


# ==================== MAIN PROCESSING ====================

def process_recording(subject_id: str, run_id: str) -> Optional[List[float]]:
    """
    Process a single recording to extract seizure-specific max HR changes.

    Args:
        subject_id: Subject identifier
        run_id: Run identifier

    Returns:
        List of max HR changes for each seizure, or None if processing failed
    """
    try:
        # Load raw data
        data_obj, annotations = pre.load_data(DATA_PATH, subject_id, run_id)

        if not data_obj or not data_obj.data:
            print(f"  No ECG data found")
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
            print(f"  Signal quality check failed: {reason}")
            return None

        # Preprocess signal
        preprocessed_signal = preprocess_for_rpeak_detection(raw_signal, fs)

        # Detect R-peaks
        peaks = detect_r_peaks_robust(preprocessed_signal, fs)

        if peaks.size < 2:
            print(f"  Insufficient R-peaks detected: {peaks.size}")
            return None

        # Calculate RR intervals
        rr_ms = rr_intervals_ms_from_peaks(peaks, fs)

        if rr_ms.size < 10:
            print(f"  Insufficient valid RR intervals: {rr_ms.size}")
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

        # Process each seizure
        seizure_maxima = []

        if events_secs:
            for start_s, end_s in events_secs:
                start_sample = int(start_s * fs)
                end_sample = int(end_s * fs)

                # Find peaks within seizure interval
                mask = (peaks >= start_sample) & (peaks <= end_sample)
                local_peaks = peaks[mask]

                if local_peaks.size < 2:
                    continue

                # Calculate local RR intervals
                local_rr = rr_intervals_ms_from_peaks(local_peaks, fs)

                if local_rr.size < 3:
                    continue

                # Compute max HR change
                max_change = compute_max_hr_change_from_rr(local_rr, WINDOW_RR)

                if max_change is not None:
                    seizure_maxima.append(max_change)
        else:
            # No seizures annotated - use entire recording
            max_change = compute_max_hr_change_from_rr(rr_ms, WINDOW_RR)
            if max_change is not None:
                seizure_maxima.append(max_change)

        if not seizure_maxima:
            print(f"  No valid seizure maxima computed")
            return None

        print(f"  Processed {len(seizure_maxima)} seizures, mean max HR change: {np.mean(seizure_maxima):.2f} bpm")

        return seizure_maxima

    except Exception as e:
        print(f"  Error: {e}")
        traceback.print_exc()
        return None


def main():
    """Main processing pipeline."""
    print("=" * 70)
    print("Improved Responder Detection with Robust R-Peak Detection")
    print("=" * 70)
    print()

    # Discover recordings
    print("Discovering recordings...")
    recordings = discover_all_recordings(DATA_PATH)

    if not recordings:
        print("No recordings found!")
        return

    print(f"\nProcessing {len(recordings)} recordings...\n")

    # Process all recordings
    patient_seizure_maxima = {}
    processed_count = 0
    failed_count = 0

    for subject_id, run_id in recordings:
        print(f"Processing {subject_id} {run_id}...")

        seizure_maxima = process_recording(subject_id, run_id)

        if seizure_maxima is not None:
            patient_id = Path(subject_id).name
            patient_seizure_maxima.setdefault(patient_id, []).extend(seizure_maxima)
            processed_count += 1
        else:
            failed_count += 1

    print(f"\n{'=' * 70}")
    print(f"Processing complete: {processed_count} successful, {failed_count} failed")
    print(f"{'=' * 70}\n")

    # Classify patients
    responders = []
    non_responders = []
    unknown = []

    for patient_id, maxima in patient_seizure_maxima.items():
        valid = [m for m in maxima if m is not None and not np.isnan(m)]

        if len(valid) == 0:
            unknown.append(patient_id)
            continue

        mean_max = float(np.mean(valid))

        if mean_max > BPM_THRESHOLD:
            responders.append(patient_id)
        else:
            non_responders.append(patient_id)

    # Sort lists
    responders.sort()
    non_responders.sort()
    unknown.sort()

    # Write results to text file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Responders and non-responders using improved R-peak detection\n")
        f.write("# Preprocessing: Bandpass 5-15 Hz, baseline removal, notch filter, normalization\n")
        f.write("# R-peak detection: Pan-Tompkins inspired algorithm with adaptive thresholding\n")
        f.write(f"# BPM threshold: {BPM_THRESHOLD}\n")
        f.write(f"# Window size: {WINDOW_RR} RR intervals\n\n")
        f.write("responders = " + repr(responders) + "\n\n")
        f.write("non_responders = " + repr(non_responders) + "\n\n")
        f.write("unknown = " + repr(unknown) + "\n")

    # Write detailed CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["subject_id", "n_seizures", "mean_max_bpm", "std_max_bpm",
                        "min_max_bpm", "max_max_bpm", "label"])

        for pid, maxima in sorted(patient_seizure_maxima.items()):
            valid = [m for m in maxima if m is not None and not np.isnan(m)]
            n = len(valid)

            if n > 0:
                mean_m = float(np.mean(valid))
                std_m = float(np.std(valid)) if n > 1 else 0.0
                min_m = float(np.min(valid))
                max_m = float(np.max(valid))
                label = "responder" if mean_m > BPM_THRESHOLD else "non_responder"
            else:
                mean_m = std_m = min_m = max_m = float("nan")
                label = "unknown"

            writer.writerow([pid, n, mean_m, std_m, min_m, max_m, label])

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Responders:     {len(responders):3d}")
    print(f"Non-responders: {len(non_responders):3d}")
    print(f"Unknown:        {len(unknown):3d}")
    print(f"Total patients: {len(responders) + len(non_responders) + len(unknown):3d}")
    print("=" * 70)
    print(f"\nResults written to:")
    print(f"  - {OUTPUT_TXT}")
    print(f"  - {OUTPUT_CSV}")
    print()


if __name__ == "__main__":
    main()
