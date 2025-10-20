#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Responder Detection with Elgendi R-Peak Detection Method
========================================================

This script improves upon find_responder_jh.py by implementing:
1. Proper ECG preprocessing optimized for Elgendi method
2. Elgendi R-peak detection algorithm (robust and validated)
3. Signal quality assessment
4. Enhanced artifact handling

Uses the Elgendi method provided by the supervisor for R-peak detection.

Reference:
Elgendi, M. (2013). Fast QRS Detection with an Optimized Knowledge-Based Method.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal
import traceback
import sys
from typing import List, Tuple, Optional
from scipy.signal import butter, filtfilt, medfilt
import warnings
import os
import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==================== CONFIGURATION ====================

DATA_PATH = "/home/swolf/asim_shared/raw_data/ds005873-1.1.0"

# Output files
OUTPUT_TXT = "responders_and_non_responders_elgendi.txt"
OUTPUT_CSV = "patient_responder_summary_elgendi.csv"

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

PRE = None  # wird pro Prozess gesetzt

def init_worker():
    """
    Initializer für jeden Worker-Prozess:
    - begrenzt BLAS-Threads (verhindert Oversubscription)
    - initialisiert den ECGPreprocessor (pro Prozess)
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
        # Falls Import/Init fehlschlägt, Worker bleibt nutzbar; process_recording macht fallback
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

    # Rolling sum
    #   old: min_periods=max(1, window_size // 2)
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
    ...
    """
    global PRE
    try:
        # Lazy-Init falls nicht über init_worker gesetzt (z.B. Single-Process-Run)
        if PRE is None:
            PRE = ECGPreprocessor()

        # Load raw data
        data_obj, annotations = PRE.load_data(DATA_PATH, subject_id, run_id)

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

        # Minimal preprocessing for Elgendi method
        preprocessed_signal = preprocess_for_elgendi(raw_signal, fs)

        # Detect R-peaks using Elgendi method
        peaks = detect_r_peaks_elgendi(preprocessed_signal, fs)

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
    print("Responder Detection with Elgendi R-Peak Detection")
    print("=" * 70)
    print()
    print("Method: Elgendi, M. (2013). Fast QRS Detection with an")
    print("        Optimized Knowledge-Based Method.")
    print()
    print(f"Parameters:")
    print(f"  - Bandpass: {ELGENDI_LOW}-{ELGENDI_HIGH} Hz")
    print(f"  - Filter order: {ELGENDI_ORDER}")
    print(f"  - W1 factor: {ELGENDI_W1_FACTOR}")
    print(f"  - W2 factor: {ELGENDI_W2_FACTOR}")
    print(f"  - Beta: {ELGENDI_BETA}")
    print(f"  - BPM threshold: {BPM_THRESHOLD}")
    print("=" * 70)
    print()

    # --- CLI-Argumente ---
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Anzahl paralleler Prozesse (Default: Anzahl CPU-Kerne)")
    args, _ = parser.parse_known_args()
    workers = max(1, int(args.workers or 1))

    # Discover recordings
    print("Discovering recordings...")
    recordings = discover_all_recordings(DATA_PATH)

    if not recordings:
        print("No recordings found!")
        return

    print(f"\nProcessing {len(recordings)} recordings with {workers} workers...\n")

    # --- Parallel: alle Recordings verarbeiten ---
    patient_seizure_maxima = {}
    processed_count = 0
    failed_count = 0

    # Mapping für Fortschritt
    total = len(recordings)
    submitted = 0
    done = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        future_to_rec = {
            executor.submit(process_recording, subject_id, run_id): (subject_id, run_id)
            for (subject_id, run_id) in recordings
        }
        submitted = len(future_to_rec)

        for future in as_completed(future_to_rec):
            subject_id, run_id = future_to_rec[future]
            done += 1
            try:
                seizure_maxima = future.result()
            except Exception as e:
                failed_count += 1
                print(f"[{done}/{total}] {subject_id} {run_id} -> ERROR: {e}")
                continue

            if seizure_maxima is not None:
                patient_id = Path(subject_id).name
                patient_seizure_maxima.setdefault(patient_id, []).extend(seizure_maxima)
                processed_count += 1
                print(f"[{done}/{total}] {subject_id} {run_id} -> OK ({len(seizure_maxima)} seizures)")
            else:
                failed_count += 1
                print(f"[{done}/{total}] {subject_id} {run_id} -> NO DATA")

    print(f"\n{'=' * 70}")
    print(f"Processing complete: {processed_count} successful, {failed_count} failed")
    print(f"{'=' * 70}\n")

    # --- (ab hier bleibt dein Code wie gehabt) ---
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

    responders.sort()
    non_responders.sort()
    unknown.sort()

    # Write results to text file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Responders and non-responders using Elgendi R-peak detection\n")
        f.write("# Method: Elgendi, M. (2013). Fast QRS Detection with an Optimized Knowledge-Based Method.\n")
        f.write(f"# Preprocessing: Baseline removal + normalization (Elgendi includes 8-20 Hz bandpass)\n")
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
