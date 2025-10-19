#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lädt Rohdaten mit der gleichen load_data-Logik wie in preprocessing.py
(ECGPreprocessor.load_data) und berechnet Responder direkt auf den Roh-EKG-Kanälen.

Wichtig: nutzt nur load_data aus preprocessing.py — keine preprocess_pipeline / kein Downsampling.
Referenz: preprocessing.py (lade-Logik & Data/Annotation Klassen). :contentReference[oaicite:1]{index=1}
"""

from pathlib import Path
import numpy as np
import pandas as pd
import traceback
from typing import List, Tuple, Optional
from pathlib import Path

# optional: scipy für robustere Peak-Erkennung
try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

# Pfade / Parameter — anpassen
DATA_PATH = "/home/swolf/asim_shared/raw_data/ds005873-1.1.0" 

RECORDINGS_LIST = None  # Optional: Liste von (subject_id, run_id). Wenn None: versucht, Recordings zu entdecken (siehe unten).
OUTPUT_TXT = "responders_and_non_responders_raw_via_preproc_load.txt"

# Analyse-Parameter
WINDOW_RR = 100
BPM_THRESHOLD = 50
MIN_RR_SECONDS = 0.25  # minimaler Abstand zwischen Peaks bei Erkennung (heuristisch)

from pathlib import Path
import sys
# füge project_root/Information/Data/seizeit2-main ganz vorn in sys.path ein
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Information" / "Data" / "seizeit2-main"))

# importiere die preprocessing.py (die du hochgeladen hast) und verwende ECGPreprocessor.load_data
import importlib.util
spec = importlib.util.spec_from_file_location("preprocessing_module", "preprocessing.py")
pre_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre_mod)

# Holen der Klasse
ECGPreprocessor = getattr(pre_mod, "ECGPreprocessor", None)
if ECGPreprocessor is None:
    raise RuntimeError("ECGPreprocessor nicht in preprocessing.py gefunden. Prüfe die Datei.")

# Wir instanziieren nur, um die load_data-Methode zu verwenden (keine pipeline)
pre = ECGPreprocessor()  # Standardkonstruktor — es werden nur Methoden genutzt, keine Preprocessing-Parameter

# Versuch, Recordings automatisch zu ermitteln, falls RECORDINGS_LIST None ist.
# Wir imitieren minimal die Struktur: suche sub-* Ordner und verwende runs aus Data.loadData intern.
def discover_recordings_from_filesystem(data_root: str) -> List[Tuple[str, str]]:
    root = Path(data_root)
    recs = []
    if not root.exists():
        return recs
    for sub in sorted(root.glob("sub-*")):
        # mögliche sess / runs werden später von Data.loadData intern behandelt, aber wir versuchen einfache Heuristik:
        # suche nach Unterordnern, die run-* beinhalten oder nach Dateien mit sub_run im Namen
        for run_dir in sub.rglob("*"):
            name = run_dir.name.lower()
            if name.startswith("run-"):
                recs.append((sub.name, name))
        # als Fallback: füge default run-01 hinzu (falls dataset anders strukturiert ist)
        if not any(r[0] == sub.name for r in recs):
            recs.append((sub.name, "run-01"))
    return recs

def discover_all_recordings(data_path):
    """
    Discover all available recordings in the SeizeIT2 dataset.
    
    Args:
        data_path: Path to SeizeIT2 dataset
        
    Returns:
        List of (subject_id, run_id) tuples
    """
    data_path = Path(data_path)
    recordings = []
    
    # Find all subjects
    subjects = [x for x in data_path.glob("sub-*") if x.is_dir()]
    print(f"Found {len(subjects)} subjects")
    
    for subject_dir in subjects:
        subject_id = subject_dir.name
        
        # Look for ECG sessions
        ecg_dir = subject_dir / 'ses-01' / 'ecg'
        if ecg_dir.exists():
            # Find all runs for this subject
            edf_files = list(ecg_dir.glob("*_ecg.edf"))
            
            for edf_file in edf_files:
                # Extract run ID from filename
                # Format: sub-XXX_ses-01_task-szMonitoring_run-XX_ecg.edf
                parts = edf_file.stem.split('_')
                run_part = [p for p in parts if p.startswith('run-')]
                
                if run_part:
                    run_id = run_part[0]
                    recordings.append((subject_id, run_id))
                    # print(f"  Found: {subject_id} {run_id}")
    return recordings
    

if RECORDINGS_LIST is None:
    # recordings = discover_recordings_from_filesystem(DATA_PATH)
    recordings = discover_all_recordings(DATA_PATH)


    
else:
    recordings = RECORDINGS_LIST

# Hilfsfunktionen: Peak detection, RR, Jeppesen HR-diff
def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Einfache Peak-Detection: verwendet scipy.find_peaks wenn verfügbar, sonst local maxima fallback.
    Arbeitet direkt auf Rohsignal (keine Filterung).
    """
    if signal is None or len(signal) < 10:
        return np.array([], dtype=int)
    min_dist = max(1, int(MIN_RR_SECONDS * fs))
    if find_peaks is not None:
        # heuristische Prominenz
        prom = max(1e-6, 0.5 * np.percentile(np.abs(signal - np.median(signal)), 75))
        peaks, props = find_peaks(signal, distance=min_dist, prominence=prom)
        return np.asarray(peaks, dtype=int)
    # numpy fallback
    peaks = []
    i = 1
    L = len(signal)
    while i < L - 1:
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
            i += min_dist
        else:
            i += 1
    return np.array(peaks, dtype=int)

def rr_intervals_ms_from_peaks(peaks: np.ndarray, fs: float) -> np.ndarray:
    if peaks.size < 2:
        return np.array([], dtype=float)
    diffs = np.diff(peaks).astype(float)
    rr_ms = diffs / float(fs) * 1000.0
    return rr_ms

def calculate_hr_diff_jeppesen(rr_intervals_ms: np.ndarray, window_size: int = WINDOW_RR) -> np.ndarray:
    rr = np.asarray(rr_intervals_ms, dtype=float)
    if rr.size < 3:
        return np.full(rr.shape, np.nan)
    central_diff = 0.5 * (rr[2:] - rr[:-2])
    central_diff = np.concatenate(([np.nan], central_diff, [np.nan]))
    # rolling sum (min_periods=1 damit auch kurze Iktusse berücksichtigt werden)
    hr_diff_sum = pd.Series(central_diff).rolling(window=window_size, min_periods=1).sum().to_numpy()
    return hr_diff_sum

def compute_max_hr_change_from_rr(rr_ms: np.ndarray, window_rr: int = WINDOW_RR) -> Optional[float]:
    if rr_ms is None or rr_ms.size == 0:
        return None
    hr_diff_sum = calculate_hr_diff_jeppesen(rr_ms, window_size=window_rr)
    mean_rr = pd.Series(rr_ms).rolling(window=window_rr, min_periods=1).mean().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        hr_diff_bpm = -60000.0 / (mean_rr ** 2) * hr_diff_sum
    hr_abs = np.abs(hr_diff_bpm)
    hr_abs = hr_abs[~np.isnan(hr_abs)]
    if hr_abs.size == 0:
        return None
    return float(np.nanmax(hr_abs))

# Hauptschleife: lade Rohdaten via pre.load_data und berechne Per-Seizure-Maxima
patient_seizure_maxima = {}  # subject_id -> list of maxima

for subj, run in recordings:
    try:
        print(f"Processing {subj} {run} ...")
        # load_data gibt (Data, Annotation) zurück wie in preprocessing.py. :contentReference[oaicite:2]{index=2}
        data_obj, annotations = pre.load_data(DATA_PATH, subj, run)

        if not data_obj or not data_obj.data:
            print(f"  Kein Roh-EKG für {subj}_{run} gefunden -> skip")
            continue

        # iterate over channels, wähle ECG-Kanal(e)
        selected_signals = []
        selected_fs = []
        selected_names = []
        for ch_data, ch_name, fs in zip(data_obj.data, data_obj.channels, data_obj.fs):
            if 'ecg' in ch_name.lower() or 'ekg' in ch_name.lower():
                selected_signals.append(np.asarray(ch_data, dtype=float))
                selected_fs.append(float(fs))
                selected_names.append(ch_name)
        # fallback: falls keine ECG-kennzeichnung, nehme ersten Kanal
        if not selected_signals:
            selected_signals.append(np.asarray(data_obj.data[0], dtype=float))
            selected_fs.append(float(data_obj.fs[0]))
            selected_names.append(data_obj.channels[0] if data_obj.channels else "ch0")

        # Annotations: annotations.events ist erwartete Struktur (Liste von (start,end) in Sekunden) gem. preprocessing.py
        seizure_events = annotations.events if hasattr(annotations, "events") and annotations.events else []
        # normalize events to list of (start_s, end_s)
        events_secs = []
        for e in seizure_events:
            try:
                # e kann Tuple, Liste oder Objekt sein; accept (start,end) float
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    events_secs.append((float(e[0]), float(e[1])))
                elif isinstance(e, dict) and ("start" in e or "onset" in e):
                    s = e.get("start", e.get("onset"))
                    en = e.get("end", e.get("offset", s + 10.0))
                    events_secs.append((float(s), float(en)))
            except Exception:
                continue

        # benutze erste gefundene ECG-Signal (du kannst hier erweitern falls mehrere Kanäle zusammengenommen werden sollen)
        sig = selected_signals[0]
        fs = selected_fs[0]

        # DETEKT: R-peaks auf ROH-SIGNAL (keine Filterung / kein Downsampling)
        peaks = detect_r_peaks(sig, fs)
        if peaks.size < 2:
            print(f"  Keine Peaks für {subj}_{run} gefunden -> markiere unknown")
            continue
        rr_ms_all = rr_intervals_ms_from_peaks(peaks, fs)

        seizure_maxima = []
        if events_secs:
            for (start_s, end_s) in events_secs:
                start_sample = int(start_s * fs)
                end_sample = int(end_s * fs)
                # finde peaks innerhalb des Intervalls
                mask = (peaks >= start_sample) & (peaks <= end_sample)
                local_peaks = peaks[mask]
                if local_peaks.size < 2:
                    continue
                local_rr = rr_intervals_ms_from_peaks(local_peaks, fs)
                mm = compute_max_hr_change_from_rr(local_rr, WINDOW_RR)
                if mm is not None:
                    seizure_maxima.append(mm)
        else:
            # fallback: ganze Aufnahme als Segment
            mm = compute_max_hr_change_from_rr(rr_ms_all, WINDOW_RR)
            if mm is not None:
                seizure_maxima.append(mm)

        if not seizure_maxima:
            print(f"  Keine verwertbaren Seizures/Maxima für {subj}_{run} gefunden -> skip")
            continue

        # patient_seizure_maxima.setdefault(subj, []).extend(seizure_maxima)
        patient_id = Path(subj).name  # stellt sicher: nur 'sub-###' als Key
        patient_seizure_maxima.setdefault(patient_id, []).extend(seizure_maxima)

        print(f"  {len(seizure_maxima)} Seizure-Maxima gefunden, mean={np.nanmean(seizure_maxima):.2f} bpm")

    except Exception as e:
        print(f"  Fehler bei {subj}_{run}: {e}")
        traceback.print_exc()
        continue

# Entscheidung pro Patient: Mittelwert aller Seizure-Maxima
# ------------------------------
# Patient-level Aggregation & Entscheidung (über alle Recordings)
# ------------------------------
responders = []
non_responders = []
unknown = []

for patient_id, maxima in patient_seizure_maxima.items():
    # maxima enthält alle per-seizure maxima aus allen runs dieses Patienten
    valid = [m for m in maxima if m is not None and not np.isnan(m)]
    if len(valid) == 0:
        unknown.append(patient_id)
        continue

    mean_max = float(np.nanmean(valid))  # Mittelwert über alle Iktus-Maxima des Patienten
    if mean_max > BPM_THRESHOLD:
        responders.append(patient_id)
    else:
        non_responders.append(patient_id)

# Sortiere Listen für Übersicht
responders.sort()
non_responders.sort()
unknown.sort()

# Schreibe Ergebnisdatei (Python-Listen)
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("# responders and non_responders determined from RAW data using mean across ALL recordings\n")
    f.write("responders = " + repr(responders) + "\n")
    f.write("non_responders = " + repr(non_responders) + "\n")
    f.write("unknown = " + repr(unknown) + "\n")

# optional: CSV mit n_seizures und mean pro Patient
import csv
CSV_OUT = "patient_responder_summary.csv"
with open(CSV_OUT, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["subject_id", "n_seizures", "mean_max_bpm", "label"])
    for pid, maxima in patient_seizure_maxima.items():
        valid = [m for m in maxima if m is not None and not np.isnan(m)]
        n = len(valid)
        meanm = float(np.nanmean(valid)) if n > 0 else float("nan")
        label = "responder" if (n>0 and meanm > BPM_THRESHOLD) else ("non_responder" if n>0 else "unknown")
        writer.writerow([pid, n, meanm, label])
print(f"CSV summary written to {CSV_OUT}")

print("\nFertig.")
print(f"Patient-level: Responders: {len(responders)}, Non-responders: {len(non_responders)}, Unknown: {len(unknown)}")
print(f"Output -> {OUTPUT_TXT}")


# Schreibe Ergebnisdatei
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("# responders and non_responders determined from RAW data using load_data from preprocessing.py\n")
    f.write("responders = " + repr(responders) + "\n")
    f.write("non_responders = " + repr(non_responders) + "\n")
    f.write("unknown = " + repr(unknown) + "\n")

print("\nFertig.")
print(f"Responders: {len(responders)}, Non-responders: {len(non_responders)}, Unknown: {len(unknown)}")
print(f"Output -> {OUTPUT_TXT}")