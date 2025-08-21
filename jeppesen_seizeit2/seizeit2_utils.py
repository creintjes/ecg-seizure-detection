"""
SeizeIT2-spezifische Utility-Funktionen für Jeppesen-Ansatz
"""

import os
import numpy as np
import pandas as pd
import pyedflib
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from config import SEIZEIT2_DATA_PATH, SAMPLING_RATE, MODALITIES

def get_all_subjects() -> List[str]:
    """
    Holt alle verfügbaren Subjects aus dem SeizeIT2 Dataset.
    
    Returns:
        List[str]: Liste der Subject-IDs (z.B. ['sub-001', 'sub-002', ...])
    """
    subjects = [d.name for d in SEIZEIT2_DATA_PATH.glob("sub-*") if d.is_dir()]
    return sorted(subjects)

def get_subject_recordings(subject: str) -> List[str]:
    """
    Holt alle Recordings für einen bestimmten Subject.
    
    Args:
        subject (str): Subject-ID (z.B. 'sub-001')
        
    Returns:
        List[str]: Liste der Run-IDs (z.B. ['run-01', 'run-02', ...])
    """
    subject_path = SEIZEIT2_DATA_PATH / subject / 'ses-01' / 'ecg'
    if not subject_path.exists():
        return []
    
    recordings = []
    for edf_file in subject_path.glob("*.edf"):
        # Extrahiere run-XX aus Dateinamen wie "sub-001_ses-01_task-szMonitoring_run-01_eeg.edf"
        parts = edf_file.stem.split('_')
        for part in parts:
            if part.startswith('run-'):
                recordings.append(part)
                break
    
    return sorted(list(set(recordings)))

def load_ecg_data(subject: str, recording: str) -> Tuple[np.ndarray, List[str], int]:
    """
    Lädt ECG-Daten aus SeizeIT2 EDF-Datei.
    
    Args:
        subject (str): Subject-ID (z.B. 'sub-001')
        recording (str): Recording-ID (z.B. 'run-01')
        
    Returns:
        Tuple[np.ndarray, List[str], int]: (ECG-Daten, Channel-Namen, Sampling-Rate)
    """
    ecg_file = SEIZEIT2_DATA_PATH / subject / 'ses-01' / 'ecg' / f"{subject}_ses-01_task-szMonitoring_{recording}_ecg.edf"
    
    if not ecg_file.exists():
        raise FileNotFoundError(f"ECG-Datei nicht gefunden: {ecg_file}")
    
    with pyedflib.EdfReader(str(ecg_file)) as edf:
        sampling_rates = edf.getSampleFrequencies()
        channel_names = edf.getSignalLabels()
        n_channels = edf.signals_in_file
        
        # Lade alle ECG-Kanäle
        ecg_data = []
        for i in range(n_channels):
            signal = edf.readSignal(i)
            ecg_data.append(signal)
    
    return np.array(ecg_data), channel_names, sampling_rates[0] if sampling_rates else SAMPLING_RATE

def load_annotations(subject: str, recording: str) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Lädt Seizure-Annotations aus SeizeIT2 TSV-Datei.
    
    Args:
        subject (str): Subject-ID (z.B. 'sub-001')
        recording (str): Recording-ID (z.B. 'run-01')
        
    Returns:
        Tuple[List[Tuple[float, float]], List[str]]: (Seizure-Zeitfenster, Event-Typen)
    """
    events_file = SEIZEIT2_DATA_PATH / subject / 'ses-01' / 'eeg' / f"{subject}_ses-01_task-szMonitoring_{recording}_events.tsv"
    
    if not events_file.exists():
        warnings.warn(f"Events-Datei nicht gefunden: {events_file}")
        return [], []
    
    df = pd.read_csv(events_file, delimiter="\t")
    
    seizure_events = []
    event_types = []
    
    for _, event in df.iterrows():
        # Filtere nur Seizure-Events (nicht 'bckg' oder 'impd')
        if event['eventType'] not in ['bckg', 'impd']:
            start_time = event['onset']
            end_time = event['onset'] + event['duration']
            seizure_events.append((start_time, end_time))
            event_types.append(event['eventType'])
    
    return seizure_events, event_types

def create_ecg_dataframe(subject: str, recording: str, primary_channel_idx: int = 0) -> pd.DataFrame:
    """
    Erstellt ein Pandas DataFrame mit ECG-Daten und Zeitindex.
    
    Args:
        subject (str): Subject-ID
        recording (str): Recording-ID  
        primary_channel_idx (int): Index des primären ECG-Kanals
        
    Returns:
        pd.DataFrame: DataFrame mit 'ecg' Column und DateTime-Index
    """
    ecg_data, channel_names, fs = load_ecg_data(subject, recording)
    
    # Wähle primären ECG-Kanal
    if primary_channel_idx >= len(ecg_data):
        primary_channel_idx = 0
        warnings.warn(f"Channel-Index {primary_channel_idx} nicht verfügbar, verwende Kanal 0")
    
    primary_ecg = ecg_data[primary_channel_idx]
    
    # Erstelle Zeitindex
    duration_seconds = len(primary_ecg) / fs
    time_index = pd.date_range(
        start='2000-01-01 00:00:00',  # Dummy-Startzeit
        periods=len(primary_ecg),
        freq=f'{1000/fs:.3f}ms'  # Frequenz in Millisekunden
    )
    
    # Erstelle DataFrame
    ecg_df = pd.DataFrame({
        'ecg': primary_ecg
    }, index=time_index)
    
    # Füge Metadaten als Attribute hinzu
    ecg_df.attrs['subject'] = subject
    ecg_df.attrs['recording'] = recording
    ecg_df.attrs['channel_names'] = channel_names
    ecg_df.attrs['sampling_rate'] = fs
    ecg_df.attrs['primary_channel'] = channel_names[primary_channel_idx] if primary_channel_idx < len(channel_names) else f"Channel_{primary_channel_idx}"
    
    return ecg_df

def add_seizure_labels(ecg_df: pd.DataFrame, subject: str, recording: str) -> pd.DataFrame:
    """
    Fügt Seizure-Labels zum ECG DataFrame hinzu.
    
    Args:
        ecg_df (pd.DataFrame): ECG DataFrame mit Zeitindex
        subject (str): Subject-ID
        recording (str): Recording-ID
        
    Returns:
        pd.DataFrame: DataFrame mit zusätzlichen 'seizure' und 'seizure_unfiltered' Spalten
    """
    result_df = ecg_df.copy()
    
    # Initialisiere Seizure-Spalten
    result_df['seizure'] = 0
    result_df['seizure_unfiltered'] = 0
    
    # Lade Annotations
    seizure_events, event_types = load_annotations(subject, recording)
    
    if not seizure_events:
        return result_df
    
    # Konvertiere Sekunden zu Zeitpunkten im DataFrame-Index
    start_time = result_df.index[0]
    sampling_rate = ecg_df.attrs.get('sampling_rate', SAMPLING_RATE)
    
    for (onset_sec, offset_sec), event_type in zip(seizure_events, event_types):
        # Berechne tatsächliche Zeitpunkte
        seizure_start = start_time + pd.Timedelta(seconds=onset_sec)
        seizure_end = start_time + pd.Timedelta(seconds=offset_sec)
        
        # Markiere Seizure-Zeitfenster
        seizure_mask = (result_df.index >= seizure_start) & (result_df.index <= seizure_end)
        
        # Für diese Implementierung behandeln wir alle Seizure-Types gleich
        result_df.loc[seizure_mask, 'seizure'] = 1
        result_df.loc[seizure_mask, 'seizure_unfiltered'] = 1
    
    return result_df

def get_all_patient_records_as_dict(subject: str) -> Dict[str, pd.DataFrame]:
    """
    Holt alle Recordings für einen Subject als Dictionary.
    Adaptiert für SeizeIT2-Format.
    
    Args:
        subject (str): Subject-ID (z.B. 'sub-001')
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mit Recording-IDs als Keys und ECG DataFrames als Values
    """
    recordings = get_subject_recordings(subject)
    records = {}
    
    for recording in recordings:
        try:
            # Lade ECG-Daten
            ecg_df = create_ecg_dataframe(subject, recording)
            
            # Füge Seizure-Labels hinzu
            ecg_df = add_seizure_labels(ecg_df, subject, recording)
            
            # Verwende konsistente Record-ID
            record_id = f"{subject}_{recording}"
            records[record_id] = ecg_df
            
        except Exception as e:
            warnings.warn(f"Fehler beim Laden von {subject} {recording}: {e}")
            continue
    
    return records

def get_seizure_count(subject: str, recording: str) -> int:
    """
    Zählt die Anzahl der Seizures in einem Recording.
    
    Args:
        subject (str): Subject-ID
        recording (str): Recording-ID
        
    Returns:
        int: Anzahl der Seizures
    """
    seizure_events, _ = load_annotations(subject, recording)
    return len(seizure_events)

def validate_subject_data(subject: str) -> bool:
    """
    Validiert, ob ein Subject gültige Daten hat.
    
    Args:
        subject (str): Subject-ID
        
    Returns:
        bool: True wenn Subject gültige ECG- und Event-Daten hat
    """
    try:
        recordings = get_subject_recordings(subject)
        if not recordings:
            print(f"Debug: Keine Recordings für {subject}")
            return False
        
        # Prüfe mindestens ein Recording
        first_recording = recordings[0]
        print(f"Debug: Teste Recording {first_recording}")
        ecg_data, channel_names, fs = load_ecg_data(subject, first_recording)
        
        has_data = len(ecg_data) > 0 and len(ecg_data[0]) > 1000
        print(f"Debug: ECG-Daten: {len(ecg_data)} Kanäle, {len(ecg_data[0]) if len(ecg_data) > 0 else 0} Samples")
        return has_data
        
    except Exception as e:
        print(f"Debug: Validierungsfehler für {subject}: {e}")
        return False