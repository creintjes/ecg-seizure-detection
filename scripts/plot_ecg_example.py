import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add Information/Data/seizeit2-main to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Information', 'Data', 'seizeit2-main'))

from classes.data import Data
from classes.annotation import Annotation

def plot_patient_ecg(data_path, patient_id='sub-001', duration_minutes=5):
    """
    Lädt und plottet ECG-Daten eines Patienten
    
    Args:
        data_path (str): Pfad zum ds005873-download Ordner
        patient_id (str): Patient ID (z.B. 'sub-001')
        duration_minutes (int): Anzahl Minuten zum Plotten
    """
    
    data_path = Path(data_path)
    
    # Überprüfe ob Pfad existiert
    if not data_path.exists():
        print(f"Fehler: Pfad {data_path} existiert nicht!")
        return
    
    # Finde verfügbare Aufnahmen für den Patienten
    patient_path = data_path / patient_id / 'ses-01' / 'ecg'
    if not patient_path.exists():
        print(f"Fehler: ECG-Daten für {patient_id} nicht gefunden!")
        return
    
    # Lade erste verfügbare ECG-Aufnahme
    ecg_files = list(patient_path.glob("*.edf"))
    if not ecg_files:
        print(f"Keine ECG-Dateien für {patient_id} gefunden!")
        return
    
    # Extrahiere run-info aus Dateiname
    first_file = ecg_files[0]
    run_info = first_file.name.split('_')[-2]  # z.B. 'run-01'
    
    print(f"Lade Daten für {patient_id}, {run_info}")
    
    # Lade ECG-Daten
    recording = [patient_id, run_info]
    data = Data.loadData(data_path.as_posix(), recording, modalities=['ecg'])
    
    if not data.data:
        print("Keine ECG-Daten geladen!")
        return
    
    # Lade Annotationen (optional)
    try:
        annotations = Annotation.loadAnnotation(data_path.as_posix(), recording)
    except:
        print("Warnung: Annotationen konnten nicht geladen werden")
        annotations = None
    
    # Plot ECG-Daten
    ecg_signal = data.data[0]  # Erste ECG-Kanal
    fs = data.fs[0]  # Sampling frequency
    channel_name = data.channels[0] if data.channels else 'ECG'
    
    # Zeitachse erstellen
    total_samples = len(ecg_signal)
    time_total = total_samples / fs
    time = np.linspace(0, time_total, total_samples)
    
    # Begrenzen auf gewünschte Dauer
    max_samples = int(duration_minutes * 60 * fs)
    if max_samples < total_samples:
        ecg_signal = ecg_signal[:max_samples]
        time = time[:max_samples]
    
    # Plot erstellen
    plt.figure(figsize=(15, 6))
    plt.plot(time, ecg_signal, 'b-', linewidth=0.5)
    plt.xlabel('Zeit (s)')
    plt.ylabel('Amplitude')
    plt.title(f'ECG Signal - {patient_id} {run_info} - Kanal: {channel_name}')
    plt.grid(True, alpha=0.3)
    
    # Zeige Sampling-Info
    plt.text(0.02, 0.98, f'Sampling Rate: {fs} Hz\nGesamtdauer: {time_total:.1f}s', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"ECG-Daten geladen:")
    print(f"  - Kanäle: {data.channels}")
    print(f"  - Sampling Rates: {data.fs} Hz")
    print(f"  - Datenpunkte: {len(ecg_signal)}")
    print(f"  - Dauer: {time_total:.1f} Sekunden")

if __name__ == "__main__":
    # Dataset ist bereits im Projektordner verfügbar
    dataset_path = "./ds005873-download"
    
    print("ECG Plotting Script für SeizeIT2 Dataset")
    print("=" * 50)
    print(f"Dataset Pfad: {dataset_path}")
    print()
    
    # Lade und plotte ECG für ersten Patienten
    plot_patient_ecg(dataset_path, patient_id='sub-001', duration_minutes=2)