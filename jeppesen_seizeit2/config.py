"""
Konfiguration für Jeppesen SeizeIT2 Anpassung
"""

import os
from pathlib import Path

# === DATENPFADE ===
# WICHTIG: Diesen Pfad an Ihre lokale SeizeIT2 Installation anpassen!
SEIZEIT2_DATA_PATH = Path("/home/swolf/asim_shared/raw_data/ds005873-1.1.0")    #"/home/creintj2_sw/ecg-seizure-detection/jeppesen_seizeit2/example_data")  # <- HIER ANPASSEN

# Ausgabeordner für Ergebnisse
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# === VERARBEITUNGSPARAMETER ===
RUN_PREFIX = "seizeit2_jeppesen"
PEAK_DETECTION_METHOD = 'elgendi'
SEIZURE_TYPE = "analysis"  # Verwendet alle Seizure-Events aus SeizeIT2

# Padding um Seizures (in Anzahl RR-Intervallen)
SEIZURE_PADDING_CUTOFF = (120, 120)    # Für Cutoff-Berechnung
SEIZURE_PADDING_EVAL = (120, 100)      # Für Evaluation

# === FEATURE-PARAMETER ===
WINDOW_LENGTHS = [50, 100]  # CSI/ModCSI/HR-diff Window-Größen
REFRACTORY_PERIOD_MINUTES = 3  # Refraktäre Periode nach Vorhersage

# === PARALLELVERARBEITUNG ===
MAX_WORKERS = 5  # Anzahl paralleler Prozesse

# === CHECKPOINT KONFIGURATION ===
ENABLE_CHECKPOINTS = True  # Aktiviert Checkpoint-Funktionalität
CHECKPOINT_BATCH_SIZE = 3  # Anzahl Subjects pro Batch vor Checkpoint-Speicherung
SUBJECT_TIMEOUT_MINUTES = 60  # Timeout pro Subject in Minuten

# === ELGENDI R-PEAK DETECTION PARAMETER ===
ELGENDI_PARAMS = {
    'low': 8,           # Untere Grenzfrequenz (Hz)
    'high': 20,         # Obere Grenzfrequenz (Hz)
    'order': 3,         # Butterworth Filter-Ordnung
    'w1factor': 0.12,   # QRS Window Faktor
    'w2factor': 0.65,   # Beat Window Faktor
    'beta': 0.08        # Schwellenwert-Skalierung
}

# === SEIZEIT2 SPEZIFISCHE PARAMETER ===
SAMPLING_RATE = 250  # SeizeIT2 Standard-Sampling-Rate
MODALITIES = ['ecg']  # Nur ECG für diese Anwendung

# === ENSEMBLE KOMBINATIONEN ===
ENSEMBLE_COMBINATIONS = [
    ['modcsi_100xslope_100', 'modcsi_100_filteredxslope_100'],
    ['modcsi_100xslope_100', 'csi_100xslope_100'],
    ['modcsi_100xslope_100', 'csi_100_filteredxslope_100'],
    ['modcsi_100_filteredxslope_100', 'csi_100xslope_100'],
    ['modcsi_100_filteredxslope_100', 'csi_100_filteredxslope_100'],
    ['csi_100xslope_100', 'csi_100_filteredxslope_100']
]

def validate_config():
    """Validiert die Konfiguration"""
    if not SEIZEIT2_DATA_PATH.exists():
        raise FileNotFoundError(
            f"SeizeIT2 Datenpfad nicht gefunden: {SEIZEIT2_DATA_PATH}\n"
            "Bitte SEIZEIT2_DATA_PATH in config.py anpassen!"
        )
    
    # Prüfe ob BIDS-Struktur vorhanden
    subjects = list(SEIZEIT2_DATA_PATH.glob("sub-*"))
    if not subjects:
        raise ValueError(
            f"Keine Subjects (sub-*) in {SEIZEIT2_DATA_PATH} gefunden!\n"
            "Überprüfen Sie die BIDS-Struktur."
        )
    
    print(f"✓ SeizeIT2 Pfad validiert: {len(subjects)} Subjects gefunden")
    return True