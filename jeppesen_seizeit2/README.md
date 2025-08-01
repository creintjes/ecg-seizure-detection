# Jeppesen 2019 Ansatz für SeizeIT2 Dataset

## Überblick

Diese Implementierung adaptiert den Jeppesen 2019 ECG-basierten Epilepsie-Vorhersage-Ansatz für das SeizeIT2 Dataset.

## Architektur

### Datenstruktur Unterschiede:
- **Original (Aarhus)**: TDMS-Dateien, patientenspezifische Ordnerstruktur
- **SeizeIT2**: EDF-Dateien, BIDS-Format mit `sub-XXX/ses-01/modality/` Struktur

### Feature-Pipeline:
1. **ECG Laden**: SeizeIT2 EDF-Format → ECG Channel
2. **R-Peak Erkennung**: Elgendi-Methode (8-20Hz Bandpass)
3. **RR-Intervall Extraktion**: Peak-zu-Peak Zeitabstände
4. **Feature-Berechnung**: CSI, ModCSI, HR-diff, Tachogram-Slope
5. **Schwellenwert-Klassifikation**: Baseline-basierte Cutoffs

### Dateien:
- `jeppesen_seizeit2.py` - Hauptskript mit Parallelverarbeitung
- `seizeit2_utils.py` - SeizeIT2-spezifische Utility-Funktionen
- `feature_extraction.py` - Jeppesen Feature-Implementierung
- `config.py` - Konfiguration für Datenpfade und Parameter

## Verwendung

```python
# Datenpfad anpassen in config.py
SEIZEIT2_DATA_PATH = "/path/to/seizeit2/bids"

# Ausführung
python jeppesen_seizeit2.py
```

## Anpassungen für SeizeIT2

1. **Datenloader**: EDF statt TDMS, BIDS-Struktur
2. **Annotation-Format**: TSV events statt proprietäre Seizure-IDs
3. **Sampling-Rate**: 250Hz (SeizeIT2) vs 512Hz (Original)
4. **Channel-Namen**: Automatische ECG-Channel-Erkennung