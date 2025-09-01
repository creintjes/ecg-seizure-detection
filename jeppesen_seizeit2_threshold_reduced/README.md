# Jeppesen SeizeIT2 - Threshold Reduced Version

Diese Version des Jeppesen SeizeIT2 Algorithmus verwendet **reduzierte Threshold-Werte** um mehr Alarme zu generieren.

## Änderungen gegenüber dem Original

### Threshold-Reduktion
- **Original**: `max_metric_value * 1.05` (5% über dem Baseline-Maximum)
- **Diese Version**: `max_metric_value * 0.95` (5% unter dem Baseline-Maximum)

### Konfiguration
Die Threshold-Reduktion wird über `config.py` gesteuert:

```python
# === THRESHOLD-PARAMETER ===
THRESHOLD_MULTIPLIER = 0.95  # Faktor für Cutoff-Berechnung (< 1.0 = mehr Alarme)
```

### Erwartete Auswirkungen
- **Mehr False Alarms**: Niedrigere Thresholds führen zu mehr Alarm-Trigger
- **Höhere Sensitivity**: Mehr echte Seizures werden erkannt
- **Niedrigere Precision**: Mehr False Positives

## Verwendung

```bash
# Standard-Lauf mit reduzierten Thresholds
python jeppesen_seizeit2.py

# Ergebnisse werden gespeichert in:
# /home/creintj2_sw/jeppesen/ecg-seizure-detection/jeppesen_seizeit2_threshold_reduced/results_threshold_reduced/
```

## Experimentelle Parameter

Verschiedene Threshold-Multiplier können getestet werden:
- `0.90`: Sehr aggressive Erkennung
- `0.95`: Moderate Erhöhung der Alarme (Standard)
- `1.00`: Exakt am Baseline-Maximum
- `1.05`: Original-Verhalten

## Dateistruktur

Identisch mit dem Original, aber separate Ergebnisordner:
- `results_threshold_reduced/`: Ausgabedateien mit reduzierten Thresholds
- Dateien haben Präfix: `seizeit2_jeppesen_threshold_reduced_*`