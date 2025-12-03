# TimeVQVAE-AD Batch Evaluation

Batch-Verarbeitung von TimeVQVAE-AD Ergebnissen für mehrere Dateien.

## Features

- ✅ Verarbeitet alle `*joint_anomaly_score.pkl` Dateien in einem Ordner
- ✅ CSV Export mit allen Metriken
- ✅ Einzelne Plots pro Datei (optional)
- ✅ Summary Report (TXT)
- ✅ Progress Bar
- ✅ Robustes Error Handling (überspringt fehlerhafte Dateien)

## Verwendung

### Basis-Kommando

```bash
python batch_evaluate_timevqvae.py <input_ordner>
```

### Beispiele

```bash
# Evaluiere alle Files im evaluation/results Ordner
python batch_evaluate_timevqvae.py ../TimeVQVAE-AD/evaluation/results/

# Mit custom Output-Ordner
python batch_evaluate_timevqvae.py ../TimeVQVAE-AD/evaluation/results/ --output_dir ./my_results

# Anderen Scale-Faktor verwenden
python batch_evaluate_timevqvae.py ../TimeVQVAE-AD/evaluation/results/ --scale 0.65

# Andere Clustering-Strategie
python batch_evaluate_timevqvae.py ../TimeVQVAE-AD/evaluation/results/ --clustering time_120s

# Ohne Plots (schneller)
python batch_evaluate_timevqvae.py ../TimeVQVAE-AD/evaluation/results/ --no_plots
```

### Parameter

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `input_dir` | *required* | Ordner mit pkl-Dateien |
| `--output_dir` | `input_dir/batch_results` | Ausgabe-Ordner |
| `--scale` | `0.7` | Threshold Scaling-Faktor |
| `--clustering` | `time_90s` | Clustering-Strategie |
| `--pattern` | `*joint_anomaly_score.pkl` | Datei-Pattern |
| `--no_plots` | `False` | Plots überspringen |

## Output

Das Skript erzeugt:

```
output_dir/
├── batch_results_20251203_143022.csv          # Alle Metriken
├── summary_report_20251203_143022.txt         # Text-Zusammenfassung
└── plots/                                      # Plots pro Datei
    ├── 001/
    │   └── 001_timeline.png
    ├── 002/
    │   └── 002_timeline.png
    └── ...
```

### CSV Spalten

**Dataset Info:**
- `file`, `dataset_index`, `subject_id`, `run_id`
- `duration_seconds`, `duration_hours`, `total_samples`
- `true_anomaly_samples`, `true_anomaly_percentage`
- `n_truth_events`

**Threshold Info:**
- `scale`, `original_threshold`, `scaled_threshold`
- `max_score`, `min_score`, `mean_score`

**Baseline Metrics:**
- `baseline_pred_samples`, `baseline_pred_percentage`
- `baseline_n_pred_events`
- `baseline_TP`, `baseline_FN`, `baseline_FP`
- `baseline_sensitivity`, `baseline_iou`
- `baseline_far_per_hour`

**Clustered Metrics** (wenn Clustering verfügbar):
- `clustered_n_clusters`, `clustered_n_representatives`
- `clustered_pred_samples`, `clustered_pred_percentage`
- `clustered_n_pred_events`
- `clustered_TP`, `clustered_FN`, `clustered_FP`
- `clustered_sensitivity`, `clustered_iou`
- `clustered_far_per_hour`
- `far_reduction_percentage`

## Clustering-Strategien

- `time_30s` - Cluster anomalies innerhalb 30s window
- `time_60s` - 60s window
- `time_90s` - 90s window (default)
- `time_120s` - 120s window
- `time_180s` - 180s window

## Hinweise

### Clustering Module

Das Skript benötigt das Clustering-Modul aus `TimeVQVAE-AD/evaluation/clustering.py`.

Falls nicht verfügbar:
- Warnung wird angezeigt
- Nur Baseline-Metriken werden berechnet
- Clustering-Spalten fehlen im CSV

### Fehlerhafte Dateien

Fehlerhafte Dateien werden übersprungen und geloggt:
```
⚠ Failed files (3):
  - 117_window-joint_anomaly_score.pkl
  - 118_no_window-joint_anomaly_score.pkl
  ...
```

### Performance

- Sequenzielle Verarbeitung mit Progress Bar
- ~5-10 Sekunden pro Datei (mit Plots)
- ~1-2 Sekunden pro Datei (ohne Plots: `--no_plots`)

## Beispiel-Output

```
================================================================================
TIMEVQVAE-AD BATCH EVALUATION
================================================================================
Input directory: evaluation/results/runs/window_joint_anomaly
Output directory: evaluation/results/runs/window_joint_anomaly/batch_results
Files found: 739
Scale: 0.7
Clustering: time_90s
Clustering available: True
Generate plots: True

Processing files: 100%|████████████████| 739/739 [1:23:45<00:00,  8.84it/s]

✓ Saved CSV: batch_results/batch_results_20251203_143022.csv
✓ Saved report: batch_results/summary_report_20251203_143022.txt

================================================================================
BATCH EVALUATION COMPLETE
================================================================================
Successfully processed: 739/739 files

Mean Baseline Sensitivity: 0.140
Mean Baseline FAR: 54.26 events/hour

Mean Clustered Sensitivity: 0.140
Mean Clustered FAR: 15.84 events/hour
Mean FAR Reduction: 70.8%

Results saved to: evaluation/results/runs/window_joint_anomaly/batch_results
```

## Troubleshooting

**ModuleNotFoundError: torch**
- Lösung: Verwende die richtige Python-Umgebung mit installierten Dependencies

**Clustering module not available**
- Normal, wenn `TimeVQVAE-AD/evaluation/clustering.py` nicht gefunden wird
- Baseline-Evaluation funktioniert trotzdem

**No files matching pattern found**
- Prüfe das `--pattern` Argument
- Prüfe ob Dateien im Ordner vorhanden sind
