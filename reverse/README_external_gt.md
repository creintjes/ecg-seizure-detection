# Batch Evaluation with External Ground Truth

Evaluiert TimeVQVAE-AD Ergebnisse mit Ground Truth aus einem separaten Ordner.

## Problem / Use Case

Wenn Sie **dieselben Ground Truth Labels** haben, aber **unterschiedliche Anomaly Scores** (z.B. verschiedene Modell-Konfigurationen, Thresholds, etc.), dann können Sie mit diesem Skript die GT Labels aus einem anderen Ordner laden.

**Beispiel:**
```
ordner_a/117_config1.pkl   → Anomaly Scores (Konfiguration 1)
ordner_b/117_config2.pkl   → Anomaly Scores (Konfiguration 2)
gt_ordner/117_ground_truth.pkl  → Ground Truth Labels (Y)

Problem: ordner_a und ordner_b haben unterschiedliche Y, aber eigentlich sollten beide dasselbe GT verwenden!

Lösung: Verwende gt_ordner als --gt_dir
```

## Unterschied zu batch_evaluate_timevqvae.py

| Feature | batch_evaluate_timevqvae.py | batch_evaluate_with_external_gt.py |
|---------|----------------------------|-----------------------------------|
| Ground Truth | Aus dem Score-File selbst | Optional aus externem Ordner |
| Matching | - | Nach `dataset_index` (z.B. "117") |
| Bei fehlendem GT | Verwendet eigenes Y | **Überspringt File mit Warnung** |
| Use Case | Standard-Evaluation | Mehrere Score-Varianten, gleiches GT |

## Verwendung

### Basis (wie Original)

```bash
python batch_evaluate_with_external_gt.py <score_dir> \
    --config baseline \
    --window_type window
```

→ Verwendet Y aus den Score-Files selbst (wie Original)

### Mit externem GT

```bash
python batch_evaluate_with_external_gt.py <score_dir> \
    --gt_dir <ground_truth_dir> \
    --config optimized \
    --window_type window
```

→ Lädt Y aus `ground_truth_dir`, matched nach `dataset_index`

### Vollständiges Beispiel

```bash
python batch_evaluate_with_external_gt.py \
    ./scores/config_a/ \
    --gt_dir ./ground_truth_reference/ \
    --output_dir ./results_config_a/ \
    --config config_a \
    --window_type window \
    --scale 0.7 \
    --clustering time_90s \
    --no_plots
```

## Parameter

| Parameter | Required | Default | Beschreibung |
|-----------|----------|---------|--------------|
| `input_dir` | ✓ | - | Ordner mit Score-Files |
| `--gt_dir` | ✗ | None | Ordner mit Ground Truth Files |
| `--config` | ✓ | - | Config-Name (für seizure CSV) |
| `--window_type` | ✓ | - | Window-Type (für seizure CSV) |
| `--output_dir` | ✗ | input_dir/batch_results | Output-Ordner |
| `--scale` | ✗ | 0.7 | Threshold Scaling |
| `--clustering` | ✗ | time_90s | Clustering-Strategie |
| `--pattern` | ✗ | *joint_anomaly_score.pkl | File-Pattern |
| `--no_plots` | ✗ | False | Plots überspringen |

## Matching-Logik

**Wie werden Files gematched?**

1. **dataset_index** aus Score-File extrahieren (z.B. "117")
   - Primär: Aus `data['dataset_index']` im pkl
   - Fallback: Aus Dateiname via Regex `(\d{3})`

2. Im GT-Ordner nach File mit gleichem **dataset_index** suchen

3. Falls gefunden:
   - `Y` aus GT-File laden
   - Length-Check: `len(Y) == len(a_final)`
   - Bei Mismatch: **Warnung + Skip**

4. Falls NICHT gefunden:
   - **Warnung + Skip**

**Beispiel:**
```
Score-File: scores/117_window-joint_anomaly_score.pkl
  → dataset_index = "117"

GT-Ordner: ground_truth/
  ├── 115_ground_truth.pkl
  ├── 117_reference.pkl        ← MATCH! (dataset_index="117")
  └── 119_ground_truth.pkl

→ Lädt Y aus 117_reference.pkl
```

## Output

Wie `batch_evaluate_timevqvae.py`:

```
output_dir/
├── batch_results_TIMESTAMP.csv
├── seizure_detections_TIMESTAMP.csv  ← subject, seizure_idx, detected, config, window_type
├── summary_report_TIMESTAMP.txt
└── plots/...
```

## Warnungen

Das Skript gibt Warnungen in folgenden Fällen:

```
⚠ No external GT found for dataset_index=117 (117_scores.pkl). Skipping.
⚠ GT length mismatch for 117_scores.pkl: Y=1000, a_final=2000. Skipping.
⚠ GT file 117_gt.pkl has no 'Y' key
⚠ Failed to load GT from 117_gt.pkl: <error>
```

→ File wird **übersprungen**, aber im Summary als "failed" gelistet

## Vergleich mit Original ausführen

```bash
# 1. Mit eigenem GT (Original-Verhalten)
python batch_evaluate_with_external_gt.py ./scores/ \
    --config internal_gt --window_type window

# 2. Mit externem GT
python batch_evaluate_with_external_gt.py ./scores/ \
    --gt_dir ./reference_gt/ \
    --config external_gt --window_type window

# → Vergleiche die seizure_detections CSVs!
```

## Tipps

1. **GT-Ordner gut wählen**: Verwende den Ordner mit den "korrekten" Ground Truth Labels

2. **Konsistenz prüfen**: Alle Score-Files sollten Matches im GT-Ordner haben
   ```bash
   # Check vorher
   python count_ground_truths.py ./scores/
   python count_ground_truths.py ./gt_ordner/
   # → Sollten gleiche Anzahl Files/Events haben
   ```

3. **Batch-Processing**: Evaluiere mehrere Konfigurationen mit demselben GT
   ```bash
   for config in config_a config_b config_c; do
       python batch_evaluate_with_external_gt.py \
           ./scores_${config}/ \
           --gt_dir ./reference_gt/ \
           --config ${config} \
           --window_type window
   done
   ```

## Beispiel-Workflow

```bash
# Ordnerstruktur:
# scores_baseline/  → Baseline Scores
# scores_optimized/ → Optimierte Scores
# ground_truth/     → Referenz Ground Truth

# 1. Baseline evaluieren mit Reference GT
python batch_evaluate_with_external_gt.py \
    scores_baseline/ \
    --gt_dir ground_truth/ \
    --config baseline \
    --window_type window \
    --output_dir results/baseline/

# 2. Optimized evaluieren mit Reference GT
python batch_evaluate_with_external_gt.py \
    scores_optimized/ \
    --gt_dir ground_truth/ \
    --config optimized \
    --window_type window \
    --output_dir results/optimized/

# 3. Vergleiche Seizure Detections
# results/baseline/seizure_detections_*.csv
# results/optimized/seizure_detections_*.csv
# → Beide verwenden GLEICHE Ground Truth!
```
