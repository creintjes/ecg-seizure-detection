# Seizure Detection Tables - Unified Format

This directory contains seizure detection results from different models (Matrix Profile and TimeVQVAE-AD) and a unification script to convert them into a standardized format for comparison.

## Directory Structure

```
seizure_detection_tables/
├── seizure_detection_tables_MP/      # Matrix Profile results
│   ├── no_window/                    # Without detection window
│   │   ├── FAR/                      # Optimized for low False Alarm Rate
│   │   ├── HMS/                      # Optimized for Harmonic Mean Score
│   │   └── Sensitivity/              # Optimized for high Sensitivity
│   └── window/                       # With detection window
│       ├── FAR/
│       ├── HMS/
│       └── Sensitivity/
│
├── seizure_detection_tables_TIME/    # TimeVQVAE-AD results
│   ├── no_window/                    # Strict seizure boundaries
│   │   ├── FAR/
│   │   ├── HMS/
│   │   └── Sensitivity/
│   └── window/                       # Extended detection window
│       ├── FAR/
│       ├── HMS/
│       └── Sensitivity/
│
├── unify_detection_tables.py         # Unification script
└── README.md                         # This file
```

## Data Formats

### TimeVQVAE-AD Files (per configuration)
- **`.csv`** - Tabular data with columns:
  - `subject_id`: Patient ID (e.g., sub-098)
  - `run_id`: Recording run (e.g., run-03)
  - `seizure_index`: Seizure number within run
  - `start_time`: Seizure start time in seconds
  - `end_time`: Seizure end time in seconds
  - `duration`: Seizure duration in seconds
  - `detected`: Boolean - was seizure detected?

- **`.json`** - Structured data with additional metadata

- **`_summary.txt`** - Human-readable summary report

### Matrix Profile Files (per configuration)
- **`.xlsx`** - Excel file with columns:
  - `sub`: Patient ID
  - `run`: Recording run
  - `seizure_index`: Seizure number
  - `detected`: Boolean - was seizure detected?

## Unification Script

### Purpose

The `unify_detection_tables.py` script converts detection results from both models into a unified format for:
- Direct model comparison
- Combined analysis
- Standardized reporting

### Features

1. **Data Unification**: Combines data from both models into a single format
2. **Timing Enrichment**: Adds timing information from TimeVQVAE-AD to Matrix Profile data
3. **Summary Statistics**: Generates overall performance metrics per model/configuration
4. **Model Comparison**: Creates side-by-side comparison tables
5. **Per-Subject Analysis**: Provides detailed subject-level comparison

### Usage

Basic usage (from within the `seizure_detection_tables` directory):

```bash
cd seizure_detection_tables
python unify_detection_tables.py
```

With custom directories:

```bash
python unify_detection_tables.py \
    --base-dir /path/to/seizure_detection_tables \
    --output-dir /path/to/output
```

### Output Files

The script generates the following files in `unified_results/`:

#### Main Files
- **`unified_seizure_detection_all.csv`** - Complete unified dataset with all models and configurations
- **`unified_summary_statistics.csv`** - Overall performance metrics

#### Comparison Files
- **`unified_model_comparison.csv`** - Side-by-side model performance comparison
- **`unified_per_subject_comparison.csv`** - Per-subject detection rates

#### Model-Specific Files
- **`unified_matrix_profile.csv`** - All Matrix Profile results
- **`unified_timevqvae-ad.csv`** - All TimeVQVAE-AD results

#### Configuration-Specific Files
- **`unified_far_window.csv`** - FAR configuration with window
- **`unified_far_no_window.csv`** - FAR configuration without window
- **`unified_hms_window.csv`** - HMS configuration with window
- **`unified_hms_no_window.csv`** - HMS configuration without window
- **`unified_sensitivity_window.csv`** - Sensitivity configuration with window
- **`unified_sensitivity_no_window.csv`** - Sensitivity configuration without window

## Unified Data Format

The unified CSV files contain the following standardized columns:

### Core Columns
- **`subject`**: Patient ID (standardized to sub-XXX format)
- **`run`**: Recording run ID
- **`seizure_idx`**: Seizure index within run
- **`detected`**: Boolean - seizure detection status

### Timing Columns (from TimeVQVAE-AD or enriched)
- **`seizure_start`**: Start time in seconds
- **`seizure_end`**: End time in seconds
- **`seizure_duration`**: Duration in seconds

### Model Information
- **`model`**: Model name (Matrix Profile or TimeVQVAE-AD)
- **`config`**: Configuration (FAR, HMS, or Sensitivity)
- **`window_type`**: Detection window type (window or no_window)

### Algorithm Parameters
- **`time_threshold`**: Time-based threshold (TimeVQVAE-AD only)
- **`score_threshold`**: Anomaly score threshold (TimeVQVAE-AD only)

## Example Analysis Workflow

```python
import pandas as pd

# Load unified data
df = pd.read_csv('unified_results/unified_seizure_detection_all.csv')

# Compare models for Sensitivity configuration with window
sensitivity_window = df[
    (df['config'] == 'Sensitivity') &
    (df['window_type'] == 'window')
]

# Calculate sensitivity per model
sensitivity_by_model = sensitivity_window.groupby('model')['detected'].agg(['sum', 'count', 'mean'])
print(sensitivity_by_model)

# Load summary statistics
summary = pd.read_csv('unified_results/unified_summary_statistics.csv')
print(summary)

# Load model comparison
comparison = pd.read_csv('unified_results/unified_model_comparison.csv')
print(comparison)
```

## Notes

- **Timing Information**: Matrix Profile data originally lacks timing information. The script enriches it using TimeVQVAE-AD data as a reference.
- **Subject Matching**: Subjects sub-097 to sub-125 represent the test set.
- **Detection Windows**:
  - `window`: Extended detection window (typically ±5min pre, ±3min post)
  - `no_window`: Strict seizure boundaries (0s tolerance)

## Requirements

```bash
pip install pandas openpyxl
```

## Contact

For questions or issues with the unification script, please refer to the main project README.
