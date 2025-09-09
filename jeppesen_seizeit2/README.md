# Jeppesen 2019 Algorithm - ECG Seizure Detection

This folder contains the implementation and analysis scripts for the Jeppesen 2019 algorithm adapted for the SeizeIT2 dataset. The algorithm uses ECG-based heart rate variability features for seizure detection.


## Reproducing Results

Follow these steps to reproduce the results presented in the paper:

### 1. Configuration Setup

Before running any scripts, ensure the data paths are correctly configured:

**File: `config.py`**
```python
# === DATENPFADE ===
SEIZEIT2_DATA_PATH = Path("/home/swolf/asim_shared/raw_data/ds005873-1.1.0")
RESULTS_DIR = Path("./results_full")
```

Update these paths according to your local setup:
- `SEIZEIT2_DATA_PATH`: Path to the downloaded SeizeIT2 dataset
- `RESULTS_DIR`: Directory where results will be saved

> **Note for supervisors (Tim & Simon):** Raw data is available on both ds01 and ds03 servers at: `/home/swolf/asim_shared/raw_data/ds005873-1.1.0`

### 2. Run Jeppesen Algorithm Analysis

Execute the main Jeppesen algorithm analysis on the SeizeIT2 dataset:

**Go into the right folder:**
```bash
cd jeppesen_seizeit2
```

**Run the algorithm:**
```bash
python jeppesen_seizeit2.py
```

**Key Parameters (configured in config.py):**
- **Peak detection method**: Elgendi algorithm
- **Seizure padding**: 120 RR-intervals before, 120 after (cutoff), 120 before, 100 after (evaluation)
- **Feature windows**: [50, 100] RR-intervals for CSI/ModCSI/HR-diff calculations
- **Refractory period**: 3 minutes after prediction
- **Parallel processing**: 10 workers (configurable via MAX_WORKERS)

**What this step does:**
- Processes all subjects in the SeizeIT2 dataset
- Extracts R-peaks using Elgendi method
- Calculates heart rate variability features (CSI, ModCSI, HR differences)
- Computes tachogram slope features
- Generates ensemble feature combinations
- Applies seizure detection using various parameter combinations
- Outputs detailed CSV results with per-subject metrics

**Expected Output:**
- CSV files with detection results: `seizeit2_jeppesen_detection_elgendi_padding_{parameters}_{timestamp}.csv`
- Interim checkpoint files for recovery: `interim_seizeit2_jeppesen_detection_*.csv`
- Processing logs and progress information

> **Note for supervisors (Tim & Simon):** Jeppesen analysis results are available in this repository at: `/jeppesen_seizeit2/results_full` (Runtime: ~2-3 days)

### 3. Analyze Results

After the Jeppesen algorithm analysis, analyze the results to calculate overall performance metrics:

**Run the results analysis:**
```bash
python analyze_results.py <path_to_csv_file>
```

**Example Command:**
```bash
python analyze_results.py results_full/seizeit2_jeppesen_detection_elgendi_padding_120-120-120-100_20250821_105628.csv --output-txt jeppesen_analysis_report.txt
```

**Parameter Options:**
- `csv_file` (required): Path to the results CSV file from step 2
- `--output OUTPUT`: Output CSV file for summary metrics (optional)
- `--output-txt OUTPUT_TXT`: Output TXT file for detailed analysis report (optional)

**What this step does:**
- Loads results from CSV file
- Calculates overall sensitivity across all subjects and seizures
- Computes false alarms per hour using actual recording durations
- Performs train/test split analysis (subjects 001-096 for training, 097-125 for testing)
- Generates detailed performance statistics and reports
- Provides per-subject breakdown and ensemble method comparisons

**Expected Output:**
- Detailed analysis report (if --output-txt specified)
- Summary metrics CSV (if --output specified)
- Console output with key performance indicators (sensitivity, FAH)
- Train/test split performance comparison

> **Note for supervisors (Tim & Simon):** Analysis results and reports are available in this repository at: `/jeppesen_seizeit2/results_full`

### 4. File Structure

```
jeppesen_seizeit2/
├── README.md                    # This file
├── config.py                    # Configuration parameters
├── jeppesen_seizeit2.py         # Main algorithm implementation
├── analyze_results.py           # Results analysis script
├── seizeit2_utils.py           # SeizeIT2 dataset utilities
├── feature_extraction.py       # Heart rate variability feature extraction
├── evaluation_utils.py         # Evaluation and metrics utilities
└── results_full/               # Analysis results
    ├── seizeit2_jeppesen_detection_*.csv
    ├── seizeit2_jeppesen_detection_*_analysis_report.txt
    ├── seizeit2_jeppesen_detection_*_train_test_summary.csv
    └── interim_seizeit2_jeppesen_detection_*.csv
```

### 5. Expected Runtime

- **Jeppesen Algorithm Analysis**: 2-3 days (depending on amount of workers and hardware)
- **Results Analysis**: 5-10 minutes 

