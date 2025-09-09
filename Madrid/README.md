# Madrid Algorithm - ECG Seizure Detection

This folder contains the implementation and analysis scripts for the Madrid algorithm applied to ECG-based seizure detection using the SeizeIT2 dataset.

## Reproducing Results

Follow these steps to reproduce the results presented in the paper:

### 1. Configuration Setup

Before running any scripts, ensure the data paths are correctly configured in the project root:

**File: `config.py`**
```python
# Data paths
RAW_DATA_PATH = "/home/swolf/asim_shared/raw_data/ds005873-1.1.0"
PREPROCESSED_DATA_PATH = "/home/swolf/asim_shared/preprocessed_data"
```

Update these paths according to your local setup:
- `RAW_DATA_PATH`: Path to the downloaded SeizeIT2 dataset
- `PREPROCESSED_DATA_PATH`: Directory where preprocessed files will be saved

> **Note for supervisors (Tim & Simon):** Raw data is available on both ds01 and ds03 servers at the path specified above: `/home/swolf/asim_shared/raw_data/ds005873-1.1.0`

### 2. Data Preprocessing

The first step is to preprocess the raw ECG data using the parameters specified in our paper:

**Window Configuration (as used in paper):**
- Window size: 3600 seconds (1 hour)
- Stride: 1800 seconds (30 minutes, 50% overlap)
- Sampling frequency: 8 Hz (downsampled from original 250 Hz)

**Go in Madrid Folder:**
```bash
cd Madrid
```

**Run preprocessing:**
```bash
python preprocessing/preprocess_all_data.py
```

This script will:
- Load raw ECG data from `RAW_DATA_PATH`
- Apply bandpass filtering (0.5-40 Hz)
- Downsample to 8 Hz
- Create 3600-second windows with 1800-second stride
- Save preprocessed data to `PREPROCESSED_DATA_PATH/downsample_freq=8,window_size=3600_0,stride=1800_0_reproduced/`
- Generate preprocessing summary statistics

**Expected output:**
- Individual `.pkl` files for each recording: `{subject_id}_{run_id}_preprocessed.pkl`
- Summary file: `preprocessing_summary.xlsx`
- Processing statistics and data overview

> **Note for supervisors (Tim & Simon):** Preprocessed data is available on both ds01 and ds03 servers at: `/home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new`

### 3. Handle Short Recordings 

Some ECG recordings may be shorter than the 3600-second window size, resulting in files with 0 windows that cannot be analyzed by Madrid. This step recovers these "lost" recordings using flexible windowing.

**Check for files with 0 windows:**
```bash
python preprocessing/reprocess_empty_windows.py \
    --preprocessed-dir /path/to/preprocessed/data \
    --dry-run
```

**Reprocess empty files with flexible windowing (recommended approach):**
```bash
python preprocessing/reprocess_empty_windows.py \
    --preprocessed-dir /home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new \
    --output-dir /home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new \
    --original-data-dir /path/to/seizeit2/data
```

**Parameter explanation:**
- `--preprocessed-dir`: Directory containing the preprocessed files (including those with 0 windows)
- `--output-dir`: Where to save the reprocessed files with flexible windows
- `--original-data-dir`: Path to raw SeizeIT2 dataset (needed to reload original ECG data)

**Why set output-dir to the same directory:**
- **All data in one place**: Both original and recovered files in the same location
- **Safe overwriting**: Only replaces the empty files (0 windows), preserves all other files

> **Note for supervisors (Tim & Simon):** Reprocessed data (including recovered short recordings) is available on both ds01 and ds03 servers at: `/home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new`

### 4. Madrid Algorithm Analysis

After preprocessing is complete, run the Madrid algorithm analysis. 

**Example Command (as used in our paper):**
```bash
python madrid_windowed_batch_processor_parallel.py \
    --data-dir /home/swolf/asim_shared/preprocessed_data/downsample_freq=8,window_size=3600_0,stride=1800_0_new \
    --existing-results-dir results_8hz_window3600_stride1800_new20min \
    --output-dir results_8hz_window3600_stride1800_new20min \
    --n-workers 20 \
    --train-minutes 20
```

**Parameter Explanation:**
- `--data-dir`: Directory containing preprocessed .pkl files from step 2
- `--existing-results-dir`: Skip files already processed (for resuming interrupted runs)
- `--output-dir`: Directory where JSON results will be saved
- `--n-workers 20`: Use 20 parallel processes (adjust based on your CPU cores)
- `--train-minutes 20`: Use first 20 minutes of each recording for training

**Expected Output:**
- JSON files with Madrid analysis results: `madrid_windowed_results_{subject_id}_{run_id}_{timestamp}.json`
- Processing logs and progress information

> **Note for supervisors (Tim & Simon):** Madrid analysis results are available on ds01 server at: `/home/creintj2_sw/ecg-seizure-detection/Madrid/results_madrid/results_8hz_window3600_stride1800_new20min` (Runtime: ~4 days total. You can use these existing results for reproduction instead of re-running the analysis.)

### 5. False Alarm Reduction with Clustering

After the Madrid algorithm analysis, apply clustering-based false alarm reduction to improve detection performance.

**Run the clustering false alarm reducer:**
```bash
python madrid_clustering_false_alarm_reducer_train_test.py <results_dir>
```

**Example Commands (as used in our paper):**

*Without extended seizure window (strict seizure boundaries):*
```bash
python madrid_clustering_false_alarm_reducer_train_test.py results_8hz_window3600_stride1800_new20min -t 0.01 --pre-seizure-minutes 0 --post-seizure-minutes 0
```

*With extended seizure window (default settings):*
```bash
python madrid_clustering_false_alarm_reducer_train_test.py results_8hz_window3600_stride1800_new20min -t 0.01
```

**Parameter Options:**
- `results_dir` (required): Directory containing Madrid windowed results JSON files from step 4
- `-o OUTPUT_DIR`: Output directory for clustering results (default: `results_dir/train_test_clustered_results`)
- `-t THRESHOLD`: Anomaly score threshold for detection (default: use top-ranked anomaly per window)
- `--pre-seizure-minutes`: Minutes before seizure start to consider as detection window (default: 5.0)
- `--post-seizure-minutes`: Minutes after seizure end to consider as detection window (default: 3.0)

**Paper Analysis Approach:**
We conducted the analysis with two different seizure window configurations to evaluate their impact on clustering strategies:

1. **Strict seizure boundaries** (`--pre-seizure-minutes 0 --post-seizure-minutes 0`): Only the exact annotated seizure period is considered for detection evaluation
2. **Extended seizure window** (default: 5 minutes pre, 3 minutes post): Allows detection within a wider time window around seizures

Each configuration typically results in different optimal clustering strategies.

**What this step does:**
- Performs train/test split clustering-based false alarm reduction
- Uses subjects 001-096 for training, subjects 097-125 for testing
- Evaluates multiple clustering strategies
- Selects best strategy based on training set performance
- Applies selected strategy to test set
- Reduces false alarms while maintaining seizure detection sensitivity

**Expected Output:**
- Strategy comparison results in `train_test_clustered_results/strategy_comparison/`
- Metrics before clustering in `train_test_clustered_results/metrics_before/`
- Metrics after clustering in `train_test_clustered_results/metrics_after/`
- Final clustered results in `train_test_clustered_results/clusters/`
- Complete summary in `train_test_clustered_results/complete_train_test_results_{timestamp}.json`

> **Note for supervisors (Tim & Simon):** Clustering results are available on ds01 server at:
> - **Without extended seizure window:** `/home/creintj2_sw/ecg-seizure-detection/Madrid/results_madrid/clustering_withoutSDW`
> - **With extended seizure window:** `/home/creintj2_sw/ecg-seizure-detection/Madrid/results_madrid/clustering_withSDW`

### 6. Threshold Trade-off Analysis (Test Set Only)

After determining the optimal clustering strategies from step 5, perform threshold trade-off analysis to generate Sensitivity vs False Alarm Rate curves. This analysis uses the best clustering strategy identified in the previous step and evaluates it on the test set only (subjects 097-125).

**Choose the appropriate script based on your clustering results:**

*With extended seizure windows (withSDW) - uses time_180s clustering strategy:*
```bash
python madrid_clustered_threshold_tradeoff_test_only_withSDW.py <results_dir>
```

*Without extended seizure windows (withoutSDW) - uses time_600s clustering strategy:*
```bash
python madrid_clustered_threshold_tradeoff_test_only_withoutSDW.py <results_dir>
```

**Example Commands (based on paper results):**
```bash
# For extended seizure window analysis (typically uses time_180s strategy)
python madrid_clustered_threshold_tradeoff_test_only_withSDW.py results_8hz_window3600_stride1800_new20min -n 100

# For strict seizure boundary analysis (typically uses time_600s strategy)  
python madrid_clustered_threshold_tradeoff_test_only_withoutSDW.py results_8hz_window3600_stride1800_new20min -n 100 --post-seizure-minutes 0 --pre-seizure-minutes 0
```

**Parameter Options:**
- `results_dir` (required): Directory containing Madrid windowed results JSON files
- `-o OUTPUT_DIR`: Output directory for plots and results (default: `results_dir/test_only_clustered_tradeoff`)
- `-n NUM_THRESHOLDS`: Number of thresholds to test (default: 50, recommended: 100)
- `--pre-seizure-minutes`: Minutes before seizure start to consider as detection window (default: 5.0)
- `--post-seizure-minutes`: Minutes after seizure end to consider as detection window (default: 3.0)

**Script Selection Note:**
The choice between `withSDW` and `withoutSDW` scripts depends on your optimal clustering strategy from step 5:
- If your best strategy was **time_180s** → use `madrid_clustered_threshold_tradeoff_test_only_withSDW.py`
- If your best strategy was **time_600s** → use `madrid_clustered_threshold_tradeoff_test_only_withoutSDW.py`

**Important:** If you want to use a different clustering strategy than the hardcoded ones (time_180s or time_600s), you need to modify line 37 in the respective script:
```python
self.clustering_time_threshold = 180  # Change this value to your desired strategy (e.g., 300, 900, 1200, 1500, 1800)
```

**What this step does:**
- Tests multiple anomaly score thresholds 
- Applies the specified clustering strategy to reduce false alarms
- Evaluates only on test set (subjects 097-125) to avoid overfitting
- Generates Sensitivity vs False Alarm Rate trade-off curves

**Expected Output:**
- Trade-off analysis plots in `test_only_clustered_tradeoff/`
- CSV files with threshold performance metrics
- JSON files with detailed results for each threshold
- Sensitivity vs FAR curves for publication

> **Note for supervisors (Tim & Simon):** Threshold trade-off results are available on ds01 server at:
> - **With extended seizure window:** `/home/creintj2_sw/ecg-seizure-detection/Madrid/results_madrid/trade_off_FAR_vs_Sens_withSDW`
> - **Without extended seizure window:** `/home/creintj2_sw/ecg-seizure-detection/Madrid/results_madrid/trade_off_FAR_vs_Sens_n100_withoutSDW`

### 7. Seizure Type Analysis

After completing the threshold trade-off analysis, perform seizure type analysis to understand which types of seizures are best detected by the Madrid algorithm using the optimal clustering strategy.

**Run the seizure type analysis:**
```bash
python madrid_clustered_seizure_type_analysis.py <results_dir>
```

**Example Command:**
```bash
python madrid_clustered_seizure_type_analysis.py results_8hz_window3600_stride1800_new20min -t 0.01
```

**Parameter Options:**
- `results_dir` (required): Directory containing Madrid windowed results JSON files
- `-o OUTPUT_DIR`: Output directory for analysis results (default: `results_dir/seizure_type_analysis`)
- `-t THRESHOLD`: Anomaly score threshold for detection (default: use top-ranked anomaly per window)
- `--pre-seizure-minutes`: Minutes before seizure start to consider as detection window (default: 5.0)
- `--post-seizure-minutes`: Minutes after seizure end to consider as detection window (default: 3.0)

**What this step does:**
- Uses time_180s clustering strategy (same as withSDW analysis)
- Analyzes detection performance for different seizure types (focal, generalized, etc.)
- Generates seizure type-specific sensitivity and false alarm metrics
- Provides insights into which seizure characteristics are better detected
- Creates visualizations comparing detection rates across seizure types

**Expected Output:**
- Seizure type analysis report in `seizure_type_analysis/`
- Detection metrics breakdown by seizure type (JSON format)
- Visualization plots showing performance per seizure type
- Statistical analysis of seizure type detection patterns

> **Note for supervisors (Tim & Simon):** Seizure type analysis results are available on ds01 server at: `/home/creintj2_sw/ecg-seizure-detection/Madrid/results_madrid/seizureTypeAnalysis`


### 8. File Structure

```
Madrid/
├── README.md                           # This file
├── preprocessing/
│   └── preprocess_all_data.py
│   └── reprocess_empty_windows.py
│   └── preprocessing.py
├── madrid_windowed_batch_processor_parallel.py          # Madrid analysis
├── madrid_clustering_false_alarm_reducer_train_test.py  # Clustering false alarm reduction
├── madrid_clustered_threshold_tradeoff_test_only_withSDW.py    # Threshold trade-off (time_180s)
├── madrid_clustered_threshold_tradeoff_test_only_withoutSDW.py # Threshold trade-off (time_600s)
├── madrid_clustered_seizure_type_analysis.py            # Seizure type analysis
├── models/
│   └── madrid_v2.py                   # Madrid algorithm implementation
└── results/                           # Analysis outputs
```



### 9. Expected Runtime

- **Preprocessing**: 1-2 hours (depending on hardware and number of recordings)
- **Flexible Window Recovery**: 30 minutes
- **Madrid Analysis**: 4 days in total (depending on amount of workers and hardware)
- **Clustering False Alarm Reduction**: 5-10 minutes 
- **Threshold Trade-off Analysis**: 5-10 minutes 
- **Seizure Type Analysis**: 5-10 minutes 

