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

### 2. Data Preprocessing

The first step is to preprocess the raw ECG data using the parameters specified in our paper:

**Window Configuration (as used in paper):**
- Window size: 3600 seconds (1 hour)
- Stride: 1800 seconds (30 minutes, 50% overlap)
- Sampling frequency: 8 Hz (downsampled from original 250 Hz)

**Run preprocessing:**
```bash
python preprocessing/Madrid/preprocess_all_data.py
```

This script will:
- Load raw ECG data from `RAW_DATA_PATH`
- Apply bandpass filtering (0.5-40 Hz)
- Downsample to 8 Hz
- Create 3600-second windows with 1800-second stride
- Save preprocessed data to `PREPROCESSED_DATA_PATH/downsample_freq=8,window_size=3600_0,stride=1800_0_new/`
- Generate preprocessing summary statistics

**Expected output:**
- Individual `.pkl` files for each recording: `{subject_id}_{run_id}_preprocessed.pkl`
- Summary file: `preprocessing_summary.xlsx`
- Processing statistics and data overview

### 3. Handle Short Recordings (Optional but Recommended)

Some ECG recordings may be shorter than the 3600-second window size, resulting in files with 0 windows that cannot be analyzed by Madrid. This step recovers these "lost" recordings using flexible windowing.

**Check for files with 0 windows:**
```bash
python preprocessing/Madrid/reprocess_empty_windows.py \
    --preprocessed-dir /path/to/preprocessed/data \
    --dry-run
```

**Reprocess empty files with flexible windowing (recommended approach):**
```bash
python preprocessing/Madrid/reprocess_empty_windows.py \
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

### 5. File Structure

```
Madrid/
├── README.md                           # This file
├── preprocessing/
│   └── Madrid/
│       └── preprocess_all_data.py     # Main preprocessing script
├── madrid_windowed_batch_processor_parallel.py  # Madrid analysis
├── models/
│   └── madrid_v2.py                   # Madrid algorithm implementation
└── results/                           # Analysis outputs
```


### 6. Paper Parameters Summary

The following parameters were used in our paper and are configured in the preprocessing script:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Window Size | 3600 seconds | 1-hour analysis windows |
| Stride | 1800 seconds | 30-minute stride (50% overlap) |
| Sampling Rate | 8 Hz | Downsampled from original 250 Hz |
| Filter Range | 0.5-40 Hz | Bandpass filter |
| Dataset | SeizeIT2 | 125 patients, 11,000+ hours |

### 7. Hardware Requirements

- **Memory**: Minimum 16 GB RAM recommended
- **Storage**: ~50-100 GB for preprocessed data
- **CPU**: Multi-core processor recommended for parallel processing
- **GPU**: Optional, can accelerate Madrid algorithm

### 8. Troubleshooting

**Common Issues:**

1. **Path not found error**: 
   - Verify `config.py` paths are correct
   - Ensure SeizeIT2 dataset is downloaded and extracted

2. **Memory errors during preprocessing**:
   - Reduce batch size in preprocessing script
   - Process subset of recordings first

3. **Import errors**:
   - Ensure all dependencies are installed
   - Check Python path includes project root

**Dependencies:**
```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

### 9. Expected Runtime

- **Preprocessing**: 2-6 hours (depending on hardware and number of recordings)
- **Flexible Window Recovery**: 30 minutes - 2 hours (only for affected files)
- **Madrid Analysis**: 1-4 hours per batch (depending on parameters and hardware)

### 10. Output Verification

After successful preprocessing, verify the output:

```bash
ls -la /path/to/preprocessed/data/downsample_freq=8,window_size=3600_0,stride=1800_0_new/
```

You should see:
- Multiple `.pkl` files (one per recording)
- `preprocessing_summary.xlsx` with statistics
- File sizes indicating successful processing

