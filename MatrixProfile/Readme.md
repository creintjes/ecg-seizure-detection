# ECG Seizure Detection using Matrix Profile 

## Description  
This project implements an **ECG-based anomaly detection pipeline using Matrix Profiles** to automatically detect epileptic seizures.  

- Matrix Profiles (MPs) are computed to highlight unusual patterns in ECG recordings.  
- A **grid search** evaluates different hyperparameter configurations to optimize **sensitivity** and **false alarm rates**.  
- The main results (as used in the related paper) were generated with `mp_generate_results_relative.py`.  


## 1. Project Structure  

```plaintext

├── configs/                      # Configuration files (validation/test splits)
├── legacy_code/                  # Old or experimental code
├── mv_mp/                        # Multivariate matrix profile storage
├── notebooks/                    # Jupyter notebooks (e.g., experiments, visualization)
├── results/                      # Result files (Excel files - output of hp tuning)
├── rf_train_data/                # Random Forest training samples (Not in this repo, but this folder will be created if enabled. The files were to large to push and store in git.)
│
├── create_save_mps.py            # Script to compute & save uni- and multivariate MPs
├── create_save_multivariate_mps.py # Script to compute & save multivariate MPs and find epileptic seizures based on recordings. NOTE: The execution can take up to one day for one recording.
├── create_val_test_filelists.py  # Script to create the excel file with the val/test split (See paper for split justification).
├── data_helpers.py               # Helper functions for loading preprocessed ECG data
├── load_mp.py                    # Utility for loading stored matrix profiles
├── matrix_profile.py             # Core MatrixProfile class with all relevant functions (ECG features, anomaly detection, post processing)
├── mp_generate_results.py        # Grid search (absolute anomaly count) used for first and second paper draft
├── mp_generate_results_relative.py # Grid search (relative anomaly ratio, MAIN script) used for final paper
├── rf_postprocessing.py          # Train/evaluate Random Forest classifier from RF samples - post processing of the MPs.
└── requirements.txt              # Python dependencies
```
## 2. Installation  

### 2.1 Requirements  
- **Python 3.10+**  
- Dependencies from `requirements.txt`, including:  
  - `numpy`, `pandas`, `scikit-learn`  
  - `stumpy` (matrix profile algorithms)  
  - `neurokit2` (R-peak detection & HRV features) 

### 2.2 Setup  

```bash
# Clone the repository
git clone https://github.com/creintjes/ecg-seizure-detection.git
cd ecg-seizure-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Switch to the MatrixProfile folder
cd MatrixProfile 
```


## 3. Usage  


### 3.1 Reproduce Results – Run Main Experiment (Relative Anomaly Ratio)  

On a server, it is **highly recommended to use a tmux session** so the experiment keeps running even if your SSH connection drops.  

Create and attach to a tmux session:  
```bash
tmux new -s mp_experiment
```

Then run the experiment inside tmux:  
```bash
python3 mp_generate_results_relative.py
```

This will:  
- Run grid search over parameters (see below).  
- Save results in Excel files, e.g. `results/hp_tuning_mp_results_resp_relative.xlsx`.  

To detach from the session (leaving the experiment running in the background), press:  
```
Ctrl+b  d
```

To reattach later:  
```bash
tmux attach -t mp_experiment
```



### 3.2 Used Validation/Test Splits  
- Validation: `sub-001` … `sub-096`  
- Test: `sub-097` … `sub-125`  


### 3.4 Key Parameters in `mp_generate_results_relative.py`  
- `anomaly_ratio`: relative anomaly fraction (`0.01, 0.02, 0.04, 0.06`)  
- `max_gap_annos_in_sec`: maximum allowed gap between anomalies (`0.2–30s`)  
- `n_cons`: minimum consecutive anomalies to count (`1–35`)  
- `pre_thresh_sec`, `post_thresh_sec`: tolerance window before/after seizures "Seizure Detection Window" - read the paper for more information 
- `batch_size_load`: number of files per processing batch  

### 3.5 Optional: Random Forest Postprocessing  
If `append_rf_samples_to_xlsx()` is enabled in `mp_generate_results_relative.py`, then this will produce samples which can be used to train a rf. This was only demonstrated on one recording. Training took >1d for this file. Therefore, it was not demonstrated on the whole data set. Code may has to be improved.:  
```bash
python rf_postprocessing.py
```
This trains a **Random Forest classifier** on extracted MP features.  



## 4. Evaluation Metrics  

Used evaluation metrics include:  
- **sensitivity** – proportion of correctly detected seizures  
- **false_alarms_per_hour** – number of false detections per hour  
- **resp_sensitivity** – sensitivity for “responder” patients (≥66% detection)  


## 5. Configuration  

- Paths for **preprocessed data** (`DIR_preprocessed`) and **MP storage** (`MPs_path`) are set inside the scripts and may need adjustments if the code is used outside the DS03.  
- `requirements.txt` contains all dependencies.  

