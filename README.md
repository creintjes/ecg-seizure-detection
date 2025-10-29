# ECG-Based Seizure Detection

This repository contains the code and resources for our research **"ECG-Based Detection of Epileptic Seizures in Real-World Wearable Settings: Insights from the SeizeIT2 Dataset"**.

## 🧠 Project Overview

Epileptic seizures can cause serious medical and psychosocial consequences. While EEG is the clinical gold standard for seizure detection, wearable ECG devices offer a more practical alternative for daily use.

In this project, we evaluate the performance of three modern anomaly detection algorithms on ECG data for the purpose of seizure detection:

- **Madrid** - See [Madrid/README.md](Madrid/README.md)
- **TimeVQVAE-AD** - See [TimeVQVAE-AD/README.md](TimeVQVAE-AD/README.md)
- **Matrix Profile** - See [MatrixProfile/Readme.md](MatrixProfile/Readme.md)

We aim to identify approaches that achieve a favorable balance between **sensitivity** and **false alarm rate** (FAR), using the [SeizeIT2 dataset](https://doi.org/10.48550/arXiv.2502.01224).


## 📁 Project Structure

```
ecg-seizure-detection/
├── MatrixProfile/          # Matrix Profile experiments and implementation
├── Madrid/                 # Madrid experiments and implementation
├── TimeVQVAE-AD/          # TimeVQVAE-AD experiments and implementation
├── Information/           # Documentation and dataset information
├── find_missing_ecg/      # Scripts for ECG data quality analysis
├── find_responder/        # Responder/non-responder analysis
├── preprocessing/         # Data preprocessing scripts
├── utils/                 # Shared utility functions and metrics
├── config.py             # Global configuration
├── requirements.txt      # Python dependencies
└── README.md             # This file
```


## ⚙️ Installation
### 1. Clone Repo
```bash
git clone https://github.com/creintjes/ecg-seizure-detection.git
cd ecg-seizure-detection
```
### 2. Install Requirements

```bash
pip install -r requirements.txt
```
### 3. Download Data:

```bash
nohup ./filtered_download_script.sh > filtered_download.log 2>&1 &
```
### 4. Preprocess Data:
```bash
python3 preprocess_all_data.py
```

### 5. Analyze ECG Data Quality
Before running experiments, you can analyze ECG signal quality to identify potential issues:

```bash
cd find_missing_ecg

# Analyze data for missing or empty ECG signals
python3 analyze_data.py

# Analyze test patients for signal saturation
python3 analyze_test_patients_saturation.py results/data_analysis.json

# Extract saturated test patients with threshold (e.g., 10%)
python3 extract_saturated_test_patients.py results/data_analysis.json 10

# Analyze seizure loss due to saturation
python3 analyze_seizure_loss_from_saturation.py
```

Results will be saved in the `find_missing_ecg/results/` directory.

### 6. Classify Responders and Non-Responders 
Identify patients who show significant heart rate changes during seizures (responders) vs. those who don't (non-responders):

```bash
cd find_responder

# Classify patients based on HR changes during first seizure
# Uses Elgendi R-peak detection with extended window analysis
python3 find_responder_elgendi_first_seizure_extended.py
```

**Classification criteria:**
- **Responders**: HR change >50 bpm during seizure (100 RR interval window)
- **Non-Responders**: HR change ≤50 bpm during seizure

Results are saved as:
- `responders_and_non_responders_elgendi_first_seizure_extended.txt` - Lists of responder/non-responder patient IDs
- `patient_responder_summary_elgendi_first_seizure_extended.csv` - Detailed metrics per patient

### 7. Run Model Experiments
After preprocessing, follow the steps described in the respective README files of each model to reproduce the experiments:

- [Madrid/README.md](Madrid/README.md)
- [TimeVQVAE-AD/README.md](TimeVQVAE-AD/README.md)
- [MatrixProfile/Readme.md](MatrixProfile/Readme.md)

