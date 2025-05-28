# MERLIN with Heart Rate (HR) Data - Notebook Implementation Specification

## Overview

This specification outlines a notebook-based implementation of MERLIN anomaly detection using Heart Rate (HR) time series extracted from ECG signals. The approach focuses on simple HR extraction and direct anomaly detection without additional features like HRV or CSI.

## Motivation

Based on analysis of `notebooks/merlin_experiment.ipynb`, the current raw ECG approach faces challenges:
- **High computational complexity**: Raw ECG at 256 Hz requires large window sizes (256-2560 samples)
- **Memory constraints**: MERLIN failed due to minimum window size requirements  
- **Feature relevance**: Heart rate patterns may be more directly related to seizure events

## Heart Rate Extraction Pipeline

### Simple HR Extraction

```python
def extract_heart_rate_from_ecg(ecg_signal, fs=256, target_fs=4):
    """
    Extract heart rate time series from ECG signal
    
    Args:
        ecg_signal: Preprocessed ECG signal (256 Hz)
        fs: ECG sampling frequency (256 Hz) 
        target_fs: Target HR sampling frequency (4 Hz)
    
    Returns:
        hr_signal: Heart rate time series (BPM)
        hr_timestamps: Corresponding timestamps
    """
    # Step 1: R-peak detection using scipy.signal.find_peaks
    # Step 2: Calculate RR intervals 
    # Step 3: Convert to instantaneous heart rate (BPM)
    # Step 4: Interpolate to regular 4 Hz grid
    
    return hr_signal, hr_timestamps
```

## MERLIN Configuration for HR Data

### HR-Optimized Window Sizes

```python
# HR-optimized MERLIN configurations (4 Hz sampling)
MERLIN_HR_CONFIGS = [
    {"name": "5-30sec", "min_length": 20, "max_length": 120},   # 5-30 seconds
    #{"name": "10-60sec", "min_length": 40, "max_length": 240},  # 10-60 seconds  
    #{"name": "30-120sec", "min_length": 120, "max_length": 480} # 30-120 seconds
]
```

## Notebook Implementation Structure

### Single Notebook: `notebooks/merlin_hr_experiment.ipynb`

```markdown
# MERLIN Heart Rate Seizure Detection Experiment

## 1. Setup and Imports
- Import required libraries (aeon, scipy, numpy, matplotlib)
- Load SeizeIT2 data classes

## 2. Heart Rate Extraction Functions  
- implement_hr_extraction()
- plot_hr_vs_ecg_comparison()

## 3. Single Patient/Run Testing
- Load one patient with known seizures (e.g., sub-001 run-03)
- Extract HR time series
- Apply MERLIN with one window configurations
- Evaluate against ground truth

## 4. Multi-Patient Testing
- Loop through 5-10 patients with seizures
- Compare performance across patients
- Statistical analysis of results

## 5. Results Summary and Visualization
- Clinical interpretation of results
```

## Experimental Design

### Phase 1: Single Patient Implementation 
1. **Setup**: Create `notebooks/merlin_hr_experiment.ipynb`
2. **HR Extraction**: Implement simple R-peak detection and HR calculation
3. **MERLIN Application**: Test HR-MERLIN on one patient/run

### Phase 2: Multi-Patient Validation (1 week)  
1. **Patient Selection**: Identify 5-10 patients with seizures from existing data
2. **Batch Processing**: Process multiple patients automatically
3. **Statistical Analysis**: Compare performance across patients
4. **Parameter Optimization**: Fine-tune MERLIN window sizes for HR data

## Expected Advantages

### Computational Efficiency
- **64x data reduction**: 256 Hz → 4 Hz  
- **Smaller memory footprint**: Enables longer analysis windows
- **Faster processing**: Real-time capable


## Implementation Details

### Key Functions to Implement

```python
def detect_r_peaks_simple(ecg_signal, fs=256):
    """Simple R-peak detection using scipy.signal.find_peaks"""
    
def calculate_heart_rate(r_peaks, fs=256, target_fs=4):  
    """Convert R-peaks to regular HR time series"""
    
def run_merlin_hr_experiment(hr_signal, config):
    """Run MERLIN on HR data with specific configuration"""
    
def evaluate_hr_detection(anomaly_regions, true_seizures):
    """Evaluate HR-based anomaly detection against seizures"""
```

### Testing Strategy

**Single Patient Test (Week 1):**
- Patient: sub-001 run-03
- Duration: 2 hours  
- HR sampling: 4 Hz (28,800 samples vs 1,843,200 ECG samples)
- MERLIN windows: 20-480 samples (5 seconds - 2 minutes)

**Multi-Patient Test:**
- Patients: 5-10 from existing seizure_patients list
- Metrics: Sensitivity, FAR, processing time

## Performance Metrics

### Primary Metrics
- **Sensitivity**: Proportion of seizures correctly detected
- **False Alarm Rate (FAR)**: False alarms per hour  

