# MERLIN Implementation for ECG Seizure Detection

## Overview

This document outlines the implementation plan for applying the **MERLIN anomaly detection algorithm** to ECG-based seizure detection using the SeizeIT2 dataset. MERLIN is a parameter-free discord discovery algorithm that identifies the most anomalous subsequences in time series data.

## Algorithm Understanding

### MERLIN Core Concepts

**MERLIN** (Parameter-Free Discovery of Arbitrary Length Anomalies) is a distance-based anomaly detector that:

- Uses sliding window approach to find **discords** (anomalous subsequences)
- Searches for subsequences with **no similar neighbors** within a distance threshold
- Automatically adjusts parameters using **DRAG (Distance Range Adjustment using Gaussian)**
- Z-normalizes each subsequence for comparison
- Returns binary anomaly labels for each time point

### Key Parameters

- `min_length`: Minimum subsequence length (≥4, default=5)
- `max_length`: Maximum subsequence length (≤len(series)/2, default=50)
- `max_iterations`: Maximum DRAG iterations per length (default=500)

### Algorithm Steps

1. **Validation**: Check input parameters and detect constant regions
2. **Length Iteration**: Test all subsequence lengths from `min_length` to `max_length`
3. **Distance Threshold**: Start with `r = 2√L` for length L
4. **Discord Search**: Use DRAG to find subsequences without neighbors within distance r
5. **Threshold Adaptation**: Reduce r iteratively if no discord found
6. **Anomaly Marking**: Mark positions of discovered discords as anomalies

## ECG Data Characteristics

### SeizeIT2 Dataset Properties

- **Sampling Rate**: 256 Hz
- **Signal Type**: Single-channel ECG
- **Seizure Events**: Annotated focal seizures with start/end times
- **Data Length**: Hours of continuous recording per patient
- **Recording Type**: Discontinuous monitoring
- **Challenges**: Noise, artifacts, heart rate variability

### ECG Signal Processing Considerations

1. **Preprocessing**:
   - Bandpass filtering (0.5-40 Hz) to remove baseline drift and noise
   - Artifact detection and removal
   - Signal quality assessment

2. **Feature Engineering**:
   - Raw ECG signal (256 Hz) only

3. **Window Size Selection**:
   - For raw ECG: 256-2560 samples (1-10 seconds at 256Hz)
   - Must balance: seizure duration vs. computational efficiency

## Implementation Plan

### Phase 1: Single-Run Prototype

**Objective**: Implement and test MERLIN on one ECG recording with known seizures

**Target Patient**: First patient from seizure_patients list with good data quality

**Implementation Steps**:

1. **Data Preparation**:
   ```python
   # Load ECG data for selected patient/run
   patient_id, run_id = seizure_patients[0][:2]
   data = load_ecg_data(patient_id, run_id)
   annotations = load_seizure_annotations(patient_id, run_id)
   
   # Preprocess ECG signal
   ecg_filtered = preprocess_ecg(data.ecg, fs=256)
   ```

2. **MERLIN Setup**:
   ```python
   from aeon.anomaly_detection.distance_based import MERLIN
   
   # Configure for ECG data
   detector = MERLIN(
       min_length=256,    # 1 second at 256Hz
       max_length=2560,   # 10 seconds at 256Hz
       max_iterations=500
   )
   ```

3. **Anomaly Detection**:
   ```python
   # Apply MERLIN to ECG signal
   anomalies = detector.fit_predict(ecg_filtered)
   
   # Convert to time-based results
   anomaly_times = convert_to_timestamps(anomalies, fs=256)
   ```

4. **Evaluation**:
   ```python
   # Compare with ground truth seizures
   metrics = evaluate_detection(anomaly_times, seizure_annotations)
   ```

### Phase 2: Parameter Optimization

**Raw ECG Signal Analysis** (256 Hz):
   - Window sizes: 1-10 seconds (256-2560 samples)
   - Pros: Full temporal resolution, direct seizure signature detection
   - Cons: High computational cost, noise sensitivity
   - Focus: Optimize window lengths for seizure detection

### Phase 3: Multi-Patient Validation

**Window Size Optimization**:
- Test different min_length/max_length combinations
- Balance between seizure detection and false positives
- Consider typical seizure durations (10-120 seconds)

**Multi-Patient Testing**:
- Apply optimized parameters to multiple patients
- Validate generalization across different patients
- Document patient-specific variations

## Experimental Design

### Single-Run Experiment

**Patient Selection**:
```python
# Select patient with multiple seizures and good signal quality
target_patient = select_optimal_patient(
    seizure_patients,
    min_seizures=2,
    min_duration_hours=2
)
```

**Data Segmentation**:
```python
# Use first 2 hours of recording
experiment_duration = 2 * 3600  # 2 hours in seconds
ecg_segment = ecg_data[:experiment_duration * fs]
seizure_events = filter_seizures_in_timeframe(annotations, experiment_duration)
```

**MERLIN Configuration Matrix**:
```python
configs = [
    # Raw ECG configurations only
    {"min_length": 256, "max_length": 1280},   # 1-5 sec windows
    {"min_length": 512, "max_length": 2560},   # 2-10 sec windows
    {"min_length": 256, "max_length": 2560},   # 1-10 sec windows
    {"min_length": 768, "max_length": 1536},   # 3-6 sec windows
]
```

**Evaluation Metrics**:
- **Sensitivity**: True Positive Rate for seizure detection
- **False Alarm Rate**: False positives per hour


## Implementation Structure

### File Organization

```
models/
├── merlin/
│   ├── __init__.py
│   ├── merlin_detector.py      # Main MERLIN implementation
│   ├── preprocessing.py        # ECG preprocessing functions
│   ├── evaluation.py          # Metrics and evaluation
│   └── visualization.py       # Result plotting
│
scripts/
├── run_merlin_experiment.py   # Single-run experiment script
├── evaluate_merlin.py         # Evaluation script
└── optimize_parameters.py     # Parameter tuning
│
notebooks/
└── merlin_analysis.ipynb      # Interactive analysis notebook
```

### Key Functions

```python
class ECGMerlinDetector:
    def __init__(self, min_length=256, max_length=2560):
        self.detector = MERLIN(min_length, max_length)
    
    def preprocess_ecg(self, ecg_data, fs=256):
        """Preprocess raw ECG signal"""
        pass
    
    def detect_anomalies(self, signal):
        """Apply MERLIN detection"""
        pass
    
    def evaluate_results(self, anomalies, seizure_annotations):
        """Calculate detection metrics"""
        pass
```

## Success Criteria

### Phase 1 Success (Single-Run)

- [ ] Successfully apply MERLIN to ECG data without errors
- [ ] Detect at least 1 seizure in test recording (Sensitivity > 0)
- [ ] False alarm rate < 10 per hour
- [ ] Generate clear visualization of results
- [ ] Document computational performance

### Phase 2 Success (Parameter Optimization)

- [ ] Identify optimal window size ranges for raw ECG
- [ ] Achieve Sensitivity > 0.7 for best configuration
- [ ] False alarm rate < 5 per hour for best configuration
- [ ] Document computational performance trade-offs

### Phase 3 Success (Multi-Patient Validation)

- [ ] Establish optimal parameters for ECG seizure detection
- [ ] Validate performance across multiple patients
- [ ] Create automated pipeline for new patients
- [ ] Document clinical interpretation guidelines

## Next Steps

1. **Implement Single-Run Experiment**: Start with one patient recording
2. **Create Evaluation Framework**: Standardized metrics and visualization
3. **ECG Preprocessing Pipeline**: Robust filtering and artifact removal
4. **Parameter Grid Search**: Systematic optimization of window sizes
5. **Multi-Patient Validation**: Extend to multiple patients for generalization

## References

- Nakamura, M. et al. "MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives," IEEE ICDM 2020
- SeizeIT2 Dataset: ECG-based seizure detection dataset
- Aeon Library: Python toolkit for time series analysis