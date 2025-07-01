# MADRID - Multi-Length Anomaly Detection

This directory contains the Python implementation of the MADRID (Multi-Length Anomaly Detection with Irregular Discords) algorithm, converted from the original MATLAB implementation with GPU acceleration support.

## Overview

MADRID is a time series anomaly detection algorithm that identifies discords (anomalous subsequences) across multiple lengths simultaneously. It's particularly effective for detecting anomalies of varying durations, making it ideal for ECG-based seizure detection.

## Features

- **Multi-length Detection**: Simultaneously searches for anomalies across different subsequence lengths
- **GPU Acceleration**: Optional CUDA support via CuPy for significant performance improvements
- **Online Processing**: Processes data incrementally with train/test split capability
- **Adaptive Pruning**: Uses DAMP (Discord Aware Matrix Profile) for efficient computation
- **ECG Optimized**: Pre-configured parameters for seizure detection scenarios

## Installation

### Basic Installation
```bash
pip install -r requirements_madrid.txt
```

### GPU Support (Optional)
For CUDA 11.x:
```bash
pip install cupy-cuda11x>=10.0.0
```

For CUDA 12.x:
```bash
pip install cupy-cuda12x>=12.0.0
```

## Quick Start

### Basic Usage
```python
from models.madrid import MADRID
import numpy as np

# Create sample data
data = np.random.randn(5000)

# Initialize MADRID
madrid = MADRID(use_gpu=True, enable_output=True)

# Run detection
results = madrid.fit(
    T=data,
    min_length=50,     # Minimum anomaly length
    max_length=200,    # Maximum anomaly length
    step_size=25,      # Step between lengths
    train_test_split=2500  # Training portion
)

# Get anomaly information
anomalies = madrid.get_anomaly_scores(threshold_percentile=95)

# Plot results
madrid.plot_results()
```

### ECG Seizure Detection
```python
# Configure for seizure detection (250Hz sampling rate)
madrid_ecg = MADRID(use_gpu=True, enable_output=True)

# Parameters for seizure detection
fs = 250  # Sampling frequency
min_seizure_duration = 0.5  # seconds
max_seizure_duration = 30   # seconds

results = madrid_ecg.fit(
    T=ecg_data,
    min_length=int(min_seizure_duration * fs),  # 125 samples
    max_length=int(max_seizure_duration * fs),  # 7500 samples
    step_size=int(0.5 * fs),                   # 125 samples (0.5s steps)
    train_test_split=len(ecg_data)//3          # Use 1/3 for training
)

# Analyze seizure events
seizure_events = madrid_ecg.get_anomaly_scores(threshold_percentile=90)
```

## API Reference

### MADRID Class

#### Constructor
```python
MADRID(use_gpu=False, enable_output=False)
```

**Parameters:**
- `use_gpu` (bool): Enable GPU acceleration if available
- `enable_output` (bool): Enable verbose output and plotting

#### Methods

##### fit(T, min_length, max_length, step_size, train_test_split, factor=None)
Run MADRID algorithm on time series data.

**Parameters:**
- `T` (np.ndarray): Input time series data
- `min_length` (int): Minimum subsequence length for anomaly detection
- `max_length` (int): Maximum subsequence length for anomaly detection
- `step_size` (int): Step size between different lengths
- `train_test_split` (int): Index separating training and test data
- `factor` (int, optional): Downsampling factor (auto-selected if None)

**Returns:**
- Tuple of (multi_length_discord_table, best_scores, best_locations)

##### get_anomaly_scores(threshold_percentile=95)
Extract anomaly scores and locations above threshold.

**Parameters:**
- `threshold_percentile` (float): Percentile threshold for anomaly detection

**Returns:**
- Dictionary with anomaly information

##### plot_results(figsize=(15, 10))
Plot comprehensive MADRID results including time series, discord scores, and heatmaps.

## Parameter Guidelines

### For ECG Seizure Detection (250Hz)

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| `min_length` | 125 (0.5s) | Minimum seizure duration |
| `max_length` | 7500 (30s) | Maximum seizure duration |
| `step_size` | 125 (0.5s) | Analysis step size |
| `train_test_split` | len(data)//3 | Use 1/3 for training |

### For General Time Series

| Anomaly Type | min_length | max_length | step_size |
|--------------|------------|------------|-----------|
| Short anomalies | 32 | 128 | 16 |
| Long anomalies | 128 | 512 | 32 |
| Multi-scale | 64 | 256 | 32 |

## Performance Optimization

### Execution Modes
MADRID automatically selects the optimal downsampling factor based on data size:

- **Factor 1:1** - Maximum accuracy, slowest
- **Factor 1:2** - Good balance of speed and accuracy
- **Factor 1:4** - Faster processing, good accuracy
- **Factor 1:8** - Fast processing, moderate accuracy
- **Factor 1:16** - Fastest processing, basic accuracy

### GPU Acceleration
GPU acceleration can provide 10-50x speedup depending on:
- Data size
- GPU memory
- CUDA version
- Parameter configuration

Check GPU status:
```python
madrid = MADRID(use_gpu=True)
print(f"Using device: {madrid.device}")
print(f"GPU available: {madrid.use_gpu}")
```

## Examples

See `examples/madrid_usage.py` for comprehensive examples including:

1. **Basic Usage** - Simple anomaly detection
2. **ECG Seizure Detection** - Simulated ECG with seizure events
3. **Parameter Tuning** - Comparing different configurations
4. **Performance Comparison** - CPU vs GPU benchmarking

Run examples:
```bash
python examples/madrid_usage.py
```

## Algorithm Details

### Core Components

1. **MASS_V2**: Efficient similarity search using FFT
2. **DAMP**: Discord Aware Matrix Profile with pruning
3. **Multi-length Processing**: Parallel analysis across length scales
4. **Normalization**: Z-score normalization for comparable scores

### Key Advantages

- **Multi-scale Analysis**: Detects anomalies of varying durations
- **Unsupervised**: No labeled training data required
- **Online Processing**: Suitable for streaming applications
- **Robust**: Built-in validation and error handling
- **Scalable**: GPU acceleration for large datasets

## Troubleshooting

### Common Issues

**Error: "Constant regions detected"**
- Solution: Add noise or increase `min_length`
- Cause: Time series has constant/near-constant segments

**Poor GPU Performance**
- Check CUDA installation and compatibility
- Ensure sufficient GPU memory
- Try smaller data chunks for very large datasets

**Memory Issues**
- Reduce data size or use higher downsampling factor
- Process data in chunks
- Use CPU mode for very large datasets

### Performance Tips

1. **Start with GPU disabled** for initial testing
2. **Use appropriate downsampling** for large datasets
3. **Tune parameters** based on expected anomaly characteristics
4. **Monitor memory usage** for large-scale processing

## Integration with Seizure Detection Pipeline

MADRID integrates seamlessly with the existing ECG seizure detection framework:

```python
# Example integration
from models.madrid import MADRID
from preprocessing import preprocess_ecg
from evaluation import calculate_metrics

# Load and preprocess ECG data
ecg_data = load_seizure_data()
processed_ecg = preprocess_ecg(ecg_data)

# Run MADRID detection
madrid = MADRID(use_gpu=True)
results = madrid.fit(processed_ecg, ...)

# Evaluate against ground truth
metrics = calculate_metrics(results, ground_truth)
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{madrid2023,
  title={MADRID: Multi-Length Anomaly Detection with Irregular Discords},
  journal={IEEE International Conference on Data Mining (ICDM)},
  year={2023}
}
```

## License

This implementation is part of the ECG seizure detection project and follows the same licensing terms as the main project.