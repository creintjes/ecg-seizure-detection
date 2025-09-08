# Madrid - MADRID Algorithm Analysis

This directory contains analysis scripts and tools for the MADRID (Multi-Length Anomaly Detection with Irregular Discords) algorithm, specifically adapted for ECG seizure detection.

## Overview

MADRID is a multi-length anomaly detection algorithm that can identify anomalous subsequences of varying durations simultaneously. This makes it particularly suitable for seizure detection, where seizure events can have different durations and characteristics.

## Files

### `madrid_seizure_analysis.py`
Main analysis script for seizure-focused MADRID analysis. This script mirrors the functionality of `Merlin/merlin_seizure_analysis.py` but uses the MADRID algorithm instead.

**Key Features:**
- Load seizure-only preprocessed data with phase labels
- Analyze seizure phases separately (pre-seizure, ictal, post-seizure)
- Multiple MADRID configurations optimized for different seizure patterns
- GPU acceleration support for large datasets
- Detailed performance analysis with seizure-specific metrics
- Continuous data analysis with multi-length anomaly detection

## Usage

### Basic Usage
```bash
# Analyze a single seizure file
python madrid_seizure_analysis.py --data-path path/to/seizure_file.pkl

# Analyze all seizure files in a directory
python madrid_seizure_analysis.py --data-dir data/seizure_segments/

# Filter by specific subject
python madrid_seizure_analysis.py --data-dir data/seizure_segments/ --subject sub-001

# Filter by specific seizure index
python madrid_seizure_analysis.py --data-dir data/seizure_segments/ --seizure-idx 0
```

### Advanced Options
```bash
# Use specific configuration
python madrid_seizure_analysis.py --data-path file.pkl --single-config medium_term

# Disable GPU acceleration
python madrid_seizure_analysis.py --data-path file.pkl --no-gpu

# Override sampling rate detection
python madrid_seizure_analysis.py --data-path file.pkl --override-fs 250

# Force downsample to lower rate
python madrid_seizure_analysis.py --data-path file.pkl --force-downsample 125

# Include windowed analysis
python madrid_seizure_analysis.py --data-path file.pkl --windowed-analysis --window-size 30.0

# Plot detailed results
python madrid_seizure_analysis.py --data-path file.pkl --plot-results

# Load custom configurations
python madrid_seizure_analysis.py --data-path file.pkl --config-file custom_config.json

# Generate configuration template
python madrid_seizure_analysis.py --save-template
```

## Sampling Rate Support

The Madrid analysis script is designed to work with **different sampling rates** automatically:

### Supported Sampling Rates
- **≤ 50 Hz**: Very low rate (warnings issued, focus on long-term analysis)
- **50-125 Hz**: Standard ECG rates (optimal for seizure detection)
- **125-250 Hz**: High-quality rates (excellent for detailed analysis)
- **250-500 Hz**: Very high rates (adaptive stepping, GPU recommended)
- **> 500 Hz**: Extremely high rates (downsampling recommended)

### Automatic Adaptations
- **Adaptive Step Sizes**: Automatically adjusts analysis steps based on sampling rate
- **Performance Optimization**: Higher rates use larger steps to reduce computation
- **Memory Management**: Intelligent handling of large datasets
- **Quality Warnings**: Alerts for suboptimal sampling rates

### Manual Overrides
```bash
# Override detected sampling rate
python madrid_seizure_analysis.py --data-path file.pkl --override-fs 250

# Downsample high-rate data
python madrid_seizure_analysis.py --data-path file.pkl --force-downsample 125

# Combine for optimal processing
python madrid_seizure_analysis.py \
    --data-path high_rate_file.pkl \
    --force-downsample 250 \
    --single-config medium_term
```

### Sampling Rate Detection
The script automatically detects sampling rates from:
1. `data['sampling_rate']` field
2. `data['metadata']['sampling_rate']` field  
3. Estimation from data length and duration
4. Default fallback (125 Hz)

## MADRID Configurations

The script includes several pre-optimized configurations for different seizure detection scenarios:

### `ultra_short` (0.5-5s)
- **Purpose**: Detect very short anomalies
- **Target**: Cardiac arrhythmias, brief artifacts
- **Parameters**: min=0.5s, max=5s, step=0.5s

### `short_term` (2-15s)
- **Purpose**: Detect short-term anomalies
- **Target**: Brief seizure patterns, autonomic changes
- **Parameters**: min=2s, max=15s, step=1s

### `medium_term` (10-60s)
- **Purpose**: Detect medium-term anomalies
- **Target**: Seizure onset/offset, autonomic responses
- **Parameters**: min=10s, max=60s, step=5s

### `long_term` (30-180s)
- **Purpose**: Detect long-term anomalies
- **Target**: Prolonged seizures, post-ictal changes
- **Parameters**: min=30s, max=180s, step=10s

### `extended_term` (2-10min)
- **Purpose**: Detect extended anomalies
- **Target**: Long seizures, recovery patterns
- **Parameters**: min=120s, max=600s, step=30s

## Custom Configurations

You can create custom MADRID configurations using a JSON file:

```json
{
  "configs": [
    {
      "name": "custom_short",
      "min_length_seconds": 1.0,
      "max_length_seconds": 8.0,
      "step_size_seconds": 1.0,
      "description": "Custom short anomaly detection"
    }
  ]
}
```

## Output Analysis

The script provides comprehensive analysis including:

### Continuous Analysis
- **Configuration Performance**: Detection time, anomaly rates, regions found
- **Phase-wise Analysis**: Anomaly rates for each seizure phase
- **Seizure Discrimination**: Ratios between ictal and normal phases
- **Performance Metrics**: Real-time capability, GPU speedup

### Windowed Analysis (Optional)
- **Phase-specific Windows**: Analysis of individual seizure phases
- **Statistical Summary**: Mean and standard deviation of anomaly rates
- **Window Success Rate**: Percentage of successfully processed windows

### Visualization (Optional)
- **ECG Signal**: Original signal with phase labels
- **Anomaly Detection**: Binary anomaly detection overlay
- **Discord Scores**: Scores across different subsequence lengths
- **Phase Comparison**: Bar chart of anomaly rates by phase

## GPU Acceleration

MADRID supports GPU acceleration via CuPy for significant performance improvements:

```bash
# Install GPU support (CUDA 12.x)
pip install cupy-cuda12x

# Enable GPU (default)
python madrid_seizure_analysis.py --data-path file.pkl

# Disable GPU
python madrid_seizure_analysis.py --data-path file.pkl --no-gpu
```

Expected speedup: 10-50x depending on data size and GPU capabilities.

## Integration with Project

This analysis script integrates with the broader ECG seizure detection project:

1. **Input**: Uses seizure-only preprocessed data from `preprocess_seizure_only.py`
2. **Processing**: Applies MADRID algorithm with seizure-optimized parameters
3. **Output**: Provides detailed analysis compatible with project evaluation metrics
4. **Comparison**: Results can be compared with Merlin, TimeVQVAE-AD, and Matrix Profile

## Dependencies

- Main MADRID implementation: `../models/madrid.py`
- Standard scientific Python stack: numpy, scipy, matplotlib, pandas
- Optional GPU support: cupy
- Data handling: pickle for preprocessed seizure data

## Example Workflow

```bash
# 1. Generate configuration template
python madrid_seizure_analysis.py --save-template

# 2. Analyze seizure data with plotting
python madrid_seizure_analysis.py \
    --data-dir ../data/seizure_segments/ \
    --subject sub-001 \
    --max-files 5 \
    --plot-results \
    --windowed-analysis \
    --verbose

# 3. Compare with custom configuration
python madrid_seizure_analysis.py \
    --data-path ../data/seizure_segments/sub-001_run-03_seizure_00_preprocessed.pkl \
    --config-file madrid_seizure_config_template.json \
    --single-config seizure_medium
```

## Performance Tips

1. **Start with small datasets** to test configurations
2. **Use GPU acceleration** for large datasets (>100,000 samples)
3. **Adjust configurations** based on expected seizure characteristics
4. **Monitor memory usage** for very long recordings
5. **Use windowed analysis** for detailed phase investigation

## Troubleshooting

### Common Issues
- **Import Error**: Ensure `../models/madrid.py` is available
- **GPU Issues**: Install appropriate CuPy version or use `--no-gpu`
- **Memory Error**: Reduce data size or use CPU mode
- **Configuration Error**: Check that min_length < max_length and sufficient data

### Performance Optimization
- Use appropriate step sizes (larger steps = faster processing)
- Consider downsampling very high-frequency data
- Use shorter max_length for faster processing
- Enable GPU for datasets > 50,000 samples

## Comparison with Merlin

| Aspect | MADRID | Merlin |
|--------|--------|---------|
| **Multi-length** | ✓ Built-in | Single length |
| **GPU Support** | ✓ CuPy | CPU only |
| **Online Processing** | ✓ Yes | Batch |
| **Pruning** | ✓ DAMP | No |
| **Configuration** | Length ranges | Single length |
| **Memory Usage** | Moderate | Low |
| **Speed** | Fast (GPU) | Fast (CPU) |

Both algorithms are effective for seizure detection but offer different advantages depending on the specific use case and computational resources.