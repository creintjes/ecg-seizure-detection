# Cluster-Based False Positive Reduction

This module implements a machine learning approach to reduce false positives in Madrid seizure detection while maintaining high sensitivity. The system uses cluster-level features, ECG signal characteristics, and contextual information to distinguish between true seizure hits and false positive clusters.

## Overview

The pipeline consists of 6 main steps:
1. **Cluster Analysis** - Load and analyze Madrid clustering results
2. **Feature Extraction** - Extract comprehensive features from clusters and ECG data
3. **Model Training** - Train multiple ML models with patient-stratified cross-validation
4. **Clinical Evaluation** - Assess clinical performance metrics
5. **Visualization** - Generate comprehensive analysis plots
6. **Model Deployment** - Package the best model for integration

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn xgboost
```

### Basic Usage

```bash
# Run with default configuration
python main.py --quick-run

# Run with custom paths
python main.py --cluster-dir "path/to/clusters" --madrid-dir "path/to/madrid"

# Run with configuration file
python main.py --config config.json
```

### Configuration

Create a `config.json` file:

```json
{
  "cluster_results_dir": "madrid_results copy/tolerance_adjusted_smart_clustered",
  "madrid_results_dir": "madrid_results copy/madrid_dir_400_examples_tolerance", 
  "raw_data_dir": "path/to/raw/ecg/data",
  "output_dir": "cluster_reduction/results",
  "sampling_rate": 8,
  "test_size": 0.2,
  "balance_strategy": "smote"
}
```

## Architecture

### Module Structure

```
cluster_reduction/
├── main.py                 # Main execution script
├── cluster_analyzer.py     # Cluster data loading and analysis
├── feature_extractor.py    # Feature extraction (cluster + signal + contextual)
├── ml_pipeline.py          # ML training with patient-stratified CV
├── visualization.py        # Comprehensive visualization engine
├── requirements.txt        # Dependencies
└── README.md              # This file
```

### Features Extracted

#### Cluster-Level Features (25 features)
- **Composition**: cluster_size, anomaly_density, temporal_spread
- **Madrid Algorithm**: avg/max/min anomaly_score, m_value statistics, rank distribution
- **Temporal Patterns**: cluster_duration, inter_anomaly_intervals, temporal_clustering_coeff

#### Signal-Level Features (27 features)
- **Heart Rate**: hr_mean, hr_std, hr_trend, hr_acceleration
- **HRV Time Domain**: rr_rmssd, rr_pnn50, rr_sdnn, rr_triangular_index
- **HRV Frequency Domain**: hrv_lf_power, hrv_hf_power, hrv_lf_hf_ratio
- **Signal Quality**: signal_to_noise_ratio, artifact_probability, r_peak_confidence
- **Autonomic**: cardiac_sympathetic_index, sample_entropy, deceleration_capacity

#### Contextual Features (11 features)
- **Patient**: seizure_count, avg_seizure_duration, baseline_hr
- **Recording**: recording_duration, time_of_day (sin/cos), seizure_proximity
- **Temporal**: time_since_recording_start, relative_position_in_recording

**Total: 63 features** across all categories

### Machine Learning Pipeline

#### Models Supported
- **Random Forest** - Interpretable tree-based ensemble
- **XGBoost** - High-performance gradient boosting
- **Logistic Regression** - Linear baseline with regularization
- **SVM** - Support vector machine with RBF/linear kernels

#### Cross-Validation Strategy
- **Patient-stratified GroupKFold** to prevent data leakage
- **5-fold CV** with patients as groups
- **Multiple metrics**: ROC-AUC, Precision, Recall, F1, Accuracy

#### Data Balancing
- **SMOTE** - Synthetic minority oversampling (default)
- **Random Undersampling** - Reduce majority class
- **Cost-sensitive learning** - Built into model parameters

## Usage Examples

### Example 1: Basic Pipeline Run

```python
from main import ClusterReductionPipeline, create_default_config

# Create configuration
config = create_default_config()
config['cluster_results_dir'] = 'your/cluster/path'
config['madrid_results_dir'] = 'your/madrid/path'

# Run pipeline
pipeline = ClusterReductionPipeline(config)
results = pipeline.run_complete_pipeline()
```

### Example 2: Feature Extraction Only

```python
from cluster_analyzer import ClusterAnalyzer
from feature_extractor import MasterFeatureExtractor

# Load clusters
analyzer = ClusterAnalyzer(cluster_dir, madrid_dir)
results = analyzer.load_and_process_data()

# Extract features
extractor = MasterFeatureExtractor(raw_data_dir='/path/to/ecg')
features = extractor.extract_features_batch(
    list(results['clusters'].values()),
    results['madrid_results']
)
```

### Example 3: Model Training Only

```python
from ml_pipeline import MLPipeline

# Initialize pipeline
ml_pipeline = MLPipeline(random_state=42)

# Train models
results = ml_pipeline.train_and_evaluate_all_models(
    feature_vectors=features,
    test_size=0.2,
    balance_strategy='smote'
)

# Get best model
best_model = ml_pipeline.best_model
print(f"Best: {best_model.model_name}, F1: {best_model.test_scores['f1']:.3f}")
```

### Example 4: Visualization Generation

```python
from visualization import VisualizationEngine

# Create visualizations
viz = VisualizationEngine(output_dir='plots/')
plots = viz.generate_comprehensive_report(
    feature_vectors=features,
    ml_results=ml_results,
    clinical_metrics=clinical_metrics
)
```

## Output Structure

After running the pipeline, the output directory contains:

```
results/
├── cluster_analysis_report.txt          # Cluster statistics
├── extracted_features.csv               # Feature matrix
├── models/                              # Trained models
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── ...
├── model_performance_report.txt         # ML performance
├── clinical_metrics.json               # Clinical evaluation
├── visualizations/                     # Analysis plots
│   ├── feature_distributions.png
│   ├── model_comparison.png
│   ├── feature_importance_*.png
│   ├── clinical_performance_dashboard.png
│   └── ...
├── deployment/                         # Deployment package
│   ├── cluster_classifier.pkl
│   ├── deployment_config.json
│   └── integrate_classifier.py
└── pipeline_summary.json              # Complete summary
```

## Performance Targets

### Expected Improvements
- **False Positive Reduction**: 50-80% decrease in FP rate
- **Sensitivity Maintenance**: ≥70% seizure detection rate
- **Precision Improvement**: 30-60% increase in precision
- **Clinical Utility**: <2 false alarms per hour

### Baseline vs Target Performance

| Metric | Baseline | Target | 
|--------|----------|--------|
| Sensitivity | 73.3% | ≥70% |
| False Alarm Rate | 74.0% | <40% |
| Precision | 26.0% | >50% |
| False Positives | 1000 | <500 |

## Integration with Madrid Pipeline

The trained classifier can be integrated into the existing Madrid pipeline:

```python
# Load trained classifier
from deployment.integrate_classifier import load_cluster_classifier, filter_false_positives

# Filter clusters after Madrid processing
model = load_cluster_classifier()
filtered_clusters = filter_false_positives(madrid_clusters, threshold=0.6)

# Result: Reduced false positives while maintaining seizure detection
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Path Errors**
   - Ensure cluster and Madrid result directories exist
   - Check file permissions
   - Use absolute paths if relative paths fail

3. **Memory Issues**
   - Reduce feature extraction batch size
   - Use 'undersample' instead of 'smote' for balancing
   - Enable garbage collection in long runs

4. **Model Training Failures**
   - Check for NaN values in features
   - Ensure sufficient data for cross-validation
   - Try simpler models (LogisticRegression) first

### Debug Mode

```bash
# Enable verbose logging
python main.py --verbose --config config.json

# Check logs
tail -f cluster_reduction.log
```

## Advanced Configuration

### Custom Feature Selection

```python
# Modify feature extractors
extractor = MasterFeatureExtractor()
extractor.cluster_extractor.feature_names = ['cluster_size', 'avg_anomaly_score']  # Subset
```

### Hyperparameter Tuning

```python
# Custom parameter grids
ml_pipeline = MLPipeline()
ml_pipeline.models['random_forest']['params'] = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, None]
}
```

### Custom Evaluation Metrics

```python
# Add custom scoring
from sklearn.metrics import make_scorer
from ml_pipeline import MLPipeline

def custom_clinical_score(y_true, y_pred):
    # Custom metric emphasizing sensitivity
    pass

ml_pipeline = MLPipeline()
# Add to scoring in evaluate_model_cv()
```

## Contributing

To extend the pipeline:

1. **Add new features**: Extend feature extractors in `feature_extractor.py`
2. **Add new models**: Extend model dictionary in `ml_pipeline.py`
3. **Add new visualizations**: Add methods to `visualization.py`
4. **Improve evaluation**: Extend `ModelEvaluator` class

## Citation

If you use this code in your research, please cite:

```
@software{cluster_reduction_2025,
  title={Cluster-Based False Positive Reduction for Madrid Seizure Detection},
  author={ECG Seizure Detection Team},
  year={2025},
  url={https://github.com/your-repo/cluster-reduction}
}
```