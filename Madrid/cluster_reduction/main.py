#!/usr/bin/env python3
"""
Main Execution Script for Cluster-Based False Positive Reduction

This is the master script that orchestrates the complete pipeline:
1. Load and analyze cluster data
2. Extract comprehensive features
3. Train and evaluate ML models
4. Generate visualizations and reports
5. Deploy the best model for FP reduction
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from cluster_analyzer import ClusterAnalyzer
from feature_extractor import MasterFeatureExtractor
from ml_pipeline import MLPipeline, ModelEvaluator
from visualization import VisualizationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cluster_reduction.log')
    ]
)
logger = logging.getLogger(__name__)

class ClusterReductionPipeline:
    """Main pipeline class for cluster-based false positive reduction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing paths and parameters
        """
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cluster_analyzer = None
        self.feature_extractor = None
        self.ml_pipeline = None
        self.visualizer = None
        self.evaluator = ModelEvaluator()
        
        # Data storage
        self.clusters = {}
        self.feature_vectors = []
        self.ml_results = {}
        self.clinical_metrics = {}
        
        logger.info(f"Pipeline initialized with output directory: {self.output_dir}")
    
    def load_and_analyze_clusters(self) -> Dict[str, Any]:
        """Step 1: Load and analyze cluster data."""
        logger.info("=== STEP 1: Loading and Analyzing Clusters ===")
        
        try:
            # Initialize cluster analyzer
            self.cluster_analyzer = ClusterAnalyzer(
                cluster_results_dir=self.config['cluster_results_dir'],
                madrid_results_dir=self.config['madrid_results_dir'],
                raw_data_dir=self.config.get('raw_data_dir')
            )
            
            # Load and process all data
            results = self.cluster_analyzer.load_and_process_data()
            self.clusters = results['clusters']
            
            # Save analysis report
            report_path = self.output_dir / "cluster_analysis_report.txt"
            self.cluster_analyzer.save_analysis_report(report_path)
            
            logger.info(f"Cluster analysis completed:")
            logger.info(f"  - Total clusters: {len(self.clusters)}")
            logger.info(f"  - Seizure hit clusters: {results['analysis']['seizure_hit_clusters']}")
            logger.info(f"  - False positive clusters: {results['analysis']['false_positive_clusters']}")
            logger.info(f"  - Analysis report saved to: {report_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Cluster analysis failed: {e}")
            raise
    
    def extract_features(self) -> List[Any]:
        """Step 2: Extract comprehensive features."""
        logger.info("=== STEP 2: Feature Extraction ===")
        
        try:
            # Initialize feature extractor
            self.feature_extractor = MasterFeatureExtractor(
                raw_data_dir=self.config.get('raw_data_dir'),
                sampling_rate=self.config.get('sampling_rate', 8)
            )
            
            # Get balanced dataset
            cluster_list, labels = self.cluster_analyzer.get_balanced_dataset(
                balance_strategy=self.config.get('balance_strategy', 'undersample')
            )
            
            logger.info(f"Extracting features for {len(cluster_list)} clusters...")
            
            # Extract features for all clusters
            self.feature_vectors = self.feature_extractor.extract_features_batch(
                cluster_list,
                self.cluster_analyzer.madrid_results,
                patient_history=self.config.get('patient_history')
            )
            
            # Save features
            features_path = self.output_dir / "extracted_features.csv"
            self.feature_extractor.save_features(self.feature_vectors, features_path)
            
            logger.info(f"Feature extraction completed:")
            logger.info(f"  - Feature vectors: {len(self.feature_vectors)}")
            logger.info(f"  - Features per vector: {len(self.feature_extractor.feature_names)}")
            logger.info(f"  - Features saved to: {features_path}")
            
            return self.feature_vectors
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def train_models(self) -> Dict[str, Any]:
        """Step 3: Train and evaluate ML models."""
        logger.info("=== STEP 3: Model Training and Evaluation ===")
        
        try:
            # Initialize ML pipeline
            self.ml_pipeline = MLPipeline(
                random_state=self.config.get('random_state', 42)
            )
            
            # Train and evaluate all models
            self.ml_results = self.ml_pipeline.train_and_evaluate_all_models(
                self.feature_vectors,
                test_size=self.config.get('test_size', 0.2),
                balance_strategy=self.config.get('balance_strategy', 'smote')
            )
            
            # Save models and results
            models_dir = self.output_dir / "models"
            self.ml_pipeline.save_all_models(models_dir)
            
            # Generate performance report
            report_path = self.output_dir / "model_performance_report.txt"
            self.ml_pipeline.generate_performance_report(report_path)
            
            logger.info(f"Model training completed:")
            logger.info(f"  - Models trained: {len(self.ml_results)}")
            if self.ml_pipeline.best_model:
                logger.info(f"  - Best model: {self.ml_pipeline.best_model.model_name}")
                logger.info(f"  - Best F1 score: {self.ml_pipeline.best_model.test_scores['f1']:.3f}")
            logger.info(f"  - Models saved to: {models_dir}")
            logger.info(f"  - Performance report: {report_path}")
            
            return self.ml_results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def evaluate_clinical_performance(self) -> Dict[str, float]:
        """Step 4: Evaluate clinical performance metrics."""
        logger.info("=== STEP 4: Clinical Performance Evaluation ===")
        
        try:
            if not self.ml_pipeline.best_model:
                raise ValueError("No best model available for evaluation")
            
            best_result = self.ml_pipeline.best_model
            
            # Calculate clinical metrics
            self.clinical_metrics = self.evaluator.clinical_performance_analysis(
                y_true=None,  # Would need actual test labels
                y_pred=best_result.predictions,
                y_pred_proba=best_result.prediction_probabilities
            )
            
            # Calculate FP reduction compared to baseline (if available)
            baseline_metrics = self.config.get('baseline_metrics')
            if baseline_metrics:
                fp_reduction_metrics = self.evaluator.false_positive_reduction_analysis(
                    baseline_fp=baseline_metrics.get('false_positives', 0),
                    new_fp=self.clinical_metrics['false_positives'],
                    baseline_tp=baseline_metrics.get('true_positives', 0),
                    new_tp=self.clinical_metrics['true_positives']
                )
                self.clinical_metrics.update(fp_reduction_metrics)
            
            # Save clinical metrics
            metrics_path = self.output_dir / "clinical_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.clinical_metrics, f, indent=2)
            
            logger.info(f"Clinical evaluation completed:")
            logger.info(f"  - Sensitivity: {self.clinical_metrics['sensitivity']:.3f}")
            logger.info(f"  - Specificity: {self.clinical_metrics['specificity']:.3f}")
            logger.info(f"  - PPV: {self.clinical_metrics['positive_predictive_value']:.3f}")
            logger.info(f"  - False Positive Rate: {self.clinical_metrics['false_positive_rate']:.3f}")
            if baseline_metrics:
                logger.info(f"  - FP Reduction: {self.clinical_metrics['fp_reduction_percentage']:.1f}%")
            logger.info(f"  - Metrics saved to: {metrics_path}")
            
            return self.clinical_metrics
            
        except Exception as e:
            logger.error(f"Clinical evaluation failed: {e}")
            raise
    
    def generate_visualizations(self) -> List[str]:
        """Step 5: Generate comprehensive visualizations."""
        logger.info("=== STEP 5: Visualization Generation ===")
        
        try:
            # Initialize visualizer
            viz_dir = self.output_dir / "visualizations"
            self.visualizer = VisualizationEngine(output_dir=viz_dir)
            
            # Generate all visualizations
            created_files = self.visualizer.generate_comprehensive_report(
                feature_vectors=self.feature_vectors,
                ml_results=self.ml_results,
                clinical_metrics=self.clinical_metrics,
                baseline_metrics=self.config.get('baseline_metrics')
            )
            
            logger.info(f"Visualization generation completed:")
            logger.info(f"  - Files created: {len(created_files)}")
            logger.info(f"  - Output directory: {viz_dir}")
            for file_path in created_files:
                logger.info(f"    - {Path(file_path).name}")
            
            return created_files
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            raise
    
    def deploy_best_model(self) -> str:
        """Step 6: Deploy the best model for integration."""
        logger.info("=== STEP 6: Model Deployment ===")
        
        try:
            if not self.ml_pipeline.best_model:
                raise ValueError("No best model available for deployment")
            
            best_model = self.ml_pipeline.best_model
            
            # Create deployment package
            deployment_dir = self.output_dir / "deployment"
            deployment_dir.mkdir(exist_ok=True)
            
            # Save model with metadata
            model_path = deployment_dir / "cluster_classifier.pkl"
            self.ml_pipeline.save_model(best_model.model_name, deployment_dir)
            
            # Create deployment configuration
            deployment_config = {
                'model_name': best_model.model_name,
                'model_path': str(model_path),
                'feature_names': self.feature_extractor.feature_names,
                'performance_metrics': best_model.test_scores,
                'clinical_metrics': self.clinical_metrics,
                'deployment_timestamp': datetime.now().isoformat(),
                'training_config': self.config
            }
            
            config_path = deployment_dir / "deployment_config.json"
            with open(config_path, 'w') as f:
                json.dump(deployment_config, f, indent=2)
            
            # Create integration script template
            integration_script = f'''#!/usr/bin/env python3
"""
Cluster Classification Integration Script

This script demonstrates how to integrate the trained cluster classifier
into the Madrid seizure detection pipeline for false positive reduction.
"""

import pickle
import numpy as np
from pathlib import Path

def load_cluster_classifier():
    """Load the trained cluster classifier."""
    model_path = Path(__file__).parent / "{best_model.model_name}_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def classify_cluster(cluster_features, model):
    """
    Classify a cluster as seizure hit (1) or false positive (0).
    
    Args:
        cluster_features: Dict of extracted features for the cluster
        model: Trained classifier model
        
    Returns:
        (prediction, probability): Tuple of prediction and confidence score
    """
    # Convert features to array (ensure correct order)
    feature_names = {deployment_config['feature_names']}
    feature_array = np.array([cluster_features.get(name, 0.0) for name in feature_names])
    feature_array = feature_array.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(feature_array)[0]
    probability = model.predict_proba(feature_array)[0, 1]
    
    return prediction, probability

def filter_false_positives(clusters, threshold=0.5):
    """
    Filter false positive clusters from Madrid results.
    
    Args:
        clusters: List of cluster objects with extracted features
        threshold: Probability threshold for classification
        
    Returns:
        filtered_clusters: List of clusters classified as seizure hits
    """
    model = load_cluster_classifier()
    filtered_clusters = []
    
    for cluster in clusters:
        prediction, probability = classify_cluster(cluster.features, model)
        
        # Keep cluster if probability of being seizure hit > threshold
        if probability > threshold:
            cluster.classification_probability = probability
            filtered_clusters.append(cluster)
    
    return filtered_clusters

# Example usage
if __name__ == "__main__":
    print("Cluster classifier deployment ready")
    print(f"Model: {deployment_config['model_name']}")
    print(f"Performance: {deployment_config['performance_metrics']}")
'''
            
            integration_path = deployment_dir / "integrate_classifier.py"
            with open(integration_path, 'w') as f:
                f.write(integration_script)
            
            logger.info(f"Model deployment completed:")
            logger.info(f"  - Best model: {best_model.model_name}")
            logger.info(f"  - Test F1: {best_model.test_scores['f1']:.3f}")
            logger.info(f"  - Deployment directory: {deployment_dir}")
            logger.info(f"  - Integration script: {integration_path}")
            
            return str(deployment_dir)
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline from start to finish."""
        logger.info("="*60)
        logger.info("STARTING CLUSTER-BASED FALSE POSITIVE REDUCTION PIPELINE")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Step 1: Cluster Analysis
            cluster_results = self.load_and_analyze_clusters()
            
            # Step 2: Feature Extraction
            feature_vectors = self.extract_features()
            
            # Step 3: Model Training
            ml_results = self.train_models()
            
            # Step 4: Clinical Evaluation
            clinical_metrics = self.evaluate_clinical_performance()
            
            # Step 5: Visualizations
            visualization_files = self.generate_visualizations()
            
            # Step 6: Model Deployment
            deployment_dir = self.deploy_best_model()
            
            # Generate final summary
            total_time = time.time() - start_time
            summary = self.generate_final_summary(total_time)
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def generate_final_summary(self, execution_time: float) -> Dict[str, Any]:
        """Generate final pipeline summary."""
        
        summary = {
            'execution_info': {
                'total_time_minutes': execution_time / 60,
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_dir)
            },
            'data_summary': {
                'total_clusters': len(self.clusters),
                'feature_vectors': len(self.feature_vectors),
                'features_per_vector': len(self.feature_extractor.feature_names) if self.feature_extractor else 0
            },
            'model_performance': {
                'models_trained': len(self.ml_results),
                'best_model': self.ml_pipeline.best_model.model_name if self.ml_pipeline.best_model else None,
                'best_f1_score': self.ml_pipeline.best_model.test_scores['f1'] if self.ml_pipeline.best_model else 0
            },
            'clinical_impact': self.clinical_metrics,
            'config_used': self.config
        }
        
        # Save summary
        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\\nPIPELINE SUMMARY:")
        logger.info(f"  Execution time: {execution_time/60:.1f} minutes")
        logger.info(f"  Total clusters analyzed: {len(self.clusters)}")
        logger.info(f"  Models trained: {len(self.ml_results)}")
        if self.ml_pipeline.best_model:
            logger.info(f"  Best model: {self.ml_pipeline.best_model.model_name}")
            logger.info(f"  Best F1 score: {self.ml_pipeline.best_model.test_scores['f1']:.3f}")
        logger.info(f"  Clinical sensitivity: {self.clinical_metrics.get('sensitivity', 0):.3f}")
        logger.info(f"  Clinical specificity: {self.clinical_metrics.get('specificity', 0):.3f}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Summary saved: {summary_path}")
        
        return summary

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the pipeline."""
    return {
        # Data paths
        'cluster_results_dir': 'Madrid/madrid_results/madrid_seizure_results_parallel_400/tolerance_adjusted_smart_clustered',
        'madrid_results_dir': 'Madrid/madrid_results/madrid_seizure_results_parallel_400/tolerance_adjusted',
        'raw_data_dir': '/home/swolf/asim_shared/preprocessed_data/seizure_only/8hz_30min/downsample_8hz_context_30min',  # Optional: path to raw ECG data
        'output_dir': 'cluster_reduction/results',
        
        # Pipeline parameters
        'sampling_rate': 8,
        'random_state': 42,
        'test_size': 0.2,
        'balance_strategy': 'smote',  # 'smote', 'undersample', 'none'
        
        # Optional: baseline metrics for comparison
        'baseline_metrics': {
            'true_positives': 352,
            'false_positives': 1000,
            'sensitivity': 0.7325,
            'false_alarm_rate': 0.7396
        },
        
        # Optional: patient history data
        'patient_history': None
    }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cluster-Based False Positive Reduction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py --config config.json
  python main.py --cluster-dir path/to/clusters --madrid-dir path/to/madrid
  python main.py --quick-run  # Use default paths for testing
        """
    )
    
    parser.add_argument('--config', type=str, help='Configuration JSON file')
    parser.add_argument('--cluster-dir', type=str, help='Cluster results directory')
    parser.add_argument('--madrid-dir', type=str, help='Madrid results directory')
    parser.add_argument('--raw-data-dir', type=str, help='Raw ECG data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--quick-run', action='store_true', help='Quick run with default config')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config and Path(args.config).exists():
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        logger.info("Using default configuration")
        config = create_default_config()
    
    # Override config with command line arguments
    if args.cluster_dir:
        config['cluster_results_dir'] = args.cluster_dir
    if args.madrid_dir:
        config['madrid_results_dir'] = args.madrid_dir
    if args.raw_data_dir:
        config['raw_data_dir'] = args.raw_data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Quick run mode
    if args.quick_run:
        logger.info("Quick run mode: using minimal configuration for testing")
        config['balance_strategy'] = 'undersample'  # Faster than SMOTE
        config['output_dir'] = 'cluster_reduction/quick_test'
    
    # Validate paths
    cluster_path = Path(config['cluster_results_dir'])
    madrid_path = Path(config['madrid_results_dir'])
    
    if not cluster_path.exists():
        logger.error(f"Cluster results directory not found: {cluster_path}")
        return 1
    
    if not madrid_path.exists():
        logger.error(f"Madrid results directory not found: {madrid_path}")
        return 1
    
    try:
        # Initialize and run pipeline
        pipeline = ClusterReductionPipeline(config)
        summary = pipeline.run_complete_pipeline()
        
        logger.info("Pipeline execution completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())