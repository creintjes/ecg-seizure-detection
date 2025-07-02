#!/usr/bin/env python3
"""
Visualization Module for Cluster-Based False Positive Reduction

This module creates comprehensive visualizations for:
- Feature analysis and distributions
- Model performance comparisons
- Feature importance analysis
- Clinical performance metrics
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class VisualizationEngine:
    """Main visualization engine for cluster reduction analysis."""
    
    def __init__(self, output_dir: str = "cluster_reduction/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set consistent style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def create_feature_distribution_plots(self, feature_vectors: List, 
                                        top_n_features: int = 20) -> str:
        """Create feature distribution plots comparing seizure vs non-seizure clusters."""
        
        # Convert to DataFrame
        data = []
        for fv in feature_vectors:
            row = {'label': 'Seizure Hit' if fv.label == 1 else 'False Positive'}
            row.update(fv.features)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate feature importance (simple variance-based)
        feature_cols = [col for col in df.columns if col != 'label']
        feature_importance = {}
        
        for col in feature_cols:
            try:
                # Calculate separation between classes
                seizure_data = df[df['label'] == 'Seizure Hit'][col]
                fp_data = df[df['label'] == 'False Positive'][col]
                
                if len(seizure_data) > 0 and len(fp_data) > 0:
                    # Use Cohen's d as measure of separation
                    mean_diff = abs(seizure_data.mean() - fp_data.mean())
                    pooled_std = np.sqrt(((len(seizure_data) - 1) * seizure_data.var() + 
                                        (len(fp_data) - 1) * fp_data.var()) / 
                                       (len(seizure_data) + len(fp_data) - 2))
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    feature_importance[col] = cohens_d
                else:
                    feature_importance[col] = 0
            except:
                feature_importance[col] = 0
        
        # Select top features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
        top_feature_names = [f[0] for f in top_features]
        
        # Create subplots
        n_cols = 4
        n_rows = (len(top_feature_names) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
        
        # Plot distributions
        for i, feature in enumerate(top_feature_names):
            if i < len(axes):
                ax = axes[i]
                
                # Create histogram
                seizure_data = df[df['label'] == 'Seizure Hit'][feature].dropna()
                fp_data = df[df['label'] == 'False Positive'][feature].dropna()
                
                if len(seizure_data) > 0 and len(fp_data) > 0:
                    ax.hist(seizure_data, alpha=0.6, label='Seizure Hit', bins=20, color='red')
                    ax.hist(fp_data, alpha=0.6, label='False Positive', bins=20, color='blue')
                    
                    ax.set_title(f'{feature}\n(Cohen\'s d = {feature_importance[feature]:.3f})')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(feature)
        
        # Remove empty subplots
        for i in range(len(top_feature_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        output_path = self.output_dir / 'feature_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature distribution plots saved to {output_path}")
        return str(output_path)
    
    def create_correlation_matrix(self, feature_vectors: List, 
                                top_n_features: int = 30) -> str:
        """Create correlation matrix for top features."""
        
        # Convert to DataFrame
        data = []
        for fv in feature_vectors:
            row = fv.features.copy()
            row['label'] = fv.label
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Select numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in numeric_cols:
            numeric_cols.remove('label')
        
        # Select top features by variance
        feature_variances = df[numeric_cols].var().sort_values(ascending=False)
        top_features = feature_variances.head(top_n_features).index.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[top_features].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 12))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix (Top Features by Variance)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = self.output_dir / 'correlation_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation matrix saved to {output_path}")
        return str(output_path)
    
    def create_model_comparison_plot(self, ml_results: Dict[str, Any]) -> str:
        """Create model performance comparison plot."""
        
        # Extract metrics
        models = list(ml_results.keys())
        metrics = ['precision', 'recall', 'f1', 'roc_auc']
        
        # Prepare data
        cv_data = {metric: [] for metric in metrics}
        test_data = {metric: [] for metric in metrics}
        
        for model_name in models:
            result = ml_results[model_name]
            for metric in metrics:
                cv_data[metric].append(result.cv_scores.get(metric, 0))
                test_data[metric].append(result.test_scores.get(metric, 0))
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # CV scores
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i * width, cv_data[metric], width, label=metric.capitalize())
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Cross-Validation Performance')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Test scores
        for i, metric in enumerate(metrics):
            ax2.bar(x + i * width, test_data[metric], width, label=metric.capitalize())
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.set_title('Test Set Performance')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        output_path = self.output_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {output_path}")
        return str(output_path)
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float], 
                                     top_n: int = 20, model_name: str = "Model") -> str:
        """Create feature importance plot."""
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        if not top_features:
            logger.warning("No feature importance data available")
            return ""
        
        # Extract names and values
        feature_names = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(feature_names) * 0.4)))
        
        bars = ax.barh(range(len(feature_names)), importance_values, alpha=0.7)
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_values)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, v in enumerate(importance_values):
            ax.text(v + max(importance_values) * 0.01, i, f'{v:.3f}', 
                   va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
        return str(output_path)
    
    def create_roc_curves(self, ml_results: Dict[str, Any], 
                         y_true: np.ndarray) -> str:
        """Create ROC curves for all models."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each model
        for model_name, result in ml_results.items():
            try:
                from sklearn.metrics import roc_curve, auc
                
                y_pred_proba = result.prediction_probabilities
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc_score = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
                
            except Exception as e:
                logger.warning(f"Could not create ROC curve for {model_name}: {e}")
                continue
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Model Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        output_path = self.output_dir / 'roc_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {output_path}")
        return str(output_path)
    
    def create_precision_recall_curves(self, ml_results: Dict[str, Any], 
                                     y_true: np.ndarray) -> str:
        """Create Precision-Recall curves for all models."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot PR curve for each model
        for model_name, result in ml_results.items():
            try:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                
                y_pred_proba = result.prediction_probabilities
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                ap_score = average_precision_score(y_true, y_pred_proba)
                
                ax.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})', linewidth=2)
                
            except Exception as e:
                logger.warning(f"Could not create PR curve for {model_name}: {e}")
                continue
        
        # Plot baseline
        positive_ratio = np.mean(y_true)
        ax.axhline(y=positive_ratio, color='k', linestyle='--', alpha=0.5, 
                  label=f'Random Classifier (AP = {positive_ratio:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves - Model Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        output_path = self.output_dir / 'precision_recall_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curves saved to {output_path}")
        return str(output_path)
    
    def create_confusion_matrix_plot(self, ml_results: Dict[str, Any]) -> str:
        """Create confusion matrix plots for all models."""
        
        n_models = len(ml_results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for i, (model_name, result) in enumerate(ml_results.items()):
            ax = axes[i]
            
            cm = result.confusion_matrix
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['False Positive', 'Seizure Hit'],
                       yticklabels=['False Positive', 'Seizure Hit'])
            
            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            # Add performance metrics as text
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
            ax.text(2.1, 0.5, metrics_text, transform=ax.transData, 
                   verticalalignment='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Remove empty subplots
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        output_path = self.output_dir / 'confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plots saved to {output_path}")
        return str(output_path)
    
    def create_clinical_performance_dashboard(self, clinical_metrics: Dict[str, float], 
                                            baseline_metrics: Optional[Dict[str, float]] = None) -> str:
        """Create clinical performance dashboard."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Sensitivity vs Specificity
        sensitivity = clinical_metrics.get('sensitivity', 0)
        specificity = clinical_metrics.get('specificity', 0)
        
        ax1.scatter(1 - specificity, sensitivity, s=200, c='red', alpha=0.7, label='Current Model')
        
        if baseline_metrics:
            baseline_sens = baseline_metrics.get('sensitivity', 0)
            baseline_spec = baseline_metrics.get('specificity', 0)
            ax1.scatter(1 - baseline_spec, baseline_sens, s=200, c='blue', alpha=0.7, label='Baseline')
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate (1 - Specificity)')
        ax1.set_ylabel('True Positive Rate (Sensitivity)')
        ax1.set_title('ROC Space - Clinical Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 2. Clinical Metrics Bar Chart
        metrics_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Diagnostic Accuracy']
        metrics_values = [
            clinical_metrics.get('sensitivity', 0),
            clinical_metrics.get('specificity', 0),
            clinical_metrics.get('positive_predictive_value', 0),
            clinical_metrics.get('negative_predictive_value', 0),
            clinical_metrics.get('diagnostic_accuracy', 0)
        ]
        
        bars = ax2.bar(metrics_names, metrics_values, alpha=0.7, color='skyblue')
        ax2.set_ylabel('Score')
        ax2.set_title('Clinical Performance Metrics')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Confusion Matrix Visualization
        tp = clinical_metrics.get('true_positives', 0)
        tn = clinical_metrics.get('true_negatives', 0)
        fp = clinical_metrics.get('false_positives', 0)
        fn = clinical_metrics.get('false_negatives', 0)
        
        cm_data = np.array([[tn, fp], [fn, tp]])
        
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        ax3.set_title('Confusion Matrix')
        
        # 4. False Positive Reduction (if baseline provided)
        if baseline_metrics:
            baseline_fp = baseline_metrics.get('false_positives', 0)
            baseline_tp = baseline_metrics.get('true_positives', 0)
            
            fp_reduction = ((baseline_fp - fp) / baseline_fp * 100) if baseline_fp > 0 else 0
            sensitivity_change = ((tp - baseline_tp) / baseline_tp * 100) if baseline_tp > 0 else 0
            
            categories = ['FP Reduction (%)', 'Sensitivity Change (%)']
            values = [fp_reduction, sensitivity_change]
            colors = ['green' if v >= 0 else 'red' for v in values]
            
            bars = ax4.bar(categories, values, color=colors, alpha=0.7)
            ax4.set_ylabel('Percentage Change')
            ax4.set_title('Performance Improvement vs Baseline')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (5 if value >= 0 else -5),
                        f'{value:.1f}%', ha='center', 
                        va='bottom' if value >= 0 else 'top')
        else:
            ax4.text(0.5, 0.5, 'No baseline\nfor comparison', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Baseline Comparison (N/A)')
        
        plt.tight_layout()
        output_path = self.output_dir / 'clinical_performance_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Clinical performance dashboard saved to {output_path}")
        return str(output_path)
    
    def create_feature_category_analysis(self, feature_importance: Dict[str, float]) -> str:
        """Analyze feature importance by categories."""
        
        # Define feature categories
        categories = {
            'Cluster': ['cluster_size', 'anomaly_density', 'temporal_spread', 'spatial_consistency',
                       'cluster_duration', 'inter_anomaly', 'temporal_clustering'],
            'Madrid Algorithm': ['avg_anomaly_score', 'max_anomaly_score', 'score_variance', 
                               'optimal_m_value', 'm_value_diversity', 'rank'],
            'Heart Rate': ['hr_mean', 'hr_std', 'hr_trend', 'hr_acceleration', 'hr_min', 'hr_max'],
            'HRV': ['rr_', 'hrv_', 'pnn50', 'rmssd', 'sdnn', 'triangular'],
            'Signal Quality': ['signal_to_noise', 'baseline_drift', 'artifact', 'r_peak', 'missing_beats'],
            'Autonomic': ['cardiac_sympathetic', 'sample_entropy', 'deceleration', 'acceleration'],
            'Contextual': ['patient_', 'recording_', 'time_', 'seizure_proximity', 'relative_position']
        }
        
        # Categorize features
        category_importance = {cat: [] for cat in categories.keys()}
        uncategorized = []
        
        for feature, importance in feature_importance.items():
            categorized = False
            for category, keywords in categories.items():
                if any(keyword in feature.lower() for keyword in keywords):
                    category_importance[category].append(importance)
                    categorized = True
                    break
            if not categorized:
                uncategorized.append((feature, importance))
        
        # Calculate average importance per category
        avg_importance = {}
        for category, importances in category_importance.items():
            if importances:
                avg_importance[category] = np.mean(importances)
            else:
                avg_importance[category] = 0
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Category importance
        categories_sorted = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        cat_names = [c[0] for c in categories_sorted]
        cat_values = [c[1] for c in categories_sorted]
        
        bars = ax1.bar(cat_names, cat_values, alpha=0.7)
        ax1.set_ylabel('Average Feature Importance')
        ax1.set_title('Feature Importance by Category')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(cat_values)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Feature count per category
        cat_counts = [len(category_importance[cat]) for cat in cat_names]
        
        ax2.bar(cat_names, cat_counts, alpha=0.7, color='orange')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Feature Count by Category')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add count labels
        for i, count in enumerate(cat_counts):
            ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = self.output_dir / 'feature_category_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature category analysis saved to {output_path}")
        return str(output_path)
    
    def generate_comprehensive_report(self, feature_vectors: List, ml_results: Dict[str, Any],
                                    clinical_metrics: Dict[str, float],
                                    baseline_metrics: Optional[Dict[str, float]] = None) -> List[str]:
        """Generate all visualizations and return list of created files."""
        
        logger.info("Generating comprehensive visualization report...")
        
        created_files = []
        
        try:
            # Feature analysis
            logger.info("Creating feature distribution plots...")
            file_path = self.create_feature_distribution_plots(feature_vectors)
            created_files.append(file_path)
            
            logger.info("Creating correlation matrix...")
            file_path = self.create_correlation_matrix(feature_vectors)
            created_files.append(file_path)
            
            # Model performance
            logger.info("Creating model comparison plot...")
            file_path = self.create_model_comparison_plot(ml_results)
            created_files.append(file_path)
            
            logger.info("Creating confusion matrix plots...")
            file_path = self.create_confusion_matrix_plot(ml_results)
            created_files.append(file_path)
            
            # Feature importance (for best model)
            if ml_results:
                best_model_name = max(ml_results.keys(), 
                                    key=lambda k: ml_results[k].test_scores.get('f1', 0))
                best_result = ml_results[best_model_name]
                
                logger.info(f"Creating feature importance plot for {best_model_name}...")
                file_path = self.create_feature_importance_plot(
                    best_result.feature_importance, model_name=best_model_name
                )
                created_files.append(file_path)
                
                logger.info("Creating feature category analysis...")
                file_path = self.create_feature_category_analysis(best_result.feature_importance)
                created_files.append(file_path)
            
            # Clinical performance
            logger.info("Creating clinical performance dashboard...")
            file_path = self.create_clinical_performance_dashboard(clinical_metrics, baseline_metrics)
            created_files.append(file_path)
            
            # ROC and PR curves
            if ml_results:
                # We need y_true for these plots - this would come from the test set
                logger.info("ROC and PR curves require test labels - skipping for now")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        logger.info(f"Visualization report completed. Created {len(created_files)} files.")
        return created_files

def main():
    """Example usage of visualization engine."""
    logger.info("Visualization module loaded successfully")
    
    # Initialize visualization engine
    viz = VisualizationEngine()
    logger.info(f"Visualization output directory: {viz.output_dir}")
    
    return viz

if __name__ == "__main__":
    viz = main()