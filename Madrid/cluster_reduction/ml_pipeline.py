#!/usr/bin/env python3
"""
Machine Learning Pipeline for Cluster-Based False Positive Reduction

This module implements the complete ML pipeline including:
- Patient-stratified cross-validation
- Model training and optimization
- Performance evaluation
- Feature importance analysis
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ML libraries
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, matthews_corrcoef
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelResults:
    """Data class for storing model results."""
    model_name: str
    model: Any
    scaler: Any
    feature_selector: Any
    cv_scores: Dict[str, float]
    test_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    predictions: np.ndarray
    prediction_probabilities: np.ndarray
    confusion_matrix: np.ndarray
    best_params: Dict[str, Any]

class MLPipeline:
    """Machine Learning Pipeline for cluster classification."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_names = []
        
        # Set random seeds
        np.random.seed(random_state)
        
    def prepare_data(self, feature_vectors: List) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare data for ML pipeline."""
        # Convert feature vectors to DataFrame
        data = []
        patient_groups = []
        
        for fv in feature_vectors:
            row = {'cluster_id': fv.cluster_id, 'label': fv.label}
            row.update(fv.features)
            data.append(row)
            
            # Extract patient ID for grouping
            patient_id = fv.cluster_id.split('_')[0] if '_' in fv.cluster_id else fv.cluster_id
            patient_groups.append(patient_id)
        
        df = pd.DataFrame(data)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['cluster_id', 'label']]
        X = df[feature_cols].values
        y = df['label'].values
        groups = np.array(patient_groups)
        
        self.feature_names = feature_cols
        
        logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        logger.info(f"Number of patient groups: {len(np.unique(groups))}")
        
        return df, X, y, groups
    
    def setup_models(self) -> Dict[str, Any]:
        """Setup ML models with hyperparameter grids."""
        models = {}
        
        # Random Forest
        models['random_forest'] = {
            'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, 30, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2', None]
            }
        }
        
        # Logistic Regression
        models['logistic_regression'] = {
            'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'params': {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                'classifier__solver': ['liblinear', 'saga']
            }
        }
        
        # Support Vector Machine
        models['svm'] = {
            'model': SVC(random_state=self.random_state, probability=True),
            'params': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'classifier__kernel': ['rbf', 'linear', 'poly']
            }
        }
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            models['xgboost'] = {
                'model': XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    n_jobs=-1
                ),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 6, 10],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__subsample': [0.8, 0.9, 1.0],
                    'classifier__colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        
        self.models = models
        logger.info(f"Setup {len(models)} models: {list(models.keys())}")
        return models
    
    def create_pipeline(self, model, balance_strategy: str = 'smote', 
                       feature_selection: bool = True, scaling: str = 'robust') -> ImbPipeline:
        """Create a complete ML pipeline."""
        steps = []
        
        # Scaling
        if scaling == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaling == 'robust':
            steps.append(('scaler', RobustScaler()))
        
        # Balancing
        if balance_strategy == 'smote':
            steps.append(('balancer', SMOTE(random_state=self.random_state)))
        elif balance_strategy == 'undersample':
            steps.append(('balancer', RandomUnderSampler(random_state=self.random_state)))
        
        # Feature selection
        if feature_selection:
            # Use SelectKBest with f_classif for initial selection
            k_features = min(50, len(self.feature_names))  # Select top 50 or all if fewer
            steps.append(('feature_selector', SelectKBest(f_classif, k=k_features)))
        
        # Classifier
        steps.append(('classifier', model))
        
        return ImbPipeline(steps)
    
    def evaluate_model_cv(self, pipeline, X: np.ndarray, y: np.ndarray, 
                         groups: np.ndarray, param_grid: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Evaluate model using patient-stratified cross-validation."""
        
        # Setup GroupKFold for patient-stratified CV
        cv = GroupKFold(n_splits=5)
        
        # Custom scoring
        scoring = {
            'roc_auc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'accuracy': 'accuracy'
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            refit='f1',  # Optimize for F1 score
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        logger.info("Starting cross-validation...")
        grid_search.fit(X, y, groups=groups)
        
        # Extract CV scores
        cv_scores = {
            metric: grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
            for metric in scoring.keys()
        }
        
        logger.info(f"CV completed. Best F1: {cv_scores['f1']:.3f}")
        return grid_search.best_estimator_, cv_scores
    
    def evaluate_model_test(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set."""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        test_scores = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred)
        }
        
        return test_scores, y_pred, y_pred_proba
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        importance_dict = {}
        
        try:
            # Get the classifier from the pipeline
            classifier = model.named_steps['classifier']
            
            # Get feature names after selection
            if 'feature_selector' in model.named_steps:
                selector = model.named_steps['feature_selector']
                selected_features = np.array(feature_names)[selector.get_support()]
            else:
                selected_features = feature_names
            
            # Extract importance based on model type
            if hasattr(classifier, 'feature_importances_'):
                # Tree-based models (RF, XGBoost)
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                # Linear models (LogisticRegression, SVM with linear kernel)
                importances = np.abs(classifier.coef_[0])
            else:
                # Fallback: use permutation importance or zeros
                logger.warning(f"Cannot extract feature importance for {type(classifier)}")
                importances = np.zeros(len(selected_features))
            
            # Create importance dictionary
            for feature, importance in zip(selected_features, importances):
                importance_dict[feature] = float(importance)
                
        except Exception as e:
            logger.warning(f"Failed to extract feature importance: {e}")
            importance_dict = {name: 0.0 for name in feature_names}
        
        return importance_dict
    
    def train_and_evaluate_all_models(self, feature_vectors: List, 
                                    test_size: float = 0.2,
                                    balance_strategy: str = 'smote') -> Dict[str, ModelResults]:
        """Train and evaluate all models."""
        
        # Prepare data
        df, X, y, groups = self.prepare_data(feature_vectors)
        
        # Patient-stratified train-test split
        unique_groups = np.unique(groups)
        train_groups, test_groups = train_test_split(
            unique_groups, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=None  # Can't stratify by class for groups
        )
        
        train_mask = np.isin(groups, train_groups)
        test_mask = np.isin(groups, test_groups)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train = groups[train_mask]
        
        logger.info(f"Train set: {X_train.shape[0]} samples, {len(train_groups)} patients")
        logger.info(f"Test set: {X_test.shape[0]} samples, {len(test_groups)} patients")
        logger.info(f"Train class distribution: {np.bincount(y_train)}")
        logger.info(f"Test class distribution: {np.bincount(y_test)}")
        
        # Setup models
        self.setup_models()
        
        # Train and evaluate each model
        results = {}
        
        for model_name, model_config in self.models.items():
            logger.info(f"\n=== Training {model_name} ===")
            
            try:
                # Create pipeline
                pipeline = self.create_pipeline(
                    model_config['model'],
                    balance_strategy=balance_strategy,
                    feature_selection=True,
                    scaling='robust'
                )
                
                # Cross-validation
                best_model, cv_scores = self.evaluate_model_cv(
                    pipeline, X_train, y_train, groups_train, model_config['params']
                )
                
                # Test evaluation
                test_scores, y_pred, y_pred_proba = self.evaluate_model_test(best_model, X_test, y_test)
                
                # Feature importance
                feature_importance = self.get_feature_importance(best_model, self.feature_names)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Store results
                result = ModelResults(
                    model_name=model_name,
                    model=best_model,
                    scaler=best_model.named_steps.get('scaler'),
                    feature_selector=best_model.named_steps.get('feature_selector'),
                    cv_scores=cv_scores,
                    test_scores=test_scores,
                    feature_importance=feature_importance,
                    predictions=y_pred,
                    prediction_probabilities=y_pred_proba,
                    confusion_matrix=cm,
                    best_params=best_model.get_params()
                )
                
                results[model_name] = result
                
                # Log results
                logger.info(f"CV F1: {cv_scores['f1']:.3f}, Test F1: {test_scores['f1']:.3f}")
                logger.info(f"Test Precision: {test_scores['precision']:.3f}, Test Recall: {test_scores['recall']:.3f}")
                logger.info(f"Test ROC-AUC: {test_scores['roc_auc']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Select best model based on test F1 score
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k].test_scores['f1'])
            self.best_model = results[best_model_name]
            logger.info(f"\nBest model: {best_model_name} (Test F1: {self.best_model.test_scores['f1']:.3f})")
        
        self.results = results
        return results
    
    def save_model(self, model_name: str, output_dir: str):
        """Save trained model to disk."""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_result = self.results[model_name]
        
        # Save model
        model_file = output_path / f"{model_name}_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_result.model, f)
        
        # Save feature importance
        importance_file = output_path / f"{model_name}_feature_importance.json"
        with open(importance_file, 'w') as f:
            json.dump(model_result.feature_importance, f, indent=2)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'cv_scores': model_result.cv_scores,
            'test_scores': model_result.test_scores,
            'feature_names': self.feature_names,
            'confusion_matrix': model_result.confusion_matrix.tolist(),
            'best_params': {k: str(v) for k, v in model_result.best_params.items()}
        }
        
        metadata_file = output_path / f"{model_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model {model_name} saved to {output_path}")
    
    def save_all_models(self, output_dir: str):
        """Save all trained models."""
        for model_name in self.results.keys():
            self.save_model(model_name, output_dir)
    
    def load_model(self, model_path: str) -> Any:
        """Load a saved model."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def predict_clusters(self, model_name: str, feature_vectors: List) -> Tuple[np.ndarray, np.ndarray]:
        """Use trained model to predict cluster labels."""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not trained")
        
        # Prepare features
        _, X, _, _ = self.prepare_data(feature_vectors)
        
        # Predict
        model = self.results[model_name].model
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def generate_performance_report(self, output_path: str):
        """Generate comprehensive performance report."""
        if not self.results:
            raise ValueError("No models trained yet")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("Machine Learning Pipeline Performance Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall summary
            f.write("MODEL COMPARISON SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Model':<20} {'CV F1':<8} {'Test F1':<8} {'Test Prec':<10} {'Test Rec':<10} {'Test AUC':<10}\n")
            f.write("-" * 70 + "\n")
            
            for model_name, result in self.results.items():
                cv_f1 = result.cv_scores['f1']
                test_f1 = result.test_scores['f1']
                test_prec = result.test_scores['precision']
                test_rec = result.test_scores['recall']
                test_auc = result.test_scores['roc_auc']
                
                f.write(f"{model_name:<20} {cv_f1:<8.3f} {test_f1:<8.3f} {test_prec:<10.3f} {test_rec:<10.3f} {test_auc:<10.3f}\n")
            
            # Best model details
            if self.best_model:
                f.write(f"\n\nBEST MODEL: {self.best_model.model_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Cross-Validation Scores:\n")
                for metric, score in self.best_model.cv_scores.items():
                    f.write(f"  {metric}: {score:.3f}\n")
                
                f.write(f"\nTest Set Scores:\n")
                for metric, score in self.best_model.test_scores.items():
                    f.write(f"  {metric}: {score:.3f}\n")
                
                f.write(f"\nConfusion Matrix:\n")
                cm = self.best_model.confusion_matrix
                f.write(f"  TN: {cm[0,0]}, FP: {cm[0,1]}\n")
                f.write(f"  FN: {cm[1,0]}, TP: {cm[1,1]}\n")
                
                # Top features
                f.write(f"\nTop 20 Important Features:\n")
                f.write("-" * 25 + "\n")
                sorted_features = sorted(
                    self.best_model.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for feature, importance in sorted_features[:20]:
                    f.write(f"  {feature:<40} {importance:.4f}\n")
        
        logger.info(f"Performance report saved to {output_file}")

class ModelEvaluator:
    """Advanced model evaluation and analysis."""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def clinical_performance_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate clinical performance metrics."""
        
        # Basic confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # Performance ratios
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Clinical utility
        diagnostic_accuracy = (tp + tn) / (tp + tn + fp + fn)
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # AUC and MCC
        auc = roc_auc_score(y_true, y_pred_proba)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'diagnostic_accuracy': diagnostic_accuracy,
            'balanced_accuracy': balanced_accuracy,
            'auc_roc': auc,
            'matthews_correlation_coefficient': mcc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def false_positive_reduction_analysis(self, baseline_fp: int, new_fp: int, 
                                        baseline_tp: int, new_tp: int) -> Dict[str, float]:
        """Analyze false positive reduction compared to baseline."""
        
        fp_reduction_abs = baseline_fp - new_fp
        fp_reduction_pct = (fp_reduction_abs / baseline_fp * 100) if baseline_fp > 0 else 0.0
        
        sensitivity_change = (new_tp / baseline_tp - 1) * 100 if baseline_tp > 0 else 0.0
        
        # Calculate improvement ratio
        if fp_reduction_abs > 0 and abs(sensitivity_change) < 10:  # < 10% sensitivity loss
            improvement_ratio = fp_reduction_abs / max(abs(sensitivity_change), 1)
        else:
            improvement_ratio = 0.0
        
        return {
            'fp_reduction_absolute': fp_reduction_abs,
            'fp_reduction_percentage': fp_reduction_pct,
            'sensitivity_change_percentage': sensitivity_change,
            'improvement_ratio': improvement_ratio,
            'new_false_positives': new_fp,
            'new_true_positives': new_tp
        }

def main():
    """Example usage of ML pipeline."""
    logger.info("ML Pipeline module loaded successfully")
    
    # Initialize pipeline
    pipeline = MLPipeline(random_state=42)
    
    # Example of available models
    models = pipeline.setup_models()
    logger.info(f"Available models: {list(models.keys())}")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()