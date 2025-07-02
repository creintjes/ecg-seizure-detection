#!/usr/bin/env python3
"""
Feature Extraction Module for Cluster-Based False Positive Reduction

This module implements comprehensive feature extraction for:
1. Cluster-level features
2. Signal-level ECG/HRV features  
3. Contextual features
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import find_peaks
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class FeatureVector:
    """Data class for extracted features."""
    cluster_id: str
    features: Dict[str, float]
    feature_names: List[str]
    label: int  # 1 for seizure hit, 0 for false positive

class ClusterFeatureExtractor:
    """Extract cluster-level features from anomaly clusters."""
    
    def __init__(self):
        self.feature_names = [
            # Cluster composition
            'cluster_size',
            'anomaly_density',
            'temporal_spread',
            'spatial_consistency',
            
            # Madrid algorithm features
            'avg_anomaly_score',
            'max_anomaly_score',
            'min_anomaly_score',
            'score_variance',
            'score_std',
            'score_range',
            'optimal_m_value',
            'm_value_diversity',
            'avg_rank',
            'min_rank',
            'max_rank',
            'rank_spread',
            
            # Temporal pattern features
            'cluster_duration',
            'inter_anomaly_mean_interval',
            'inter_anomaly_std_interval',
            'temporal_clustering_coeff',
            'onset_proximity',
            
            # Statistical features
            'score_skewness',
            'score_kurtosis',
            'rank_correlation_score'
        ]
    
    def extract_features(self, cluster) -> Dict[str, float]:
        """Extract all cluster-level features."""
        features = {}
        
        if not cluster.anomalies:
            return {name: 0.0 for name in self.feature_names}
        
        # Get basic data
        scores = [a.anomaly_score for a in cluster.anomalies]
        ranks = [a.rank for a in cluster.anomalies]
        times = [a.location_time_seconds for a in cluster.anomalies]
        m_values = [a.m_value for a in cluster.anomalies]
        
        # Cluster composition features
        features['cluster_size'] = len(cluster.anomalies)
        features['temporal_spread'] = max(times) - min(times) if len(times) > 1 else 0.0
        features['anomaly_density'] = len(cluster.anomalies) / max(features['temporal_spread'], 1.0)
        features['spatial_consistency'] = cluster.spatial_consistency
        
        # Madrid algorithm features
        features['avg_anomaly_score'] = np.mean(scores)
        features['max_anomaly_score'] = np.max(scores)
        features['min_anomaly_score'] = np.min(scores)
        features['score_variance'] = np.var(scores)
        features['score_std'] = np.std(scores)
        features['score_range'] = np.max(scores) - np.min(scores)
        
        # M-value features
        features['optimal_m_value'] = stats.mode(m_values)[0] if m_values else 0
        features['m_value_diversity'] = len(set(m_values))
        
        # Rank features
        features['avg_rank'] = np.mean(ranks)
        features['min_rank'] = np.min(ranks)
        features['max_rank'] = np.max(ranks)
        features['rank_spread'] = np.max(ranks) - np.min(ranks)
        
        # Temporal pattern features
        features['cluster_duration'] = features['temporal_spread']
        
        if len(times) > 1:
            sorted_times = sorted(times)
            intervals = [sorted_times[i+1] - sorted_times[i] for i in range(len(sorted_times)-1)]
            features['inter_anomaly_mean_interval'] = np.mean(intervals)
            features['inter_anomaly_std_interval'] = np.std(intervals)
            
            # Temporal clustering coefficient (how clustered anomalies are in time)
            expected_interval = features['temporal_spread'] / (len(times) - 1)
            actual_std = np.std(intervals)
            features['temporal_clustering_coeff'] = actual_std / expected_interval if expected_interval > 0 else 0
        else:
            features['inter_anomaly_mean_interval'] = 0.0
            features['inter_anomaly_std_interval'] = 0.0
            features['temporal_clustering_coeff'] = 0.0
        
        # Proximity to seizure onset (approximate - would need ground truth)
        # For now, use minimum time as proxy
        features['onset_proximity'] = np.min(times)
        
        # Statistical features
        if len(scores) > 2:
            features['score_skewness'] = stats.skew(scores)
            features['score_kurtosis'] = stats.kurtosis(scores)
        else:
            features['score_skewness'] = 0.0
            features['score_kurtosis'] = 0.0
        
        # Correlation between rank and score
        if len(ranks) > 1 and len(scores) > 1:
            features['rank_correlation_score'] = stats.pearsonr(ranks, scores)[0]
        else:
            features['rank_correlation_score'] = 0.0
        
        # Handle NaN values
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        return features

class SignalFeatureExtractor:
    """Extract ECG and HRV features from raw signal data."""
    
    def __init__(self, sampling_rate: int = 8):
        self.sampling_rate = sampling_rate
        self.feature_names = [
            # Heart rate features
            'hr_mean',
            'hr_std',
            'hr_trend',
            'hr_acceleration',
            'hr_min',
            'hr_max',
            'hr_range',
            
            # RR interval features
            'rr_mean',
            'rr_std',
            'rr_rmssd',
            'rr_pnn50',
            'rr_sdnn',
            'rr_triangular_index',
            
            # HRV frequency domain
            'hrv_lf_power',
            'hrv_hf_power',
            'hrv_lf_hf_ratio',
            'hrv_total_power',
            
            # Signal quality features
            'signal_to_noise_ratio',
            'baseline_drift',
            'artifact_probability',
            'r_peak_confidence',
            'missing_beats_percentage',
            
            # Autonomic features
            'cardiac_sympathetic_index',
            'sample_entropy',
            'deceleration_capacity',
            'acceleration_capacity'
        ]
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """Simple R-peak detection algorithm."""
        try:
            # Bandpass filter for QRS enhancement
            nyquist = self.sampling_rate / 2
            low_freq = 5 / nyquist
            high_freq = min(15, nyquist - 1) / nyquist
            
            if high_freq >= 1.0:
                filtered_signal = ecg_signal  # Skip filtering if frequency too high
            else:
                sos = signal.butter(4, [low_freq, high_freq], btype='band', output='sos')
                filtered_signal = signal.sosfilt(sos, ecg_signal)
            
            # Find peaks
            height_threshold = np.std(filtered_signal) * 0.5
            distance = int(0.6 * self.sampling_rate)  # Minimum 600ms between R-peaks
            
            peaks, properties = find_peaks(
                filtered_signal,
                height=height_threshold,
                distance=distance
            )
            
            # Calculate confidence based on peak quality
            if len(peaks) > 0:
                peak_heights = filtered_signal[peaks]
                confidence = np.mean(peak_heights) / np.std(filtered_signal)
            else:
                confidence = 0.0
            
            return peaks, confidence
            
        except Exception as e:
            logger.warning(f"R-peak detection failed: {e}")
            return np.array([]), 0.0
    
    def calculate_heart_rate_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate heart rate and HRV features from RR intervals."""
        features = {}
        
        if len(rr_intervals) == 0:
            return {name: 0.0 for name in self.feature_names if 'hr_' in name or 'rr_' in name or 'hrv_' in name}
        
        # Convert RR intervals to heart rate
        heart_rates = 60.0 / (rr_intervals / self.sampling_rate)
        heart_rates = heart_rates[heart_rates > 30]  # Remove artifacts
        heart_rates = heart_rates[heart_rates < 200]  # Remove artifacts
        
        if len(heart_rates) == 0:
            return {name: 0.0 for name in self.feature_names if 'hr_' in name or 'rr_' in name or 'hrv_' in name}
        
        # Heart rate features
        features['hr_mean'] = np.mean(heart_rates)
        features['hr_std'] = np.std(heart_rates)
        features['hr_min'] = np.min(heart_rates)
        features['hr_max'] = np.max(heart_rates)
        features['hr_range'] = features['hr_max'] - features['hr_min']
        
        # Calculate trend
        if len(heart_rates) > 1:
            x = np.arange(len(heart_rates))
            slope, _, _, _, _ = stats.linregress(x, heart_rates)
            features['hr_trend'] = slope
            
            # Calculate acceleration (second derivative approximation)
            if len(heart_rates) > 2:
                diff2 = np.diff(heart_rates, 2)
                features['hr_acceleration'] = np.mean(np.abs(diff2))
            else:
                features['hr_acceleration'] = 0.0
        else:
            features['hr_trend'] = 0.0
            features['hr_acceleration'] = 0.0
        
        # RR interval features (convert to milliseconds)
        rr_ms = (rr_intervals / self.sampling_rate) * 1000
        rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]  # Remove artifacts
        
        if len(rr_ms) > 0:
            features['rr_mean'] = np.mean(rr_ms)
            features['rr_std'] = np.std(rr_ms)
            features['rr_sdnn'] = np.std(rr_ms)
            
            # RMSSD
            if len(rr_ms) > 1:
                diff_rr = np.diff(rr_ms)
                features['rr_rmssd'] = np.sqrt(np.mean(diff_rr**2))
                
                # pNN50
                pnn50_count = np.sum(np.abs(diff_rr) > 50)
                features['rr_pnn50'] = (pnn50_count / len(diff_rr)) * 100
            else:
                features['rr_rmssd'] = 0.0
                features['rr_pnn50'] = 0.0
            
            # Triangular index (approximation)
            hist, bins = np.histogram(rr_ms, bins=50)
            features['rr_triangular_index'] = len(rr_ms) / np.max(hist) if np.max(hist) > 0 else 0.0
        else:
            features['rr_mean'] = 0.0
            features['rr_std'] = 0.0
            features['rr_sdnn'] = 0.0
            features['rr_rmssd'] = 0.0
            features['rr_pnn50'] = 0.0
            features['rr_triangular_index'] = 0.0
        
        # Frequency domain HRV (simplified)
        if len(rr_ms) > 10:
            try:
                # Interpolate RR series
                time_axis = np.cumsum(rr_ms)
                interp_time = np.arange(time_axis[0], time_axis[-1], 250)  # 4Hz interpolation
                interp_rr = np.interp(interp_time, time_axis, rr_ms)
                
                # Calculate power spectral density
                freqs, psd = signal.welch(interp_rr, fs=4, nperseg=min(256, len(interp_rr)))
                
                # Define frequency bands
                lf_band = (freqs >= 0.04) & (freqs <= 0.15)
                hf_band = (freqs >= 0.15) & (freqs <= 0.4)
                
                features['hrv_lf_power'] = np.sum(psd[lf_band])
                features['hrv_hf_power'] = np.sum(psd[hf_band])
                features['hrv_total_power'] = np.sum(psd)
                features['hrv_lf_hf_ratio'] = features['hrv_lf_power'] / features['hrv_hf_power'] if features['hrv_hf_power'] > 0 else 0.0
                
            except Exception as e:
                logger.warning(f"HRV frequency analysis failed: {e}")
                features['hrv_lf_power'] = 0.0
                features['hrv_hf_power'] = 0.0
                features['hrv_total_power'] = 0.0
                features['hrv_lf_hf_ratio'] = 0.0
        else:
            features['hrv_lf_power'] = 0.0
            features['hrv_hf_power'] = 0.0
            features['hrv_total_power'] = 0.0
            features['hrv_lf_hf_ratio'] = 0.0
        
        return features
    
    def calculate_signal_quality_features(self, ecg_signal: np.ndarray, r_peaks: np.ndarray, confidence: float) -> Dict[str, float]:
        """Calculate signal quality metrics."""
        features = {}
        
        # Signal-to-noise ratio (simplified)
        if len(ecg_signal) > 0:
            signal_power = np.var(ecg_signal)
            # Estimate noise as high-frequency content
            diff_signal = np.diff(ecg_signal)
            noise_power = np.var(diff_signal)
            features['signal_to_noise_ratio'] = signal_power / noise_power if noise_power > 0 else 10.0
        else:
            features['signal_to_noise_ratio'] = 0.0
        
        # Baseline drift (low-frequency trend)
        if len(ecg_signal) > 10:
            try:
                # Detrend signal
                detrended = signal.detrend(ecg_signal)
                baseline_power = np.var(ecg_signal - detrended)
                features['baseline_drift'] = baseline_power / np.var(ecg_signal) if np.var(ecg_signal) > 0 else 0.0
            except:
                features['baseline_drift'] = 0.0
        else:
            features['baseline_drift'] = 0.0
        
        # Artifact probability (based on amplitude variations)
        if len(ecg_signal) > 0:
            amplitude_changes = np.abs(np.diff(ecg_signal))
            threshold = np.mean(amplitude_changes) + 3 * np.std(amplitude_changes)
            artifacts = np.sum(amplitude_changes > threshold)
            features['artifact_probability'] = artifacts / len(amplitude_changes)
        else:
            features['artifact_probability'] = 0.0
        
        # R-peak confidence
        features['r_peak_confidence'] = confidence
        
        # Missing beats percentage
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks)
            expected_rr = np.median(rr_intervals)
            long_intervals = rr_intervals > (expected_rr * 1.5)
            features['missing_beats_percentage'] = np.sum(long_intervals) / len(rr_intervals) * 100
        else:
            features['missing_beats_percentage'] = 100.0  # No beats detected
        
        return features
    
    def calculate_autonomic_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Calculate autonomic nervous system features."""
        features = {}
        
        if len(rr_intervals) < 5:
            return {
                'cardiac_sympathetic_index': 0.0,
                'sample_entropy': 0.0,
                'deceleration_capacity': 0.0,
                'acceleration_capacity': 0.0
            }
        
        # Cardiac Sympathetic Index (simplified)
        rr_ms = (rr_intervals / self.sampling_rate) * 1000
        if len(rr_ms) > 1:
            features['cardiac_sympathetic_index'] = np.mean(rr_ms) / np.std(rr_ms)
        else:
            features['cardiac_sympathetic_index'] = 0.0
        
        # Sample Entropy (simplified implementation)
        def sample_entropy(data, m=2, r=None):
            """Calculate sample entropy."""
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
                C = np.sum([
                    len([1 for j in range(len(patterns)) if i != j and _maxdist(patterns[i], patterns[j], m) <= r])
                    for i in range(len(patterns))
                ])
                return C / (len(patterns) * (len(patterns) - 1))
            
            try:
                phi_m = _phi(m)
                phi_m1 = _phi(m + 1)
                return -np.log(phi_m1 / phi_m) if phi_m > 0 else 0.0
            except:
                return 0.0
        
        try:
            features['sample_entropy'] = sample_entropy(rr_ms)
        except:
            features['sample_entropy'] = 0.0
        
        # Deceleration/Acceleration Capacity (simplified)
        if len(rr_ms) > 2:
            diff_rr = np.diff(rr_ms)
            accelerations = diff_rr[diff_rr < 0]  # HR increases (RR decreases)
            decelerations = diff_rr[diff_rr > 0]   # HR decreases (RR increases)
            
            features['acceleration_capacity'] = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0.0
            features['deceleration_capacity'] = np.mean(decelerations) if len(decelerations) > 0 else 0.0
        else:
            features['acceleration_capacity'] = 0.0
            features['deceleration_capacity'] = 0.0
        
        return features
    
    def extract_features(self, ecg_signal: np.ndarray, cluster_time_window: Tuple[float, float]) -> Dict[str, float]:
        """Extract all signal-level features from ECG data."""
        try:
            # Extract time window around cluster
            start_time, end_time = cluster_time_window
            start_sample = int(start_time * self.sampling_rate)
            end_sample = int(end_time * self.sampling_rate)
            
            # Ensure valid indices
            start_sample = max(0, start_sample)
            end_sample = min(len(ecg_signal), end_sample)
            
            if start_sample >= end_sample:
                logger.warning(f"Invalid time window: {start_time}-{end_time}")
                return {name: 0.0 for name in self.feature_names}
            
            # Extract signal segment
            signal_segment = ecg_signal[start_sample:end_sample]
            
            if len(signal_segment) == 0:
                return {name: 0.0 for name in self.feature_names}
            
            # Detect R-peaks
            r_peaks, confidence = self.detect_r_peaks(signal_segment)
            
            # Calculate RR intervals
            if len(r_peaks) > 1:
                rr_intervals = np.diff(r_peaks)
            else:
                rr_intervals = np.array([])
            
            # Extract feature groups
            hr_features = self.calculate_heart_rate_features(rr_intervals)
            quality_features = self.calculate_signal_quality_features(signal_segment, r_peaks, confidence)
            autonomic_features = self.calculate_autonomic_features(rr_intervals)
            
            # Combine all features
            all_features = {**hr_features, **quality_features, **autonomic_features}
            
            # Handle missing features
            for feature_name in self.feature_names:
                if feature_name not in all_features:
                    all_features[feature_name] = 0.0
                elif np.isnan(all_features[feature_name]) or np.isinf(all_features[feature_name]):
                    all_features[feature_name] = 0.0
            
            return all_features
            
        except Exception as e:
            logger.warning(f"Signal feature extraction failed: {e}")
            return {name: 0.0 for name in self.feature_names}

class ContextualFeatureExtractor:
    """Extract contextual features (patient, recording, temporal)."""
    
    def __init__(self):
        self.feature_names = [
            # Patient features
            'patient_seizure_count',
            'avg_seizure_duration',
            'patient_baseline_hr',
            'patient_response_consistency',
            
            # Recording context
            'recording_duration',
            'time_of_day_sin',
            'time_of_day_cos',
            'seizure_proximity',
            
            # Temporal features
            'time_since_recording_start',
            'relative_position_in_recording'
        ]
    
    def extract_features(self, cluster, madrid_results: Dict[str, Any], patient_history: Optional[Dict] = None) -> Dict[str, float]:
        """Extract contextual features."""
        features = {}
        
        if not cluster.anomalies:
            return {name: 0.0 for name in self.feature_names}
        
        # Get cluster info
        representative_anomaly = cluster.anomalies[0]
        subject_id = representative_anomaly.subject_id
        
        # Patient features (would need patient database)
        if patient_history and subject_id in patient_history:
            patient_data = patient_history[subject_id]
            features['patient_seizure_count'] = patient_data.get('total_seizures', 1)
            features['avg_seizure_duration'] = patient_data.get('avg_duration', 60.0)
            features['patient_baseline_hr'] = patient_data.get('baseline_hr', 70.0)
            features['patient_response_consistency'] = patient_data.get('response_consistency', 0.5)
        else:
            # Default values
            features['patient_seizure_count'] = 1
            features['avg_seizure_duration'] = 60.0
            features['patient_baseline_hr'] = 70.0
            features['patient_response_consistency'] = 0.5
        
        # Recording context
        session_key = f"{subject_id}_{representative_anomaly.run_id}_{representative_anomaly.seizure_id}"
        if session_key in madrid_results:
            session_data = madrid_results[session_key]
            signal_metadata = session_data.get('input_data', {}).get('signal_metadata', {})
            features['recording_duration'] = signal_metadata.get('signal_duration_seconds', 3600.0)
        else:
            features['recording_duration'] = 3600.0
        
        # Time of day (assume time is in seconds from start of day)
        cluster_time = representative_anomaly.location_time_seconds
        time_of_day = (cluster_time % 86400) / 86400  # Normalize to [0, 1]
        features['time_of_day_sin'] = np.sin(2 * np.pi * time_of_day)
        features['time_of_day_cos'] = np.cos(2 * np.pi * time_of_day)
        
        # Seizure proximity (distance to nearest seizure)
        # For now, use cluster time as proxy
        features['seizure_proximity'] = min(cluster_time, features['recording_duration'] - cluster_time)
        
        # Temporal features
        features['time_since_recording_start'] = cluster_time
        features['relative_position_in_recording'] = cluster_time / features['recording_duration']
        
        # Handle NaN values
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        return features

class MasterFeatureExtractor:
    """Master class that combines all feature extractors."""
    
    def __init__(self, raw_data_dir: Optional[str] = None, sampling_rate: int = 8):
        self.cluster_extractor = ClusterFeatureExtractor()
        self.signal_extractor = SignalFeatureExtractor(sampling_rate)
        self.contextual_extractor = ContextualFeatureExtractor()
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None
        
        # Combine all feature names
        self.feature_names = (
            self.cluster_extractor.feature_names +
            self.signal_extractor.feature_names +
            self.contextual_extractor.feature_names
        )
    
    def load_raw_ecg_data(self, subject_id: str, run_id: str, seizure_id: str) -> Optional[np.ndarray]:
        """Load raw ECG data for signal analysis."""
        if not self.raw_data_dir:
            return None
        
        try:
            # Look for preprocessed file
            pattern = f"{subject_id}_{run_id}_{seizure_id}_preprocessed.pkl"
            file_path = self.raw_data_dir / pattern
            
            if not file_path.exists():
                # Try alternative patterns
                patterns = [
                    f"{subject_id}_{run_id}_{seizure_id}.pkl",
                    f"{subject_id}_{run_id}_seizure_{seizure_id.split('_')[-1]}_preprocessed.pkl"
                ]
                
                for alt_pattern in patterns:
                    alt_path = self.raw_data_dir / alt_pattern
                    if alt_path.exists():
                        file_path = alt_path
                        break
                else:
                    logger.warning(f"ECG data file not found for {subject_id}_{run_id}_{seizure_id}")
                    return None
            
            # Load data
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract ECG signal (assume it's the main array or in 'ecg' field)
            if isinstance(data, np.ndarray):
                return data
            elif isinstance(data, dict) and 'ecg' in data:
                return data['ecg']
            elif isinstance(data, dict) and 'signal' in data:
                return data['signal']
            else:
                logger.warning(f"Unknown data format in {file_path}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load ECG data for {subject_id}_{run_id}_{seizure_id}: {e}")
            return None
    
    def extract_all_features(self, cluster, madrid_results: Dict[str, Any], patient_history: Optional[Dict] = None) -> FeatureVector:
        """Extract all features for a cluster."""
        try:
            # Extract cluster-level features
            cluster_features = self.cluster_extractor.extract_features(cluster)
            
            # Extract signal-level features
            signal_features = {}
            if cluster.anomalies and self.raw_data_dir:
                rep_anomaly = cluster.anomalies[0]
                ecg_data = self.load_raw_ecg_data(
                    rep_anomaly.subject_id,
                    rep_anomaly.run_id, 
                    rep_anomaly.seizure_id
                )
                
                if ecg_data is not None:
                    # Define time window around cluster
                    cluster_times = [a.location_time_seconds for a in cluster.anomalies]
                    window_start = min(cluster_times) - 30  # 30s before
                    window_end = max(cluster_times) + 30    # 30s after
                    
                    signal_features = self.signal_extractor.extract_features(
                        ecg_data, 
                        (window_start, window_end)
                    )
                else:
                    signal_features = {name: 0.0 for name in self.signal_extractor.feature_names}
            else:
                signal_features = {name: 0.0 for name in self.signal_extractor.feature_names}
            
            # Extract contextual features
            contextual_features = self.contextual_extractor.extract_features(
                cluster, 
                madrid_results, 
                patient_history
            )
            
            # Combine all features
            all_features = {**cluster_features, **signal_features, **contextual_features}
            
            # Create feature vector
            feature_vector = FeatureVector(
                cluster_id=cluster.cluster_id,
                features=all_features,
                feature_names=self.feature_names,
                label=1 if cluster.has_seizure_hit else 0
            )
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction failed for cluster {cluster.cluster_id}: {e}")
            # Return zero features as fallback
            zero_features = {name: 0.0 for name in self.feature_names}
            return FeatureVector(
                cluster_id=cluster.cluster_id,
                features=zero_features,
                feature_names=self.feature_names,
                label=1 if cluster.has_seizure_hit else 0
            )
    
    def extract_features_batch(self, clusters: List, madrid_results: Dict[str, Any], patient_history: Optional[Dict] = None) -> List[FeatureVector]:
        """Extract features for multiple clusters."""
        feature_vectors = []
        
        logger.info(f"Extracting features for {len(clusters)} clusters...")
        
        for i, cluster in enumerate(clusters):
            if i % 10 == 0:
                logger.info(f"Processing cluster {i+1}/{len(clusters)}")
            
            feature_vector = self.extract_all_features(cluster, madrid_results, patient_history)
            feature_vectors.append(feature_vector)
        
        logger.info(f"Feature extraction completed. Generated {len(feature_vectors)} feature vectors.")
        return feature_vectors
    
    def save_features(self, feature_vectors: List[FeatureVector], output_path: str):
        """Save extracted features to file."""
        try:
            # Convert to DataFrame for easy saving
            data = []
            for fv in feature_vectors:
                row = {'cluster_id': fv.cluster_id, 'label': fv.label}
                row.update(fv.features)
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            
            logger.info(f"Features saved to {output_file}")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Positive samples: {df['label'].sum()}")
            logger.info(f"Negative samples: {len(df) - df['label'].sum()}")
            
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            raise

def main():
    """Example usage of feature extraction."""
    # This would typically be called from the main pipeline
    logger.info("Feature extraction module loaded successfully")
    logger.info(f"Available feature extractors:")
    logger.info(f"- Cluster features: {len(ClusterFeatureExtractor().feature_names)} features")
    logger.info(f"- Signal features: {len(SignalFeatureExtractor().feature_names)} features") 
    logger.info(f"- Contextual features: {len(ContextualFeatureExtractor().feature_names)} features")
    
    master_extractor = MasterFeatureExtractor()
    logger.info(f"Total features available: {len(master_extractor.feature_names)}")

if __name__ == "__main__":
    main()