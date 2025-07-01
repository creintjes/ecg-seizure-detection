import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, decimate
from typing import List, Tuple, Dict, Optional
import warnings
from pathlib import Path
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Information', 'Data', 'seizeit2-main'))

from classes.data import Data
from classes.annotation import Annotation


class SeizureOnlyECGPreprocessor:
    """
    ECG preprocessing pipeline for seizure-only segments from SeizeIT2 dataset.
    
    Extracts and processes only ECG segments containing seizures plus surrounding
    context (30 minutes before and after each seizure) to enable efficient
    algorithm testing and parameter optimization.
    """
    
    def __init__(
        self,
        filter_params: Dict[str, float] = None,
        downsample_freq: int = 125,
        context_minutes: int = 30
    ):
        """
        Initialize seizure-only ECG preprocessor.
        
        Args:
            filter_params: Dictionary with 'low_freq', 'high_freq', 'order'
            downsample_freq: Target sampling frequency in Hz
            context_minutes: Minutes before/after seizure to include
        """
        # Default bandpass filter parameters
        if filter_params is None:
            filter_params = {
                'low_freq': 0.5,    # Remove baseline drift
                'high_freq': 40.0,  # Remove high-frequency noise
                'order': 4          # Filter order
            }
        
        self.filter_params = filter_params
        self.downsample_freq = downsample_freq
        self.context_minutes = context_minutes
        self.context_seconds = context_minutes * 60  # Convert to seconds
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate preprocessing parameters."""
        if self.filter_params['low_freq'] >= self.filter_params['high_freq']:
            raise ValueError("Low frequency must be less than high frequency")
        
        if self.context_minutes <= 0:
            raise ValueError("Context minutes must be positive")
    
    def discover_seizure_segments(self, data_path: str) -> List[Tuple]:
        """
        Scan all recordings to identify seizure events and calculate extraction windows.
        
        Args:
            data_path: Path to SeizeIT2 dataset
            
        Returns:
            List of (subject_id, run_id, seizure_idx, seizure_start, seizure_end, 
                    extract_start, extract_end) tuples
        """
        data_path = Path(data_path)
        seizure_segments = []
        
        # Find all subjects
        subjects = [x for x in data_path.glob("sub-*") if x.is_dir()]
        print(f"Scanning {len(subjects)} subjects for seizures...")
        
        for subject_dir in subjects:
            subject_id = subject_dir.name
            
            # Look for ECG sessions
            ecg_dir = subject_dir / 'ses-01' / 'ecg'
            if not ecg_dir.exists():
                continue
                
            # Find all runs for this subject
            edf_files = list(ecg_dir.glob("*_ecg.edf"))
            
            for edf_file in edf_files:
                # Extract run ID from filename
                parts = edf_file.stem.split('_')
                run_part = [p for p in parts if p.startswith('run-')]
                
                if not run_part:
                    continue
                    
                run_id = run_part[0]
                recording = [subject_id, run_id]
                
                try:
                    # Load annotations to find seizures
                    annotations = Annotation.loadAnnotation(str(data_path), recording)
                    
                    if not annotations.events:
                        continue
                    
                    # Process each seizure in this recording
                    for seizure_idx, (seizure_start, seizure_end) in enumerate(annotations.events):
                        # Calculate extraction window (seizure ± context)
                        extract_start = max(0, seizure_start - self.context_seconds)
                        extract_end = min(annotations.rec_duration, seizure_end + self.context_seconds)
                        
                        # Ensure minimum window size
                        min_duration = 2 * self.context_seconds  # 60 minutes for 30min context
                        if extract_end - extract_start < min_duration:
                            # Expand window to minimum size if possible
                            center = (seizure_start + seizure_end) / 2
                            extract_start = max(0, center - min_duration / 2)
                            extract_end = min(annotations.rec_duration, center + min_duration / 2)
                        
                        seizure_segments.append((
                            subject_id, run_id, seizure_idx,
                            seizure_start, seizure_end,
                            extract_start, extract_end
                        ))
                        
                        print(f"  Found seizure: {subject_id} {run_id} #{seizure_idx} "
                              f"({seizure_start:.1f}-{seizure_end:.1f}s, "
                              f"extract: {extract_start:.1f}-{extract_end:.1f}s)")
                
                except Exception as e:
                    print(f"Warning: Could not load annotations for {subject_id} {run_id}: {e}")
                    continue
        
        print(f"Discovered {len(seizure_segments)} seizure segments")
        return seizure_segments
    
    def extract_seizure_segment(
        self, 
        data_path: str, 
        subject_id: str, 
        run_id: str, 
        extract_start: float, 
        extract_end: float
    ) -> Tuple[Data, Annotation]:
        """
        Extract specific time segment from ECG recording.
        
        Args:
            data_path: Path to SeizeIT2 dataset
            subject_id: Subject identifier
            run_id: Run identifier
            extract_start: Start time in seconds from recording beginning
            extract_end: End time in seconds from recording beginning
            
        Returns:
            Tuple of (Data segment, Annotation segment)
        """
        recording = [subject_id, run_id]
        
        # Load full data and annotations
        data = Data.loadData(data_path, recording, modalities=['ecg'])
        annotations = Annotation.loadAnnotation(data_path, recording)
        
        # Extract time segment from each channel
        extracted_data = []
        extracted_channels = []
        extracted_fs = []
        
        for i, (channel_data, channel_name, fs) in enumerate(
            zip(data.data, data.channels, data.fs)
        ):
            if 'ecg' not in channel_name.lower():
                continue
            
            # Calculate sample indices for extraction
            start_idx = int(extract_start * fs)
            end_idx = int(extract_end * fs)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(channel_data), end_idx)
            
            # Extract segment
            segment = channel_data[start_idx:end_idx]
            
            extracted_data.append(segment)
            extracted_channels.append(channel_name)
            extracted_fs.append(fs)
        
        # Create new Data object with extracted segments
        extracted_data_obj = Data(extracted_data, tuple(extracted_channels), tuple(extracted_fs))
        
        # Adjust annotation times relative to extraction start
        adjusted_events = []
        for seizure_start, seizure_end in annotations.events:
            # Only include seizures that overlap with extraction window
            if seizure_start < extract_end and seizure_end > extract_start:
                # Adjust times relative to extraction start
                adj_start = max(0, seizure_start - extract_start)
                adj_end = min(extract_end - extract_start, seizure_end - extract_start)
                adjusted_events.append((adj_start, adj_end))
        
        # Create new Annotation object with adjusted times
        extracted_annotations = Annotation(
            adjusted_events,
            annotations.types[:len(adjusted_events)] if annotations.types else [],
            annotations.lateralization[:len(adjusted_events)] if annotations.lateralization else [],
            annotations.localization[:len(adjusted_events)] if annotations.localization else [],
            annotations.vigilance[:len(adjusted_events)] if annotations.vigilance else [],
            extract_end - extract_start  # New duration
        )
        
        return extracted_data_obj, extracted_annotations
    
    def create_seizure_context_labels(
        self, 
        segment_duration: float,
        fs: int,
        seizure_events: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Create detailed labels distinguishing between seizure phases.
        
        Args:
            segment_duration: Duration of segment in seconds
            fs: Sampling frequency
            seizure_events: List of (start, end) seizure times in segment
            
        Returns:
            Sample-level labels array with seizure phase information
        """
        n_samples = int(segment_duration * fs)
        labels = np.full(n_samples, 'normal', dtype='<U12')
        
        # Mark pre-seizure, ictal, and post-seizure periods
        for seizure_start, seizure_end in seizure_events:
            # Convert times to sample indices
            seizure_start_idx = int(seizure_start * fs)
            seizure_end_idx = int(seizure_end * fs)
            
            # Pre-seizure context (30 minutes before)
            pre_start_idx = max(0, seizure_start_idx - int(self.context_seconds * fs))
            labels[pre_start_idx:seizure_start_idx] = 'pre_seizure'
            
            # Ictal period
            labels[seizure_start_idx:seizure_end_idx] = 'ictal'
            
            # Post-seizure context (30 minutes after)
            post_end_idx = min(n_samples, seizure_end_idx + int(self.context_seconds * fs))
            labels[seizure_end_idx:post_end_idx] = 'post_seizure'
        
        return labels
    
    def apply_bandpass_filter(self, signal_data: np.ndarray, fs: int) -> np.ndarray:
        """Apply bandpass filter to ECG signal."""
        # Calculate normalized frequencies
        nyquist = fs / 2
        low_norm = self.filter_params['low_freq'] / nyquist
        high_norm = self.filter_params['high_freq'] / nyquist
        
        # Validate frequency range
        if high_norm >= 1.0:
            high_norm = 0.99
            warnings.warn(f"High frequency adjusted to {high_norm * nyquist:.1f} Hz")
        
        # Design Butterworth bandpass filter
        b, a = butter(
            self.filter_params['order'],
            [low_norm, high_norm],
            btype='band'
        )
        
        # Apply zero-phase filtering
        filtered_signal = filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def downsample(self, signal_data: np.ndarray, original_fs: int) -> np.ndarray:
        """Downsample signal to target frequency."""
        if self.downsample_freq >= original_fs:
            warnings.warn("Target frequency >= original frequency, no downsampling")
            return signal_data
        
        # Calculate decimation factor
        decimation_factor = original_fs // self.downsample_freq
        
        if decimation_factor < 2:
            warnings.warn("Decimation factor < 2, using scipy.signal.resample")
            # Use resampling for non-integer factors
            n_samples = int(len(signal_data) * self.downsample_freq / original_fs)
            return signal.resample(signal_data, n_samples)
        
        # Use decimation with anti-aliasing
        downsampled_signal = decimate(signal_data, decimation_factor, ftype='iir')
        
        return downsampled_signal
    
    def preprocess_seizure_segment(
        self,
        data_path: str,
        subject_id: str,
        run_id: str,
        seizure_idx: int,
        seizure_start: float,
        seizure_end: float,
        extract_start: float,
        extract_end: float
    ) -> Dict:
        """
        Complete preprocessing pipeline for one seizure segment.
        
        Args:
            data_path: Path to SeizeIT2 dataset
            subject_id: Subject identifier
            run_id: Run identifier
            seizure_idx: Index of seizure in recording
            seizure_start: Original seizure start time
            seizure_end: Original seizure end time
            extract_start: Extraction window start time
            extract_end: Extraction window end time
            
        Returns:
            Dictionary with processed seizure segment data
        """
        try:
            # Extract seizure segment
            data, annotations = self.extract_seizure_segment(
                data_path, subject_id, run_id, extract_start, extract_end
            )
            
            if not data.data:
                raise ValueError("No ECG data found in segment")
            
            # Process each ECG channel
            processed_channels = []
            
            for i, (channel_data, channel_name, fs) in enumerate(
                zip(data.data, data.channels, data.fs)
            ):
                if 'ecg' not in channel_name.lower():
                    continue
                
                print(f"  Processing channel: {channel_name}")
                
                # Apply bandpass filter
                filtered_signal = self.apply_bandpass_filter(channel_data, int(fs))
                
                # Downsample
                downsampled_signal = self.downsample(filtered_signal, int(fs))
                
                # Update sampling frequency
                new_fs = min(self.downsample_freq, int(fs))
                
                # Create sample-level labels
                segment_duration_actual = len(downsampled_signal) / new_fs
                labels = self.create_seizure_context_labels(
                    segment_duration_actual, new_fs, annotations.events
                )
                assert len(labels) == len(downsampled_signal), (
                    f"Label/Data mismatch after fix: {len(labels)} vs {len(downsampled_signal)}"
                )

                # Create timestamps
                n_samples = len(downsampled_signal)
                timestamps = np.linspace(extract_start, extract_end, n_samples)
                
                # Count samples by phase
                n_ictal = np.sum(labels == 'ictal')
                n_pre_seizure = np.sum(labels == 'pre_seizure')
                n_post_seizure = np.sum(labels == 'post_seizure')
                
                channel_result = {
                    'name': channel_name,
                    'data': downsampled_signal,
                    'labels': labels,
                    'timestamps': timestamps,
                    'n_samples': n_samples,
                    'n_ictal_samples': n_ictal,
                    'n_pre_seizure_samples': n_pre_seizure,
                    'n_post_seizure_samples': n_post_seizure
                }
                
                processed_channels.append(channel_result)
            
            # Combine results
            result = {
                'subject_id': subject_id,
                'run_id': run_id,
                'seizure_index': seizure_idx,
                'original_seizure_start': seizure_start,
                'original_seizure_end': seizure_end,
                'extraction_start': extract_start,
                'extraction_end': extract_end,
                'sampling_rate': new_fs,
                'preprocessing_params': {
                    'filter_params': self.filter_params,
                    'downsample_freq': self.downsample_freq,
                    'context_minutes': self.context_minutes
                },
                'channels': processed_channels,
                'metadata': {
                    'total_duration': annotations.rec_duration,
                    'seizure_duration': seizure_end - seizure_start,
                    'context_duration': 2 * self.context_seconds
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing seizure segment {subject_id} {run_id} #{seizure_idx}: {str(e)}")
            return None