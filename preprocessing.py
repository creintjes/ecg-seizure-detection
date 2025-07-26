import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, decimate
from typing import List, Tuple, Dict, Optional
import warnings
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Information', 'Data', 'seizeit2_main'))

from classes.data import Data
from classes.annotation import Annotation


class ECGPreprocessor:
    """
    ECG preprocessing pipeline for SeizeIT2 dataset.
    
    Handles bandpass filtering, downsampling, and windowing of ECG signals
    for seizure detection using anomaly detection algorithms.
    """
    
    def __init__(
        self,
        filter_params: Dict[str, float] = None,
        downsample_freq: int = 125,
        window_size: float = 30.0,
        stride: float = 15.0,
        create_window:bool = True
    ):
        """
        Initialize ECG preprocessor.
        
        Args:
            filter_params: Dictionary with 'low_freq', 'high_freq', 'order'
            downsample_freq: Target sampling frequency in Hz
            window_size: Window duration in seconds
            stride: Step size between windows in seconds
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
        self.window_size = window_size
        self.stride = stride
        self.create_window = create_window
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate preprocessing parameters."""
        if self.filter_params['low_freq'] >= self.filter_params['high_freq']:
            raise ValueError("Low frequency must be less than high frequency")
        
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("Window size and stride must be positive")
        
        if self.stride > self.window_size:
            warnings.warn("Stride is larger than window size - gaps between windows")
    
    def load_data(
        self, 
        data_path: str, 
        subject_id: str, 
        run_id: str
    ) -> Tuple[Data, Annotation]:
        """
        Load ECG data and annotations for a specific recording.
        
        Args:
            data_path: Path to SeizeIT2 dataset
            subject_id: Subject identifier (e.g., 'sub-001')
            run_id: Run identifier (e.g., 'run-01')
            
        Returns:
            Tuple of (Data object, Annotation object)
        """
        recording = [subject_id, run_id]
        
        # Load data with ECG modality
        data = Data.loadData(data_path, recording, modalities=['ecg'])
        
        # Load annotations
        annotations = Annotation.loadAnnotation(data_path, recording)
        
        return data, annotations
    
    def apply_bandpass_filter(self, signal_data: np.ndarray, fs: int) -> np.ndarray:
        """
        Apply bandpass filter to ECG signal.
        
        Args:
            signal_data: ECG signal data
            fs: Sampling frequency
            
        Returns:
            Filtered signal
        """
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
    
    def z_normalize(signal: np.ndarray) -> np.ndarray:
        """
        Apply global z-score normalization to the signal.

        Parameters:
            signal (np.ndarray): Input time series signal.

        Returns:
            np.ndarray: Z-normalized signal with mean 0 and std 1.
        """
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            raise ValueError("Standard deviation is zero; cannot normalize.")
        return (signal - mean) / std

    def downsample(self, signal_data: np.ndarray, original_fs: int) -> np.ndarray:
        """
        Downsample signal to target frequency.
        
        Args:
            signal_data: Input signal
            original_fs: Original sampling frequency
            
        Returns:
            Downsampled signal
        """
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
    
    def create_windows(
        self, 
        signal_data: np.ndarray, 
        fs: int,
        annotations: Annotation
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Create overlapping windows from signal data with sample-level seizure annotations.
        
        Args:
            signal_data: Preprocessed ECG signal
            fs: Sampling frequency
            annotations: Seizure annotations
            
        Returns:
            Tuple of (windows, sample_labels, metadata)
            - windows: List of signal windows
            - sample_labels: List of sample-level labels (0/1 arrays for each window)
            - metadata: List of window metadata including summary statistics
        """
        # Calculate window parameters in samples
        window_samples = int(self.window_size * int(fs))
        stride_samples = int(self.stride * int(fs))
        
        # Calculate number of windows
        signal_length = len(signal_data)
        n_windows = (signal_length - window_samples) // stride_samples + 1
        
        if n_windows <= 0:
            warnings.warn("Signal too short for windowing")
            return [], [], []
        
        windows = []
        sample_labels = []
        metadata = []
        
        # Extract seizure events timing
        seizure_events = annotations.events if annotations.events else []
        
        # Create full-signal sample-level labels first
        full_signal_labels = self._create_sample_level_labels(signal_length, int(fs), seizure_events)
        
        for i in range(n_windows):
            start_idx = i * stride_samples
            end_idx = start_idx + window_samples
            
            # Extract window
            window = signal_data[start_idx:end_idx]
            windows.append(window)
            
            # Extract sample-level labels for this window
            window_sample_labels = full_signal_labels[start_idx:end_idx]
            sample_labels.append(window_sample_labels)
            
            # Calculate window timing in seconds
            start_time = start_idx / int(fs)
            end_time = end_idx / int(fs)
            
            # Calculate summary statistics
            n_seizure_samples = np.sum(window_sample_labels)
            seizure_ratio = n_seizure_samples / len(window_sample_labels)
            window_label = 1 if n_seizure_samples > 0 else 0
            
            # Store metadata
            meta = {
                'start_time': start_time,
                'end_time': end_time,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'sampling_rate': int(fs),
                'window_label': window_label,
                'n_seizure_samples': int(n_seizure_samples),
                'seizure_ratio': float(seizure_ratio),
                'seizure_segments': self._get_seizure_segments_in_window(window_sample_labels, start_time, int(fs))
            }
            metadata.append(meta)
        
        return windows, sample_labels, metadata
    
    def _create_sample_level_labels(self, signal_length: int, fs: int, seizure_events: List[Tuple[float, float]]) -> np.ndarray:
        """
        Create sample-level labels for the entire signal.
        
        Args:
            signal_length: Length of signal in samples
            fs: Sampling frequency
            seizure_events: List of (start_time, end_time) seizure events in seconds
            
        Returns:
            Array of 0/1 labels for each sample
        """
        labels = np.zeros(signal_length, dtype=int)
        
        for seizure_start, seizure_end in seizure_events:
            # Convert times to sample indices
            start_sample = int(seizure_start * fs)
            end_sample = int(seizure_end * fs)
            
            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(signal_length, end_sample)
            
            if start_sample < end_sample:
                labels[start_sample:end_sample] = 1
        
        return labels
    
    def _get_seizure_segments_in_window(self, window_labels: np.ndarray, window_start_time: float, fs: int) -> List[Dict]:
        """
        Extract seizure segments within a window.
        
        Args:
            window_labels: Sample-level labels for the window
            window_start_time: Start time of the window in seconds
            fs: Sampling frequency
            
        Returns:
            List of seizure segments with timing information
        """
        segments = []
        
        # Find continuous seizure regions
        diff = np.diff(np.concatenate(([0], window_labels, [0])))
        seizure_starts = np.where(diff == 1)[0]
        seizure_ends = np.where(diff == -1)[0]
        
        for start_idx, end_idx in zip(seizure_starts, seizure_ends):
            segment = {
                'start_sample_in_window': int(start_idx),
                'end_sample_in_window': int(end_idx),
                'start_time_absolute': float(window_start_time + start_idx / fs),
                'end_time_absolute': float(window_start_time + end_idx / fs),
                'duration_seconds': float((end_idx - start_idx) / fs),
                'n_samples': int(end_idx - start_idx)
            }
            segments.append(segment)
        
        return segments

    def _get_no_window_labels(self, length: float, frequency: int, intervals: list[list[float]]) -> np.ndarray:
        """
        Generates a 1D NumPy array of zeros with specified length and frequency.
        Regions defined by pairs of float start-end intervals (converted to integer sample indices) are set to 1.

        Parameters:
        ----------
        length : float
            Total length of the signal in seconds.
        frequency : float
            Sampling frequency in Hz.
        intervals : list[list[float]]
            A list of [start, end] interval pairs as float values, specifying sample indices (e.g. [[10.0, 20.0], [30.0, 40.0]]).
            The float values should represent integers (e.g., 10.0, not 10.3).

        Returns:
        -------
        np.ndarray
            A 1D NumPy array of shape (int(length * frequency),) with 1s in specified intervals.
        """
        total_samples = int(length * frequency)
        signal = np.zeros(total_samples, dtype=int)

        for pair in intervals:
            if len(pair) != 2:
                raise ValueError(f"Each interval must contain exactly two elements. Got: {pair}")
            start = int(pair[0])*frequency
            end = int(pair[1])*frequency
            if not (0 <= start <= end <= total_samples):
                raise ValueError(f"Invalid interval range: {start} to {end}. Must be within [0, {total_samples}].")
            signal[start:end] = 1

        return signal

    def create_no_windows(
        self, 
        signal_data: np.ndarray, 
        fs: int,
        annotations: Annotation
    ) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """
        Parse the signal data.
        
        Args:
            signal_data: Preprocessed ECG signal
            fs: Sampling frequency
            annotations: Seizure annotations
            
        Returns:
            Tuple of (windows, labels, metadata)
        """
        # Calculate window parameters in samples
        # window_samples = int(self.window_size * int(fs))
        # stride_samples = int(self.stride * int(fs))
        
        # Calculate number of windows
        # signal_length = len(signal_data)
        # n_windows = (signal_length - window_samples) // stride_samples + 1
        
        # if n_windows <= 0:
        #     warnings.warn("Signal too short for windowing")
        #     return [], [], []
        
        windows = []
        labels = []
        metadata = []
        
        # Extract seizure events timing
        seizure_events = annotations.events if annotations.events else []
        
        # for i in range(n_windows):
        # start_idx = i * stride_samples
        # end_idx = start_idx + window_samples
        
        # Extract window
        # window = signal_data[start_idx:end_idx]
        windows.append(signal_data)
        
        # Calculate window timing in seconds
        start_time = 0
        end_time = float(annotations.rec_duration)
        
        # Determine label (1 for seizure, 0 for normal)
        label = self._get_no_window_labels(end_time, self.downsample_freq, seizure_events)
        labels.append(label)
        
        # Store metadata
        meta = {
            'start_time': start_time,
            'end_time': end_time,
            'start_idx': 0,
            'end_idx': end_time*self.downsample_freq,
            'sampling_rate': int(self.downsample_freq)
        }
        metadata.append(meta)
        
        return windows, labels, metadata
    def _get_window_label(
        self, 
        start_time: float, 
        end_time: float, 
        seizure_events: List[Tuple[float, float]]
    ) -> int:
        """
        Determine if window contains seizure activity.
        
        Args:
            start_time: Window start time in seconds
            end_time: Window end time in seconds
            seizure_events: List of (start, end) seizure times
            
        Returns:
            1 if seizure window, 0 if normal
        """
        for seizure_start, seizure_end in seizure_events:
            # Check for overlap between window and seizure
            if (start_time < seizure_end) and (end_time > seizure_start):
                return 1
        return 0
    
    def preprocess_pipeline(
        self,
        data_path: str,
        subject_id: str,
        run_id: str
    ) -> Dict:
        """
        Complete preprocessing pipeline for one recording.
        
        Args:
            data_path: Path to SeizeIT2 dataset
            subject_id: Subject identifier
            run_id: Run identifier
            
        Returns:
            Dictionary with processed data, labels, and metadata
        """
        try:
            # Load data
            data, annotations = self.load_data(data_path, subject_id, run_id)
            
            if not data.data:
                raise ValueError("No ECG data found")
            
            # Process each ECG channel
            processed_channels = []
            
            for i, (channel_data, channel_name, fs) in enumerate(
                zip(data.data, data.channels, data.fs)
            ):
                if 'ecg' not in channel_name.lower():
                    continue
                
                print(f"Processing channel: {channel_name}")
                
                # Apply bandpass filter
                filtered_signal = self.apply_bandpass_filter(channel_data, int(fs))
                
                # Downsample
                downsampled_signal = self.downsample(filtered_signal, int(fs))
                
                # Update sampling frequency
                new_fs = min(self.downsample_freq, int(fs))
                
                # Create windows
                if self.create_window:
                    windows, labels, metadata = self.create_windows(
                        downsampled_signal, new_fs, annotations
                    )
                else:
                    windows, labels, metadata = self.create_no_windows(
                        downsampled_signal, new_fs, annotations
                    )
                
                # Calculate seizure windows count from metadata
                n_seizure_windows = sum(1 for meta in metadata if meta.get('window_label', 0) == 1)
                
                channel_result = {
                    'channel_name': channel_name,
                    'windows': windows,
                    'labels': labels,  # Now contains sample-level labels (arrays) instead of binary window labels
                    'metadata': metadata,
                    'original_fs': int(fs),
                    'processed_fs': new_fs,
                    'n_windows': len(windows),
                    'n_seizure_windows': n_seizure_windows
                }
                
                processed_channels.append(channel_result)
            
            # Combine results
            result = {
                'subject_id': subject_id,
                'run_id': run_id,
                'channels': processed_channels,
                'preprocessing_params': {
                    'filter_params': self.filter_params,
                    'downsample_freq': self.downsample_freq,
                    'window_size': self.window_size,
                    'stride': self.stride
                },
                'recording_duration': annotations.rec_duration,
                'total_seizures': len(annotations.events) if annotations.events else 0
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {subject_id} {run_id}: {str(e)}")
            return None
    
    def batch_preprocess(
        self,
        data_path: str,
        recordings: List[Tuple[str, str]],
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Process multiple recordings in batch.
        
        Args:
            data_path: Path to SeizeIT2 dataset
            recordings: List of (subject_id, run_id) tuples
            save_path: Optional path to save results
            
        Returns:
            List of processed recording dictionaries
        """
        results = []
        
        for subject_id, run_id in recordings:
            print(f"Processing {subject_id} {run_id}...")
            
            result = self.preprocess_pipeline(data_path, subject_id, run_id)
            
            if result is not None:
                results.append(result)
                
                if save_path:
                    # Save individual result
                    filename = f"{subject_id}_{run_id}_preprocessed.pkl"
                    filepath = Path(save_path) / filename
                    pd.to_pickle(result, filepath)
        
        print(f"Successfully processed {len(results)}/{len(recordings)} recordings")
        
        return results