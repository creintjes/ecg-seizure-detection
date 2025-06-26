"""
MADRID v2 - Multi-Length Anomaly Detection with Irregular Discords
Enhanced Python implementation based on MATLAB MADRID_2_0.m

This module provides an enhanced Python implementation of the MADRID algorithm 
that follows the original MATLAB logic more closely, enabling detection of 
multiple anomalies rather than just the best per length.

Key improvements over madrid.py:
- Complete matrix profile computation following MATLAB logic
- Multiple anomaly detection within each length
- Enhanced DAMP algorithm with proper pruning
- Better anomaly extraction that can find multiple anomalies per length

Dependencies:
    - numpy
    - scipy
    - matplotlib (for plotting)
    - cupy (optional, for GPU acceleration)

Usage Examples:
    
    # Basic usage
    from models.madrid_v2 import MADRID_V2
    
    # Generate sample data
    import numpy as np
    T = np.random.randn(10000)  # Your time series data
    
    # Run MADRID with default parameters
    madrid = MADRID_V2(use_gpu=True)  # Enable GPU if available
    multi_length_table, bsf, bsf_loc = madrid.fit(
        T=T,
        min_length=64,
        max_length=256,
        step_size=32,
        train_test_split=5000
    )
    
    # Get multiple anomalies
    anomaly_info = madrid.get_anomaly_scores(threshold_percentile=90)
    print(f"Found {len(anomaly_info['anomalies'])} anomalies")
    
    # For ECG seizure detection
    # Assuming 32Hz sampling rate
    ecg_data = load_ecg_data()  # Your ECG loading function
    
    # Configure for seizure detection (10-100 seconds at 32Hz)
    madrid_ecg = MADRID_V2(use_gpu=True, enable_output=True)
    results = madrid_ecg.fit(
        T=ecg_data,
        min_length=320,    # 10 seconds
        max_length=3200,   # 100 seconds
        step_size=320,     # 10 second steps
        train_test_split=len(ecg_data)//3
    )
    
    # Get multiple anomalies
    anomalies = madrid_ecg.get_anomaly_scores(threshold_percentile=85)
    for i, anomaly in enumerate(anomalies['anomalies']):
        print(f"Anomaly {i+1}: Score={anomaly['score']:.3f}, "
              f"Location={anomaly['location']}, Length={anomaly['length']}")

Author: Generated for enhanced anomaly detection
Date: 2025
Version: 2.0
"""

import numpy as np
import time
import warnings
from typing import Union, Tuple, Optional, Dict, List, Any
from scipy.fft import fft, ifft

# Optional GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# Optional plotting support
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None

warnings.filterwarnings('ignore')


class MADRID_V2:
    """
    Enhanced MADRID (Multi-Length Anomaly Detection with Irregular Discords) implementation
    
    This version follows the MATLAB MADRID_2_0.m logic more closely to enable
    detection of multiple anomalies per length, not just the best one.
    """
    
    def __init__(self, use_gpu: bool = True, enable_output: bool = True):
        """
        Initialize MADRID v2 detector
        
        Args:
            use_gpu: Whether to use GPU acceleration (requires cupy)
            enable_output: Whether to print progress information
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.enable_output = enable_output
        
        # Initialize computation backend
        if self.use_gpu:
            self.xp = cp
            if self.enable_output:
                print("✓ GPU acceleration enabled (CuPy)")
        else:
            self.xp = np
            if self.enable_output:
                print("✓ Using CPU computation (NumPy)")
        
        # Storage for results
        self.last_results = None
        self.last_time_series = None
        self.last_params = None
        self.last_multi_length_table = None  # Store complete table for enhanced anomaly extraction
        
    def _to_device(self, array: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Move array to appropriate device"""
        if self.use_gpu and isinstance(array, np.ndarray):
            return cp.asarray(array)
        elif not self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
        
    def _to_cpu(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Move array to CPU"""
        if cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    
    def contains_constant_regions(self, T: np.ndarray, min_length: int) -> bool:
        """
        Check if time series contains constant regions that would make anomaly detection trivial
        
        Args:
            T: Time series
            min_length: Minimum subsequence length
            
        Returns:
            True if constant regions exist
        """
        # Find consecutive identical values
        diff_T = np.diff(T)
        constant_starts = np.where(np.abs(diff_T) < 1e-10)[0]
        
        if len(constant_starts) == 0:
            return False
            
        # Check for regions longer than min_length
        consecutive_groups = np.split(constant_starts, 
                                    np.where(np.diff(constant_starts) != 1)[0] + 1)
        
        max_constant_length = max(len(group) + 1 for group in consecutive_groups)
        
        # Also check overall variance
        if np.var(T) < 0.2 or max_constant_length >= min_length:
            return True
            
        return False
    
    def mass_v2(self, x: Union[np.ndarray, 'cp.ndarray'], 
                y: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        MASS v2 (Mueen's Algorithm for Similarity Search) - Enhanced version
        
        Computes the distance profile between query y and all subsequences in x
        using z-normalized Euclidean distance via FFT.
        
        Args:
            x: Time series data (longer sequence)
            y: Query subsequence
            
        Returns:
            Distance profile
        """
        x = self._to_device(x)
        y = self._to_device(y)
        
        m = len(y)
        n = len(x)
        
        if m > n:
            return self.xp.array([])
        
        # Compute query statistics
        mean_y = self.xp.mean(y)
        std_y = self.xp.std(y)
        
        if std_y == 0:
            # Handle constant query
            return self.xp.full(n - m + 1, self.xp.inf)
        
        # Compute sliding window statistics for x
        if self.use_gpu:
            # GPU version using CuPy
            x_padded = self.xp.pad(x, (m-1, 0), mode='constant', constant_values=0)
            rolling_sum = self.xp.convolve(x_padded, self.xp.ones(m), mode='valid')
            rolling_sq_sum = self.xp.convolve(x_padded**2, self.xp.ones(m), mode='valid')
        else:
            # CPU version using numpy
            rolling_sum = np.convolve(x, np.ones(m), mode='valid')
            rolling_sq_sum = np.convolve(x**2, np.ones(m), mode='valid')
        
        mean_x = rolling_sum / m
        var_x = (rolling_sq_sum / m) - (mean_x**2)
        var_x = self.xp.maximum(var_x, 1e-10)  # Avoid division by zero
        std_x = self.xp.sqrt(var_x)
        
        # Reverse query for convolution
        y_rev = y[::-1]
        
        # Zero-pad for FFT
        y_padded = self.xp.zeros(n)
        y_padded[:m] = y_rev
        
        # Compute dot products using FFT
        X = fft(x.astype(complex))
        Y = fft(y_padded.astype(complex))
        Z = X * Y
        dot_products = ifft(Z).real[m-1:n]
        
        # Compute z-normalized Euclidean distances
        numerator = 2 * (m - (dot_products - m * mean_x * mean_y) / (std_x * std_y))
        distances = self.xp.sqrt(self.xp.maximum(numerator, 0))
        
        return distances
    
    def fit(self, T: np.ndarray, min_length: int, max_length: int, 
            step_size: int, train_test_split: int, factor: int = 1) -> Tuple:
        """
        Fit MADRID model to time series data
        
        Args:
            T: Time series data
            min_length: Minimum subsequence length to search
            max_length: Maximum subsequence length to search  
            step_size: Step size between lengths
            train_test_split: Index where test data begins
            factor: Downsampling factor (1 = no downsampling)
            
        Returns:
            Tuple of (multi_length_table, bsf, bsf_loc)
        """
        # Validate inputs
        if len(T) < max_length:
            raise ValueError(f"Time series length ({len(T)}) must be >= max_length ({max_length})")
        
        if train_test_split >= len(T):
            raise ValueError(f"train_test_split ({train_test_split}) must be < time series length ({len(T)})")
        
        # Check for constant regions
        if self.contains_constant_regions(T, min_length):
            raise ValueError(
                "Time series contains constant regions that would make anomaly detection trivial. "
                "Consider: 1) Increasing min_length, 2) Adding noise, 3) Adding linear trend, "
                "4) Removing constant sections"
            )
        
        # Apply downsampling if needed
        if factor > 1:
            T_sampled = T[::factor]
            train_test_split_sampled = train_test_split // factor
            min_length_sampled = max(2, min_length // factor)
            max_length_sampled = max(min_length_sampled + 1, max_length // factor)
            step_size_sampled = max(1, step_size // factor)
        else:
            T_sampled = T.copy()
            train_test_split_sampled = train_test_split
            min_length_sampled = min_length
            max_length_sampled = max_length
            step_size_sampled = step_size
        
        if self.enable_output:
            print(f"MADRID v2 Processing:")
            print(f"  Time series length: {len(T_sampled)}")
            print(f"  Length range: {min_length_sampled}-{max_length_sampled} (step: {step_size_sampled})")
            print(f"  Train/test split: {train_test_split_sampled}")
            print(f"  Downsampling factor: {factor}")
        
        # Run enhanced MADRID algorithm
        start_time = time.time()
        multi_length_table, bsf, bsf_loc = self._madrid_core_v2(
            T_sampled, min_length_sampled, max_length_sampled, 
            step_size_sampled, train_test_split_sampled
        )
        execution_time = time.time() - start_time
        
        if self.enable_output:
            print(f"MADRID v2 execution time: {execution_time:.2f} seconds")
        
        # Store results
        self.last_results = (multi_length_table, bsf, bsf_loc)
        self.last_multi_length_table = multi_length_table  # Store complete table
        self.last_time_series = T_sampled
        self.last_params = {
            'min_length': min_length_sampled,
            'max_length': max_length_sampled,
            'step_size': step_size_sampled,
            'train_test_split': train_test_split_sampled,
            'factor': factor,
            'original_step_size': step_size
        }
        
        # Convert back to CPU arrays
        return (self._to_cpu(multi_length_table), 
                self._to_cpu(bsf), 
                self._to_cpu(bsf_loc))
    
    def _madrid_core_v2(self, T: np.ndarray, min_length: int, max_length: int,
                       step_size: int, train_test_split: int) -> Tuple:
        """
        Enhanced MADRID core algorithm following MATLAB logic
        
        This implementation builds the complete multi-length table and uses
        the strategic initialization approach from the MATLAB version.
        """
        T = self._to_device(T)
        n = len(T)
        
        # Initialize data structures
        m_set = list(range(min_length, max_length + 1, step_size))
        num_lengths = len(m_set)
        
        if self.enable_output:
            print(f"  Processing {num_lengths} different lengths: {m_set}")
        
        # Initialize multi-length table and BSF arrays
        multi_length_table = self.xp.full((num_lengths, n), -self.xp.inf)
        bsf = self.xp.zeros(num_lengths)
        bsf_loc = self.xp.full(num_lengths, self.xp.nan)
        
        # Strategic initialization: Start with middle length (MATLAB approach)
        mid_idx = len(m_set) // 2
        m_mid = m_set[mid_idx]
        
        if self.enable_output:
            print(f"  Starting with middle length: {m_mid}")
        
        # Process middle length first
        discord_score, position, left_mp = self._damp_v2(T, m_mid, train_test_split)
        normalization = 1.0 / (2 * self.xp.sqrt(m_mid))
        multi_length_table[mid_idx, :] = left_mp * normalization
        bsf[mid_idx] = discord_score * normalization
        bsf_loc[mid_idx] = position
        
        # Process other lengths using the discovered position
        for idx, m in enumerate(m_set):
            if idx == mid_idx:
                continue  # Already processed
                
            if self.enable_output and idx % max(1, num_lengths // 5) == 0:
                print(f"  Processing length {m} ({idx+1}/{num_lengths})")
            
            # Use position from middle length as starting point (MATLAB approach)
            discord_score, best_position, left_mp = self._damp_v2_with_hint(
                T, m, train_test_split, position
            )
            
            # Normalize scores
            normalization = 1.0 / (2 * self.xp.sqrt(m))
            multi_length_table[idx, :] = left_mp * normalization
            bsf[idx] = discord_score * normalization
            bsf_loc[idx] = best_position
        
        return multi_length_table, bsf, bsf_loc
    
    def _damp_v2(self, T: Union[np.ndarray, 'cp.ndarray'], 
                subseq_length: int, train_test_split: int) -> Tuple:
        """
        Enhanced DAMP implementation following MATLAB DAMP_2_0 logic
        """
        T = self._to_device(T)
        n = len(T)
        
        # Initialize left matrix profile
        left_mp = self.xp.full(n, -self.xp.inf)
        left_mp[:train_test_split] = self.xp.nan
        
        best_discord_score = -self.xp.inf
        best_position = train_test_split
        
        # Pruning vector for DAMP optimization
        bool_vec = self.xp.ones(n, dtype=bool)
        lookahead = 2**int(self.xp.ceil(self.xp.log2(16 * subseq_length)))
        
        # Process prefix to get good initial BSF (MATLAB approach)
        prefix_end = min(train_test_split + 16 * subseq_length, n - subseq_length)
        
        for i in range(train_test_split, prefix_end + 1):
            if not bool_vec[i] or i + subseq_length > n:
                if i < n:
                    left_mp[i] = left_mp[i-1] - 0.00001 if i > 0 else -self.xp.inf
                continue
            
            query = T[i:i + subseq_length]
            
            # Compute distance profile for training data
            if i > subseq_length:
                distances = self.mass_v2(T[:i], query)
                if len(distances) > 0:
                    min_distance = self.xp.min(distances)
                    left_mp[i] = min_distance
                    
                    # Update best discord
                    if min_distance > best_discord_score:
                        best_discord_score = min_distance
                        best_position = i
            
            # Forward pruning (lookahead)
            if lookahead > 0:
                self._forward_pruning(T, i, subseq_length, query, 
                                    best_discord_score, lookahead, bool_vec)
        
        # Process remaining test data with full DAMP
        for i in range(prefix_end + 1, n - subseq_length + 1):
            if not bool_vec[i]:
                left_mp[i] = left_mp[i-1] - 0.00001 if i > 0 else -self.xp.inf
                continue
            
            query = T[i:i + subseq_length]
            
            # Use DAMP progressive search strategy
            min_distance = self._damp_progressive_search(T, i, subseq_length, query, best_discord_score)
            left_mp[i] = min_distance
            
            # Update best discord
            if min_distance > best_discord_score:
                best_discord_score = min_distance
                best_position = i
            
            # Forward pruning
            if lookahead > 0:
                self._forward_pruning(T, i, subseq_length, query, 
                                    best_discord_score, lookahead, bool_vec)
        
        return best_discord_score, best_position, left_mp
    
    def _damp_v2_with_hint(self, T: Union[np.ndarray, 'cp.ndarray'], 
                          subseq_length: int, train_test_split: int, hint_position: int) -> Tuple:
        """
        DAMP with hint position (MATLAB strategic approach)
        """
        # First run normal DAMP
        discord_score, position, left_mp = self._damp_v2(T, subseq_length, train_test_split)
        
        # If hint position is different, explore it
        if hint_position != position and hint_position + subseq_length <= len(T):
            query = T[hint_position:hint_position + subseq_length]
            distances = self.mass_v2(T[:hint_position], query)
            if len(distances) > 0:
                hint_score = self.xp.min(distances)
                left_mp[hint_position] = hint_score
                
                # Update BSF if hint is better
                if hint_score > discord_score:
                    discord_score = hint_score
                    position = hint_position
        
        return discord_score, position, left_mp
    
    def _damp_progressive_search(self, T: Union[np.ndarray, 'cp.ndarray'], 
                               position: int, subseq_length: int, 
                               query: Union[np.ndarray, 'cp.ndarray'], 
                               best_so_far: float) -> float:
        """
        DAMP progressive backward search strategy from MATLAB
        """
        # Initialize search parameters
        X = 2**int(self.xp.ceil(self.xp.log2(8 * subseq_length)))
        expansion_num = 0
        approximate_distance = self.xp.inf
        first_iteration = True
        
        while approximate_distance >= best_so_far:
            # Check if we've reached the beginning
            search_start = position - X + 1 + (expansion_num * subseq_length)
            if search_start < 1:
                # Final search from beginning
                distances = self.mass_v2(T[:position], query)
                if len(distances) > 0:
                    approximate_distance = self.xp.min(distances)
                break
            else:
                if first_iteration:
                    # Search closest segment first
                    search_end = position
                    search_data = T[max(0, search_start):search_end]
                    first_iteration = False
                else:
                    # Search farther segments
                    search_end = position - (X // 2) + (expansion_num * subseq_length)
                    search_data = T[max(0, search_start):max(0, search_end)]
                
                if len(search_data) >= subseq_length:
                    distances = self.mass_v2(search_data, query)
                    if len(distances) > 0:
                        approximate_distance = self.xp.min(distances)
                    else:
                        approximate_distance = self.xp.inf
                else:
                    approximate_distance = self.xp.inf
                
                if approximate_distance < best_so_far:
                    break
                else:
                    # Expand search
                    X = 2 * X
                    expansion_num += 1
        
        return approximate_distance
    
    def _forward_pruning(self, T: Union[np.ndarray, 'cp.ndarray'], 
                        position: int, subseq_length: int, 
                        query: Union[np.ndarray, 'cp.ndarray'],
                        best_so_far: float, lookahead: int, 
                        bool_vec: Union[np.ndarray, 'cp.ndarray']):
        """
        Forward pruning for DAMP optimization
        """
        n = len(T)
        start_mass = min(position + subseq_length, n)
        end_mass = min(start_mass + lookahead, n)
        
        if end_mass - start_mass >= subseq_length:
            search_data = T[start_mass:end_mass]
            distances = self.mass_v2(search_data, query)
            
            if len(distances) > 0:
                # Find positions with distance < best_so_far
                prune_indices = self.xp.where(distances < best_so_far)[0]
                # Convert to global indices
                global_indices = prune_indices + start_mass
                # Update pruning vector
                valid_indices = global_indices[global_indices < n]
                bool_vec[valid_indices] = False
    
    def get_anomaly_scores(self, threshold_percentile: float = 95, 
                          max_anomalies_per_length: int = 5) -> dict:
        """
        Enhanced anomaly extraction that can find multiple anomalies per length
        
        Args:
            threshold_percentile: Percentile threshold for anomaly detection
            max_anomalies_per_length: Maximum anomalies to extract per length
            
        Returns:
            Dictionary with comprehensive anomaly information
        """
        if self.last_results is None or self.last_multi_length_table is None:
            raise ValueError("No results available. Run fit() first.")
        
        multi_length_table, bsf, bsf_loc = self.last_results
        params = self.last_params
        
        # Calculate global threshold across all lengths and positions
        all_valid_scores = []
        for row_idx in range(multi_length_table.shape[0]):
            row = multi_length_table[row_idx, :]
            valid_scores = row[~np.isinf(row) & ~np.isnan(row)]
            if len(valid_scores) > 0:
                all_valid_scores.extend(valid_scores)
        
        if len(all_valid_scores) == 0:
            return {'anomalies': [], 'threshold': 0, 'all_scores': bsf}
        
        threshold = np.percentile(all_valid_scores, threshold_percentile)
        
        # Extract anomalies from each length
        all_anomalies = []
        m_values = list(range(params['min_length'], params['max_length'] + 1, params['step_size']))
        
        for row_idx, m_value in enumerate(m_values):
            row = multi_length_table[row_idx, :]
            
            # Find all positions above threshold for this length
            valid_mask = ~np.isinf(row) & ~np.isnan(row) & (row >= threshold)
            if not np.any(valid_mask):
                continue
            
            # Get scores and positions above threshold
            valid_positions = np.where(valid_mask)[0]
            valid_scores = row[valid_positions]
            
            # Sort by score (descending)
            sort_indices = np.argsort(valid_scores)[::-1]
            valid_positions = valid_positions[sort_indices]
            valid_scores = valid_scores[sort_indices]
            
            # Extract top anomalies for this length (with exclusion zones)
            extracted_count = 0
            used_positions = set()
            exclusion_radius = max(1, m_value // 4)  # Exclusion zone to avoid overlaps
            
            for pos, score in zip(valid_positions, valid_scores):
                if extracted_count >= max_anomalies_per_length:
                    break
                
                # Check if this position overlaps with already selected anomalies
                too_close = any(abs(pos - used_pos) < exclusion_radius for used_pos in used_positions)
                if too_close:
                    continue
                
                # Add this anomaly
                all_anomalies.append({
                    'score': float(score),
                    'location': int(pos),
                    'length': m_value,
                    'length_index': row_idx,
                    'time_seconds': float(pos) / (250 / params['factor']) if 'factor' in params else float(pos),  # Assuming 250Hz original
                    'normalized_score': float(score),
                    'confidence': min(float(score / np.max(all_valid_scores)), 1.0),
                    'anomaly_id': f"m{m_value}_loc{pos}_score{score:.3f}"
                })
                
                used_positions.add(pos)
                extracted_count += 1
        
        # Sort all anomalies by score (descending)
        all_anomalies.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank information
        for rank, anomaly in enumerate(all_anomalies):
            anomaly['rank'] = rank + 1
            anomaly['seizure_hit'] = False  # Will be updated by validation logic
        
        return {
            'anomalies': all_anomalies,
            'threshold': float(threshold),
            'all_scores': bsf,
            'total_anomalies_found': len(all_anomalies),
            'anomalies_per_length': {
                m_val: len([a for a in all_anomalies if a['length'] == m_val])
                for m_val in m_values
            }
        }
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Enhanced plotting of MADRID v2 results
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available for plotting")
            return
            
        if self.last_results is None:
            print("No results to plot. Run fit() first.")
            return
        
        multi_length_table, bsf, bsf_loc = self.last_results
        T = self.last_time_series
        params = self.last_params
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('MADRID v2 Results', fontsize=16)
        
        # Plot 1: Time series with anomalies
        axes[0, 0].plot(T, 'b-', alpha=0.7, label='Time Series')
        if params['train_test_split'] < len(T):
            axes[0, 0].axvline(x=params['train_test_split'], color='r', 
                              linestyle='--', label='Train/Test Split')
        
        # Get anomalies and mark them
        try:
            anomaly_info = self.get_anomaly_scores(threshold_percentile=90)
            for i, anomaly in enumerate(anomaly_info['anomalies'][:10]):  # Top 10
                color = 'red' if i < 3 else 'orange'
                axes[0, 0].axvline(x=anomaly['location'], color=color, alpha=0.7, 
                                  linewidth=2, label='Top Anomaly' if i == 0 else '')
        except:
            # Fallback to BSF marking
            for i, (score, loc) in enumerate(zip(bsf, bsf_loc)):
                if not np.isnan(loc) and score > np.percentile(bsf[~np.isnan(bsf)], 90):
                    axes[0, 0].axvline(x=int(loc), color='red', alpha=0.7, 
                                      linewidth=2, label='Anomaly' if i == 0 else '')
        
        axes[0, 0].set_title('Time Series with Detected Anomalies')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Discord scores by length
        lengths = range(params['min_length'], params['max_length'] + 1, params['step_size'])
        valid_scores = bsf[~np.isnan(bsf)]
        valid_lengths = [lengths[i] for i in range(len(bsf)) if not np.isnan(bsf[i])]
        
        axes[0, 1].plot(valid_lengths, valid_scores, 'ro-', markersize=6)
        axes[0, 1].set_title('Best Discord Scores by Length')
        axes[0, 1].set_xlabel('Subsequence Length')
        axes[0, 1].set_ylabel('Discord Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Enhanced heatmap
        table_sample = multi_length_table
        if table_sample.shape[1] > 1000:
            step = table_sample.shape[1] // 1000
            table_sample = table_sample[:, ::step]
        
        table_vis = table_sample.copy()
        table_vis[np.isinf(table_vis)] = np.nan
        
        im = axes[1, 0].imshow(table_vis, aspect='auto', origin='lower', 
                              cmap='viridis', interpolation='nearest')
        axes[1, 0].set_title('Multi-Length Discord Matrix')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Length Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Anomaly distribution
        try:
            anomaly_info = self.get_anomaly_scores(threshold_percentile=85)
            if len(anomaly_info['anomalies']) > 0:
                lengths = [a['length'] for a in anomaly_info['anomalies']]
                scores = [a['score'] for a in anomaly_info['anomalies']]
                axes[1, 1].scatter(lengths, scores, alpha=0.7, s=50)
                axes[1, 1].set_title(f'All Detected Anomalies ({len(anomaly_info["anomalies"])})')
                axes[1, 1].set_xlabel('Subsequence Length')
                axes[1, 1].set_ylabel('Anomaly Score')
            else:
                axes[1, 1].text(0.5, 0.5, 'No anomalies detected', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Anomaly Distribution')
        except:
            axes[1, 1].stem(range(len(bsf)), bsf, basefmt=' ')
            axes[1, 1].set_title('Discord Scores by Length Index')
            axes[1, 1].set_xlabel('Length Index')
            axes[1, 1].set_ylabel('Discord Score')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def demo_madrid_v2():
    """
    Demonstration of MADRID v2 usage
    """
    print("MADRID v2 Demo")
    print("=" * 50)
    
    # Initialize MADRID v2
    madrid = MADRID_V2(use_gpu=True, enable_output=True)
    
    # Generate test data with multiple anomalies
    print("Generating test data with multiple anomalies...")
    np.random.seed(42)
    n = 2000
    T = np.random.randn(n)
    
    # Add multiple anomalies of different lengths
    anomaly_locations = [500, 800, 1200, 1500]
    anomaly_lengths = [50, 80, 120, 60]
    
    for loc, length in zip(anomaly_locations, anomaly_lengths):
        if loc + length < n:
            T[loc:loc+length] += np.random.randn(length) * 2
    
    # Run MADRID v2
    print("\nRunning MADRID v2...")
    results = madrid.fit(
        T=T,
        min_length=40,
        max_length=200,
        step_size=20,
        train_test_split=400
    )
    
    # Get multiple anomalies
    print("\nExtracting anomalies...")
    anomaly_info = madrid.get_anomaly_scores(threshold_percentile=85)
    
    print(f"\nFound {len(anomaly_info['anomalies'])} anomalies:")
    for i, anomaly in enumerate(anomaly_info['anomalies'][:10]):
        print(f"  {i+1:2d}. Score: {anomaly['score']:.3f}, "
              f"Location: {anomaly['location']:4d}, "
              f"Length: {anomaly['length']:3d}")
    
    # Plot results
    if PLOTTING_AVAILABLE:
        madrid.plot_results()
    
    return madrid, anomaly_info


if __name__ == "__main__":
    demo_madrid_v2()