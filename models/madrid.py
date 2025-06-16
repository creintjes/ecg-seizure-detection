"""
MADRID - Multi-Length Anomaly Detection with Irregular Discords

This module provides a Python implementation of the MADRID algorithm for time series 
anomaly detection with GPU acceleration support.

MADRID detects discords (anomalous subsequences) across multiple lengths simultaneously,
making it ideal for detecting anomalies of varying durations in time series data.

Dependencies:
    - numpy
    - scipy
    - matplotlib (for plotting)
    - cupy (optional, for GPU acceleration)

Usage Examples:
    
    # Basic usage
    from models.madrid import MADRID
    
    # Generate sample data
    import numpy as np
    T = np.random.randn(10000)  # Your time series data
    
    # Run MADRID with default parameters
    madrid = MADRID(use_gpu=True)  # Enable GPU if available
    multi_length_table, bsf, bsf_loc = madrid.fit(
        T=T,
        min_length=64,
        max_length=256,
        step_size=1,
        train_test_split=5000
    )
    
    # For ECG seizure detection
    # Assuming 250Hz sampling rate
    ecg_data = load_ecg_data()  # Your ECG loading function
    
    # Configure for seizure detection (0.5-30 seconds at 250Hz)
    madrid_ecg = MADRID(use_gpu=True, enable_output=True)
    results = madrid_ecg.fit(
        T=ecg_data,
        min_length=125,    # 0.5 seconds
        max_length=7500,   # 30 seconds
        step_size=125,     # 0.5 second steps
        train_test_split=len(ecg_data)//2
    )
    
    # Access results
    anomaly_table, best_scores, best_locations = results
    
    # Plot results
    madrid_ecg.plot_results()

"""

import numpy as np
import time
import warnings
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

class MADRID:
    """
    Multi-Length Anomaly Detection with Irregular Discords
    
    This class implements the MADRID algorithm for detecting anomalies across
    multiple subsequence lengths in time series data.
    
    Attributes:
        use_gpu (bool): Whether to use GPU acceleration
        enable_output (bool): Whether to enable verbose output and plotting
        device (str): 'cpu' or 'gpu'
    """
    
    def __init__(self, use_gpu: bool = False, enable_output: bool = False):
        """
        Initialize MADRID detector
        
        Args:
            use_gpu (bool): Enable GPU acceleration if available
            enable_output (bool): Enable verbose output and plotting
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.enable_output = enable_output
        self.device = 'gpu' if self.use_gpu else 'cpu'
        
        if use_gpu and not GPU_AVAILABLE:
            warnings.warn("GPU requested but CuPy not available. Using CPU instead.")
        
        # Initialize array library
        self.xp = cp if self.use_gpu else np
        
        # Store results for plotting
        self.last_results = None
        self.last_time_series = None
        
    def _to_device(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Move array to appropriate device (CPU/GPU)"""
        if self.use_gpu:
            return cp.asarray(array)
        return np.asarray(array)
    
    def _to_cpu(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Move array to CPU for output"""
        if self.use_gpu and hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)
    
    def contains_constant_regions(self, T: np.ndarray, min_length: int) -> bool:
        """
        Check if time series contains constant regions that would break MADRID
        
        Args:
            T: Time series data
            min_length: Minimum subsequence length
            
        Returns:
            bool: True if constant regions detected
        """
        T_dev = self._to_device(T)
        
        # Find constant indices
        diff_T = self.xp.diff(T_dev)
        constant_indices = self.xp.where(self.xp.abs(diff_T) < 1e-10)[0]
        
        if len(constant_indices) > 0:
            # Check for consecutive constant regions
            if len(constant_indices) > 1:
                const_diff = self.xp.diff(constant_indices)
                max_constant_length = self.xp.max(const_diff) if len(const_diff) > 0 else 1
                if max_constant_length >= min_length:
                    return True
        
        # Check variance
        if self.xp.var(T_dev) < 0.2:
            return True
            
        return False
    
    def mass_v2(self, x: Union[np.ndarray, 'cp.ndarray'], 
                y: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Mueen's Algorithm for Similarity Search (MASS) V2
        
        Efficiently computes the distance profile between query y and time series x
        using FFT convolution.
        
        Args:
            x: Time series data (database)
            y: Query subsequence
            
        Returns:
            Distance profile array
        """
        x = self._to_device(x)
        y = self._to_device(y)
        
        m = len(y)
        n = len(x)
        
        if m > n:
            return self.xp.array([])
        
        # Compute y stats
        mean_y = self.xp.mean(y)
        std_y = self.xp.std(y, ddof=0)
        
        if std_y == 0:
            std_y = 1e-10  # Avoid division by zero
        
        # Compute x stats using sliding window
        # Using convolution for moving statistics
        ones = self.xp.ones(m)
        
        # Moving sum and sum of squares
        sum_x = self.xp.convolve(x, ones, mode='valid')
        sum_x2 = self.xp.convolve(x**2, ones, mode='valid')
        
        # Moving mean and std
        mean_x = sum_x / m
        var_x = (sum_x2 / m) - mean_x**2
        var_x = self.xp.maximum(var_x, 1e-10)  # Avoid negative variance due to numerical errors
        std_x = self.xp.sqrt(var_x)
        
        # Reverse query for convolution
        y_rev = y[::-1]
        
        # Pad y for FFT
        y_padded = self.xp.zeros(n)
        y_padded[:m] = y_rev
        
        # FFT-based convolution for dot products
        if self.use_gpu:
            X = cp.fft.fft(x)
            Y = cp.fft.fft(y_padded)
            Z = X * Y
            z = cp.fft.ifft(Z).real
        else:
            from scipy import fft
            X = fft.fft(x)
            Y = fft.fft(y_padded)
            Z = X * Y
            z = fft.ifft(Z).real
        
        # Extract valid dot products
        dot_products = z[m-1:n]
        
        # Compute normalized distances
        numerator = dot_products - m * mean_x * mean_y
        denominator = std_x * std_y
        
        # Avoid division by zero
        denominator = self.xp.where(denominator == 0, 1e-10, denominator)
        
        # Z-normalized Euclidean distance
        dist = 2 * (m - numerator / denominator)
        dist = self.xp.maximum(dist, 0)  # Ensure non-negative
        dist = self.xp.sqrt(dist)
        
        return dist
    
    def generate_test_data(self) -> np.ndarray:
        """
        Generate test data similar to the MATLAB version
        
        Returns:
            Test time series data
        """
        np.random.seed(123456789)
        
        # Generate chirp signal
        fs = 10000
        t = np.arange(0, 10, 1/fs)
        f_start = 50
        f_end = 60
        f_in = np.linspace(f_start, f_end, len(t))
        phase_in = np.cumsum(f_in / fs)
        y = np.sin(2 * np.pi * phase_in)
        
        # Add noise
        y = y + np.random.randn(len(y)) / 12
        
        # Add anomalies
        end_of_train = len(y) // 2
        
        # Medium anomaly
        start_idx = end_of_train + 1200
        end_idx = start_idx + 64
        if end_idx < len(y):
            y[start_idx:end_idx] += np.random.randn(end_idx - start_idx) / 3
        
        # Long anomaly 1
        start_idx = end_of_train + 4180
        end_idx = start_idx + 160
        if end_idx < len(y):
            y[start_idx:end_idx] += np.random.randn(end_idx - start_idx) / 4
        
        # Long anomaly 2
        start_idx = end_of_train + 8200
        end_idx = end_of_train + 8390
        if end_idx < len(y):
            y[start_idx:end_idx] *= 0.5
        
        return y
    
    def predict_execution_time(self, ts_length: int, min_length: int, 
                             max_length: int, step_size: int, 
                             train_test_split: int) -> dict:
        """
        Predict execution times for different downsampling factors
        
        Args:
            ts_length: Length of time series
            min_length: Minimum subsequence length
            max_length: Maximum subsequence length
            step_size: Step size between lengths
            train_test_split: Train/test split index
            
        Returns:
            Dictionary with predicted times for each factor
        """
        # Polynomial coefficients from MATLAB version
        complexity_size = len(range(1, ts_length - train_test_split, 1)) * \
                         len(range(int(np.ceil(min_length)), max_length + 1, step_size))
        
        if complexity_size < 5000000:
            # Polynomial model (order 6)
            p_1 = [-4.66922312132205e-45, 1.54665628995475e-35, -1.29314859463985e-26,
                   2.01592418847342e-18, -2.54253065977245e-11, 9.77027495487874e-05, 
                   -1.27055582771851e-05]
            p_2 = [-3.79100071825804e-42, 3.15547030055575e-33, -6.62877819318290e-25,
                   2.59437174380763e-17, -8.10970871564789e-11, 7.25344313152170e-05,
                   4.68415490390476e-07]
        else:
            # Linear model
            p_1 = [3.90752957831437e-05, 0]
            p_2 = [1.94005690535588e-05, 0]
        
        p_4 = [1.26834880558841e-05, 0]
        p_8 = [1.42210521045333e-05, 0]
        p_16 = [1.82290885539705e-05, 0]
        
        predictions = {}
        
        for factor, coeffs in [(1, p_1), (2, p_2), (4, p_4), (8, p_8), (16, p_16)]:
            complexity = len(range(1, ts_length - train_test_split, factor)) * \
                        len(range(int(np.ceil(min_length/factor)), 
                                int(np.ceil(max_length/factor)) + 1, step_size))
            
            if len(coeffs) > 2:  # Polynomial
                time_pred = np.polyval(coeffs, complexity)
            else:  # Linear
                time_pred = coeffs[0] * complexity + coeffs[1]
            
            predictions[factor] = max(time_pred, 0)
        
        return predictions
    
    def fit(self, T: np.ndarray, min_length: int, max_length: int, 
            step_size: int, train_test_split: int, 
            factor: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run MADRID algorithm on time series data
        
        Args:
            T: Input time series
            min_length: Minimum subsequence length
            max_length: Maximum subsequence length  
            step_size: Step size between lengths
            train_test_split: Index separating training and test data
            factor: Downsampling factor (None for auto-selection)
            
        Returns:
            Tuple of (multi_length_discord_table, best_so_far_scores, best_locations)
        """
        # Input validation
        if self.contains_constant_regions(T, min_length):
            error_msg = (
                "BREAK: There is at least one region of length min_length that is constant, "
                "or near constant.\n\n"
                "Whether such regions should be called 'anomalies' depends on the context, "
                "but in any case they are trivial to discover and should not be reported as "
                "a success in algorithm comparison.\n\n"
                "To fix this issue:\n"
                "1) Choose a longer length for min_length.\n"
                "2) Add a small amount of noise to the entire time series\n"
                "3) Add a small linear trend to the entire time series\n"
                "4) Carefully edit the data to remove the constant sections"
            )
            raise ValueError(error_msg)
        
        ts_length = len(T)
        
        # Auto-select factor if not provided
        if factor is None:
            predictions = self.predict_execution_time(
                ts_length, min_length, max_length, step_size, train_test_split
            )
            
            if predictions[1] < 10:
                factor = 1
            else:
                # Interactive selection (simplified for automatic execution)
                if self.enable_output:
                    print("Predicted execution times:")
                    for f in [16, 8, 4, 2, 1]:
                        print(f"{f}: {predictions[f]:.1f} seconds")
                
                # Default to factor that gives reasonable time
                for f in [16, 8, 4, 2, 1]:
                    if predictions[f] < 300:  # Less than 5 minutes
                        factor = f
                        break
                else:
                    factor = 16  # Fallback to fastest
        
        # Apply downsampling
        T_sampled = T[::factor]
        min_len_sampled = int(np.ceil(min_length / factor))
        max_len_sampled = int(np.ceil(max_length / factor))
        split_sampled = int(np.ceil(train_test_split / factor))
        
        # Validate parameters
        if min_len_sampled < 2:
            raise ValueError(f"min_length/{factor} < 2, choose different parameters")
        if max_len_sampled < 2:
            raise ValueError(f"max_length/{factor} < 2, choose different parameters")
        
        if self.enable_output:
            print(f"Running MADRID with factor 1:{factor}")
            print(f"Time series length: {len(T_sampled)}")
            print(f"Length range: {min_len_sampled}-{max_len_sampled}")
        
        # Run core MADRID algorithm
        start_time = time.time()
        multi_length_table, bsf, bsf_loc = self._madrid_core(
            T_sampled, min_len_sampled, max_len_sampled, 
            step_size, split_sampled
        )
        execution_time = time.time() - start_time
        
        if self.enable_output:
            print(f"MADRID execution time: {execution_time:.2f} seconds")
        
        # Store results for plotting
        self.last_results = (multi_length_table, bsf, bsf_loc)
        self.last_time_series = T_sampled
        self.last_params = {
            'min_length': min_len_sampled,
            'max_length': max_len_sampled,
            'step_size': step_size,
            'train_test_split': split_sampled,
            'factor': factor
        }
        
        # Convert back to CPU arrays
        return (self._to_cpu(multi_length_table), 
                self._to_cpu(bsf), 
                self._to_cpu(bsf_loc))
    
    def _madrid_core(self, T: np.ndarray, min_length: int, max_length: int,
                    step_size: int, train_test_split: int) -> Tuple:
        """
        Core MADRID algorithm implementation
        
        This is a simplified version focusing on the main algorithm structure.
        The full implementation would include all the optimizations from the MATLAB version.
        """
        T = self._to_device(T)
        k = 1  # Number of top discords to find
        
        # Initialize data structures
        m_set = list(range(min_length, max_length + 1, step_size))
        num_lengths = len(m_set)
        
        multi_length_table = self.xp.full((num_lengths, len(T)), -self.xp.inf)
        bsf = self.xp.zeros(num_lengths)
        bsf_loc = self.xp.full(num_lengths, self.xp.nan)
        
        if self.enable_output:
            print(f"Processing {num_lengths} different lengths")
        
        # Process each length
        for idx, m in enumerate(m_set):
            if self.enable_output and idx % max(1, num_lengths // 10) == 0:
                print(f"Processing length {m} ({idx+1}/{num_lengths})")
            
            # Run DAMP for this length
            discord_score, position, left_mp = self._damp_simplified(
                T, m, train_test_split
            )
            
            # Normalize scores
            normalization_factor = 1.0 / (2 * self.xp.sqrt(m))
            multi_length_table[idx, :] = left_mp * normalization_factor
            bsf[idx] = discord_score * normalization_factor
            bsf_loc[idx] = position
        
        return multi_length_table, bsf, bsf_loc
    
    def _damp_simplified(self, T: Union[np.ndarray, 'cp.ndarray'], 
                        subseq_length: int, train_test_split: int) -> Tuple:
        """
        Simplified DAMP (Discord Aware Matrix Profile) implementation
        
        This is a basic version that captures the core functionality.
        """
        T = self._to_device(T)
        n = len(T)
        
        # Initialize left matrix profile
        left_mp = self.xp.full(n, -self.xp.inf)
        left_mp[:train_test_split] = self.xp.nan
        
        best_discord_score = -self.xp.inf
        best_position = 0
        
        # Process test portion
        for i in range(train_test_split, n - subseq_length + 1):
            if i + subseq_length > n:
                break
                
            # Extract query
            query = T[i:i + subseq_length]
            
            # Compute distance to all previous subsequences (left matrix profile)
            if i > subseq_length:
                distances = self.mass_v2(T[:i], query)
                if len(distances) > 0:
                    min_distance = self.xp.min(distances)
                    left_mp[i] = min_distance
                    
                    # Update best discord
                    if min_distance > best_discord_score:
                        best_discord_score = min_distance
                        best_position = i
        
        return best_discord_score, best_position, left_mp
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot MADRID results
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.last_results is None:
            print("No results to plot. Run fit() first.")
            return
        
        multi_length_table, bsf, bsf_loc = self.last_results
        T = self.last_time_series
        params = self.last_params
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('MADRID Results', fontsize=16)
        
        # Plot 1: Time series with anomalies marked
        axes[0, 0].plot(T, 'b-', alpha=0.7, label='Time Series')
        if params['train_test_split'] < len(T):
            axes[0, 0].axvline(x=params['train_test_split'], color='r', 
                              linestyle='--', label='Train/Test Split')
        
        # Mark top anomalies
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
        axes[0, 1].set_title('Discord Scores by Subsequence Length')
        axes[0, 1].set_xlabel('Subsequence Length')
        axes[0, 1].set_ylabel('Discord Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Heatmap of multi-length discord table
        # Sample the table for visualization if too large
        table_sample = multi_length_table
        if table_sample.shape[1] > 1000:
            step = table_sample.shape[1] // 1000
            table_sample = table_sample[:, ::step]
        
        # Replace -inf with NaN for better visualization
        table_vis = table_sample.copy()
        table_vis[np.isinf(table_vis)] = np.nan
        
        im = axes[1, 0].imshow(table_vis, aspect='auto', origin='lower', 
                              cmap='viridis', interpolation='nearest')
        axes[1, 0].set_title('Multi-Length Discord Table')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Length Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Top anomalies locations
        axes[1, 1].stem(range(len(bsf)), bsf, basefmt=' ')
        axes[1, 1].set_title('Discord Scores by Length Index')
        axes[1, 1].set_xlabel('Length Index')
        axes[1, 1].set_ylabel('Discord Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_anomaly_scores(self, threshold_percentile: float = 95) -> dict:
        """
        Extract anomaly scores and locations above threshold
        
        Args:
            threshold_percentile: Percentile threshold for anomaly detection
            
        Returns:
            Dictionary with anomaly information
        """
        if self.last_results is None:
            raise ValueError("No results available. Run fit() first.")
        
        multi_length_table, bsf, bsf_loc = self.last_results
        
        # Find threshold
        valid_scores = bsf[~np.isnan(bsf)]
        if len(valid_scores) == 0:
            return {'anomalies': [], 'threshold': 0, 'all_scores': bsf}
        
        threshold = np.percentile(valid_scores, threshold_percentile)
        
        # Find anomalies above threshold
        anomalies = []
        for i, (score, loc) in enumerate(zip(bsf, bsf_loc)):
            if not np.isnan(score) and score >= threshold:
                length = self.last_params['min_length'] + i * self.last_params['step_size']
                anomalies.append({
                    'score': float(score),
                    'location': int(loc) if not np.isnan(loc) else None,
                    'length': length,
                    'length_index': i
                })
        
        return {
            'anomalies': sorted(anomalies, key=lambda x: x['score'], reverse=True),
            'threshold': threshold,
            'all_scores': bsf,
            'all_locations': bsf_loc
        }


def demo_madrid():
    """
    Demonstration of MADRID usage
    """
    print("MADRID Demo")
    print("=" * 50)
    
    # Initialize MADRID
    madrid = MADRID(use_gpu=True, enable_output=True)
    
    # Generate test data
    print("Generating test data...")
    test_data = madrid.generate_test_data()
    
    # Run MADRID
    print("\nRunning MADRID...")
    results = madrid.fit(
        T=test_data,
        min_length=64,
        max_length=256,
        step_size=8,
        train_test_split=len(test_data)//2
    )
    
    # Get anomaly information
    anomaly_info = madrid.get_anomaly_scores(threshold_percentile=90)
    
    print(f"\nFound {len(anomaly_info['anomalies'])} anomalies above 90th percentile")
    print("Top 3 anomalies:")
    for i, anomaly in enumerate(anomaly_info['anomalies'][:3]):
        print(f"  {i+1}. Score: {anomaly['score']:.4f}, "
              f"Location: {anomaly['location']}, "
              f"Length: {anomaly['length']}")
    
    # Plot results
    print("\nPlotting results...")
    madrid.plot_results()
    
    return madrid, results, anomaly_info


if __name__ == "__main__":
    # Run demo
    madrid, results, anomalies = demo_madrid()