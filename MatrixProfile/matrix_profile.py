import numpy as np
import stumpy
import matplotlib.pyplot as plt
from typing import List
import neurokit2 as nk
import pandas as pd
import neurokit2 as nk
from typing import Optional
from typing import Tuple
import warnings
from neurokit2.misc import NeuroKitWarning

warnings.filterwarnings("ignore", category=NeuroKitWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MatrixProfile:
    @staticmethod

    def calculate_matrix_profile_for_sample(sample:np.ndarray, subsequence_length:int):
        return stumpy.stump(sample, subsequence_length)
    
    def compute_approx_matrix_profile(time_series: np.ndarray, subsequence_length: int, percentage: float = 0.1, pre_scrump: bool = True) -> np.ndarray:
        """
        Computes an approximate matrix profile using STUMPY's SCRUMP algorithm.

        Args:
            time_series (np.ndarray): 1D input time series as a numpy array.
            subsequence_length (int): Size of the sliding window for the matrix profile computation.
            percentage (float): Fraction of the distance matrix to compute (0 < percentage <= 1).
            pre_scrump (bool): Whether to run the pre-SCRUMP phase to improve accuracy.

        Returns:
            np.ndarray: The approximate matrix profile.
        """
        if not 0 < percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1.")

        if len(time_series) < subsequence_length:
            raise ValueError("Time series length must be greater than the subsequence length.")

        # Initialize and compute SCRUMP matrix profile
        matrix_profile = stumpy.scrump(time_series, m=subsequence_length, percentage=percentage, pre_scrump=pre_scrump)
        return matrix_profile

    def compute_multivariate_matrix_profile(features: np.ndarray, subsequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a multivariate matrix profile using STUMPY's mstump.

        Parameters:
        ----------
        features : np.ndarray
            2D array of shape (n_samples, n_features), e.g. HRV features over time.
        subsequence_length : int
            Window size (number of feature vectors) for the matrix profile.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple (matrix_profile, profile_indices) where:
            - matrix_profile has shape (n_samples - m + 1, n_features)
            - profile_indices are the indices of the nearest neighbors
        """
        if features.ndim != 2:
            raise ValueError("Expected a 2D array with shape (n_samples, n_features).")

        # STUMPY expects input shape: (n_features, n_samples)
        features_T = features.T

        # Compute the multivariate matrix profile
        matrix_profile, profile_indices = stumpy.mstump(features_T, m=subsequence_length)

        return matrix_profile, profile_indices


    def extract_rpeaks_from_ecg(ecg_signal: np.ndarray, sampling_rate: int = 256) -> np.ndarray:
        """
        Detect R-peaks from raw ECG signal using robust neurokit2 cleaning and peak detection.

        Parameters:
        ----------
        ecg_signal : np.ndarray
            1D ECG signal.
        sampling_rate : int
            Sampling rate in Hz.

        Returns:
        -------
        np.ndarray
            Array of R-peak sample indices.
        """
        if ecg_signal.ndim != 1:
            raise ValueError("ECG signal must be 1D.")

        try:
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="neurokit")
            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            peaks = rpeaks.get("ECG_R_Peaks", None)
            if peaks is None or len(peaks) == 0:
                raise RuntimeError("No R-peaks found.")
            return peaks
        except Exception as e:
            raise RuntimeError(f"R-peak detection failed: {e}")

    def compute_rr_intervals(rpeaks: np.ndarray, sampling_rate: int = 1000) -> np.ndarray:
        """
        Compute RR intervals from R-peak locations.

        Parameters
        ----------
        rpeaks : np.ndarray
            Indices of detected R-peaks in the ECG signal.
        sampling_rate : int
            Sampling rate of ECG in Hz.

        Returns
        -------
        np.ndarray
            RR intervals in milliseconds.
        """
        rr_samples = np.diff(rpeaks)
        rr_ms = rr_samples * (1000.0 / sampling_rate)
        return rr_ms.astype(np.float64)


    # def extract_hrv_features_over_windows(rr_intervals: np.ndarray,
    #                                   window_size: int = 60,
    #                                   step: int = 10,
    #                                   sampling_rate: int = 256) -> np.ndarray:
    #     """
    #     Robust extraction of HRV features from RR intervals using hrv_time.
    #     """
    #     features = []

    #     for start in range(0, len(rr_intervals) - window_size, step):
    #         rr_window = rr_intervals[start:start + window_size]

    #         # Clean window
    #         if not np.isfinite(rr_window).all():
    #             print(f"[{start}] ❌ non-finite values → skipping")
    #             continue
    #         if np.min(rr_window) < 300:
    #             print(f"[{start}] ❌ RR values too small → skipping")
    #             continue

    #         try:
    #             # Convert to peaks explicitly to avoid internal confusion
    #             peaks = nk.intervals_to_peaks(rr_window, sampling_rate=sampling_rate)
    #             hrv = nk.hrv(peaks=peaks, sampling_rate=sampling_rate, show=False)
    #             if not hrv.empty:
    #                 features.append(hrv.values[0])
    #         except Exception as e:
    #             print(f"[{start}] HRV error: {e}")
    #             continue

    #     return np.array(features)
    # def extract_hrv_features_over_windows(rr_intervals: np.ndarray,
    #                                   window_size: int = 60,
    #                                   step: int = 10,
    #                                   sampling_rate: int = 256) -> Tuple[np.ndarray, List[float]]:
    #     """
    #     Extract HRV features from RR intervals and return their timestamps (in seconds).
        
    #     Returns:
    #         - HRV feature matrix: (n_windows, n_features)
    #         - Time index list:    [sec_1, sec_2, ..., sec_n]
    #     """
    #     features = []
    #     time_stamps = []

    #     cumulative_time = np.cumsum(rr_intervals) / 1000.0  # in seconds

    #     for start in range(0, len(rr_intervals) - window_size, step):
    #         rr_window = rr_intervals[start:start + window_size]

    #         # Mittelzeitpunkt in Sekunden
    #         t_center = cumulative_time[start:start + window_size].mean()

    #         try:
    #             # Erzeuge künstliche R-Peak-Zeitpunkte aus kumulierter RR-Summe (in ms → samples)
    #             peak_positions_ms = np.cumsum(rr_window)  # in ms
    #             peak_positions_samples = np.round(peak_positions_ms * sampling_rate / 1000).astype(int)

    #             # Sicherstellen, dass Indizes strikt monoton sind
    #             if np.any(np.diff(peak_positions_samples) <= 0):
    #                 raise ValueError(f"Non-monotonic synthetic R-peaks at window")

    #             hrv = nk.hrv(peaks=peak_positions_samples, sampling_rate=sampling_rate, show=False)

    #             if not hrv.empty:
    #                 features.append(hrv.values[0])
    #                 time_stamps.append(t_center)
    #         except Exception:
    #             continue

    #     return np.array(features), time_stamps

    def extract_hrv_features_from_peaks(
        rpeaks: np.ndarray,
        window_size: int = 60,
        step: int = 10,
        sampling_rate: int = 256
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Robustly extract HRV features from sliding windows of R-peak indices.

        Args:
            rpeaks (np.ndarray): Array of R-peak indices (sample positions).
            window_size (int): Number of R-peaks per window.
            step (int): Step size for the sliding window.
            sampling_rate (int): ECG sampling rate in Hz.

        Returns:
            Tuple[np.ndarray, List[float]]:
                - HRV feature matrix of shape (n_windows, n_features)
                - List of center timestamps (in seconds) for each window
        """
        features = []
        timestamps = []

        for start in range(0, len(rpeaks) - window_size, step):
            window_peaks = rpeaks[start:start + window_size]

            # Ensure window is strictly monotonically increasing (valid R-peaks)
            if len(window_peaks) < 2 or np.any(np.diff(window_peaks) <= 0):
                print(f"[{start}] ❌ Invalid R-peaks in window → skipping")
                continue

            # Convert R-peak indices to RR intervals (ms)
            rr_window = np.diff(window_peaks) * (1000.0 / sampling_rate)  # dtype: float64

            # Calculate center time in seconds
            t_center = np.mean(window_peaks) / sampling_rate

            try:
                # Calculate HRV features from RR intervals
                hrv = nk.hrv_time(rr_window, sampling_rate=sampling_rate, show=False)
                if not hrv.empty:
                    features.append(hrv.values[0])
                    timestamps.append(t_center)
            except Exception as e:
                print(f"[{start}] HRV error: {e}")
                continue

        return np.array(features, dtype=np.float64), timestamps






    # def process_ecg_to_hrv_features(ecg_signal: np.ndarray,
    #                                 sampling_rate: int = 1000,
    #                                 rr_window_size: int = 60,
    #                                 rr_step: int = 10) -> np.ndarray:
    #     """
    #     Complete pipeline: ECG → R-peaks → RR intervals → HRV features (per window).
    #     Now includes validity checks.
    #     """
    #     rpeaks = MatrixProfile.extract_rpeaks_from_ecg(ecg_signal, sampling_rate=sampling_rate)
    #     print(f"Found {len(rpeaks)} R-peaks")
    #     # Require at least (window_size + 1) R-peaks to form one RR window
    #     if len(rpeaks) < rr_window_size + 1:
    #         print(f"Warning: Only {len(rpeaks)} R-peaks found. Need at least {rr_window_size + 1} for one window.")
    #         return np.empty((0,))

    #     rr_intervals = MatrixProfile.compute_rr_intervals(rpeaks, sampling_rate=sampling_rate)
    #     hrv_features = MatrixProfile.extract_hrv_features_over_windows(rr_intervals, rr_window_size, rr_step, sampling_rate)
    #     return hrv_features


    def process_ecg_to_hrv_features(ecg_signal: np.ndarray,
                                    sampling_rate: int = 1000,
                                    rr_window_size: int = 60,
                                    rr_step: int = 10) -> Tuple[np.ndarray, List[float]]:
        """
        Complete pipeline: ECG → R-peaks → RR intervals → HRV features (per window),
        including timing information for each window center.

        Parameters:
        ----------
        ecg_signal : np.ndarray
            Raw 1D ECG signal.
        sampling_rate : int
            Sampling frequency in Hz.
        rr_window_size : int
            Number of RR intervals per HRV feature window.
        rr_step : int
            Step size between windows.

        Returns:
        -------
        Tuple[np.ndarray, List[float]]
            - HRV feature matrix of shape (n_windows, n_features)
            - List of center timestamps (in seconds) for each window
        """
        rpeaks = MatrixProfile.extract_rpeaks_from_ecg(ecg_signal, sampling_rate=sampling_rate)
        print(f"Found {len(rpeaks)} R-peaks")

        if len(rpeaks) < rr_window_size + 1:
            print(f"Warning: Only {len(rpeaks)} R-peaks found. Need at least {rr_window_size + 1} for one window.")
            return np.empty((0,)), []

        rr_intervals = MatrixProfile.compute_rr_intervals(rpeaks, sampling_rate=sampling_rate)

        features = []
        timestamps = []
        cumulative_time = np.cumsum(rr_intervals) / 1000.0  # seconds

        # We slide a window over the RR intervals to compute HRV features per segment.
        # This loop allows us to extract features from overlapping RR windows (e.g., every 10 beats),
        # which provides a fine-grained, temporally resolved feature time series for anomaly detection.
        for start in range(0, len(rr_intervals) - rr_window_size, rr_step):
            rr_window = rr_intervals[start:start + rr_window_size]
            t_center = cumulative_time[start:start + rr_window_size].mean()

            try:
                hrv = nk.hrv_time(rr_window, sampling_rate=sampling_rate, show=False)
                if not hrv.empty:
                    features.append(hrv.values[0])
                    timestamps.append(t_center)
            except Exception as e:
                print(f"HRV error at window {start}: {e}")
                continue

        return np.array(features), timestamps



    def get_top_k_anomaly_indices(matrix_profile: np.ndarray, k: int) -> List[int]:
        """
        Identify the indices of the top-k highest values in a matrix profile, 
        which are indicative of the most anomalous subsequences.

        Parameters:
        ----------
        matrix_profile : np.ndarray
            The matrix profile array, typically computed with STUMP or STUMPI.
        k : int
            The number of top anomalies (i.e., highest matrix profile values) to return.

        Returns:
        -------
        List[int]
            A list of indices corresponding to the top-k anomaly scores in the matrix profile.
        """
        if k > len(matrix_profile):
            raise ValueError(f"Requested top-{k} anomalies, but matrix profile has only {len(matrix_profile)} elements.")

        # Get the indices of the top-k highest values in descending order
        top_k_indices = np.argsort(matrix_profile)[-k:][::-1]
        
        return top_k_indices.tolist()
    
    @staticmethod
    def mean_of_all_consecutive_anomalies(indices: List[int], n: int, max_gap: int = 1) -> List[int]:
        """
        Identifies all groups of up to 'n' consecutive or closely spaced anomaly indices
        and returns the rounded mean of each group with size >= n.

        Parameters:
        ----------
        indices : List[int]
            The list of anomaly indices (e.g., top-k indices from a matrix profile).
        n : int
            Minimum size of consecutive or closely spaced groups to be considered.
        max_gap : int, optional
            Maximum allowed gap between indices to still be considered consecutive.
            Default is 1 (i.e., strictly consecutive).

        Returns:
        -------
        List[int]
            A list of rounded means (integers) of all valid anomaly groups.
        """
        if len(indices) < n:
            return []

        sorted_indices = sorted(indices)
        result = []
        group = [sorted_indices[0]]

        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] - sorted_indices[i - 1] <= max_gap:
                group.append(sorted_indices[i])
            else:
                if len(group) >= n:
                    result.append(int(round(np.mean(group))))
                group = [sorted_indices[i]]

        # Check last group
        if len(group) >= n:
            result.append(int(round(np.mean(group))))

        return result

    @staticmethod
    def calculate_matrix_profile_for_sample_gpu(sample:np.ndarray, subsequence_length:int):
        return stumpy.gpu_stump(sample, subsequence_length)    
    
    @staticmethod
    def get_annomaly_index_from_matrix_profile(matrix_profile:np.ndarray) -> int:
        return np.argsort(matrix_profile[:, 0])[-1]
    
    @staticmethod
    def plot_matrix_profile(sample: np.ndarray, matrix_profile: np.ndarray) -> plt.Figure:
        """Plot the original time series and its matrix profile, and highlight the detected anomaly.
        
        Args:
            sample (np.ndarray): The input time series.
            matrix_profile (np.ndarray): The computed matrix profile.
        
        Returns:
            plt.Figure: The matplotlib figure with the plots.
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 4))

        ax[0].plot(sample)
        ax[0].set_title("time series")

        ax[1].plot(matrix_profile[:, 0])
        ax[1].set_title(f"Found an outlier at {MatrixProfile.get_annomaly_index_from_matrix_profile(matrix_profile=matrix_profile)}")

        fig.text(0.01, 0.01, "Matrixprofile (small = similar, large = potential outlier)", fontsize=8, color="gray")
        fig.tight_layout()
        plt.show()