import numpy as np
import stumpy
import matplotlib.pyplot as plt
from typing import List
import neurokit2 as nk
import pandas as pd
import neurokit2 as nk
from typing import Optional, Tuple, List
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

    @staticmethod
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

    def process_ecg_to_hrv_features(
        ecg_signal: np.ndarray,
        sampling_rate: int = 1000,
        rr_window_size: int = 60,
        rr_step: int = 10
    ) -> tuple[np.ndarray, list[float]]:
        """
        Complete pipeline: ECG → R-peaks → RR intervals → HRV features (per window),
        robust for all NeuroKit2 versions by using synthetic peaks from RRIs.

        Parameters
        ----------
        ecg_signal : np.ndarray
            Raw 1D ECG signal.
        sampling_rate : int
            Sampling frequency in Hz (of your ECG data!).
        rr_window_size : int
            Number of RR intervals per HRV feature window.
        rr_step : int
            Step size between windows.

        Returns
        -------
        tuple[np.ndarray, list[float]]
            - HRV feature matrix of shape (n_windows, n_features)
            - List of center timestamps (in seconds) for each window
        """
        # Step 1: Find R-peaks in the ECG signal
        rpeaks = MatrixProfile.extract_rpeaks_from_ecg(ecg_signal, sampling_rate=sampling_rate)
        print(f"Found {len(rpeaks)} R-peaks")

        if len(rpeaks) < rr_window_size + 1:
            print(f"Warning: Only {len(rpeaks)} R-peaks found. Need at least {rr_window_size + 1} for one window.")
            return np.empty((0,), dtype=np.float64), []

        # Step 2: Compute RR intervals in milliseconds
        rr_intervals = MatrixProfile.compute_rr_intervals(rpeaks, sampling_rate=sampling_rate)

        features = []
        timestamps = []
        cumulative_time = np.cumsum(rr_intervals) / 1000.0  # seconds

        for start in range(0, len(rr_intervals) - rr_window_size, rr_step):
            rr_window = rr_intervals[start:start + rr_window_size]

            # Debug output for inspection
            # print(f"[{start}] RR-Window: {rr_window[:5]} ... {rr_window[-5:]}, min={rr_window.min()}, max={rr_window.max()}, dtype={rr_window.dtype}")

            # Plausibility check for RR intervals
            if not np.isfinite(rr_window).all():
                print(f"[{start}] Non-finite RRIs → skipping")
                continue
            if np.min(rr_window) < 300 or np.max(rr_window) > 2000:
                print(f"[{start}] RR interval out of physiological range (min={np.min(rr_window)}, max={np.max(rr_window)}) → skipping")
                continue

            t_center = cumulative_time[start:start + rr_window_size].mean()

            try:
                # Convert RRIs (ms) into synthetic peak indices (sampling_rate=1000 needed for ms units)
                synthetic_peaks = nk.intervals_to_peaks(rr_window, sampling_rate=1000)
                # Build dict for nk.hrv (expects key "ECG_R_Peaks")
                peaks_dict = {"ECG_R_Peaks": synthetic_peaks}
                # Calculate HRV features from synthetic peaks (always with sampling_rate=1000 for ms!)
                hrv = nk.hrv(peaks=peaks_dict, sampling_rate=1000, show=False)
                if not hrv.empty:
                    features.append(hrv.values[0])
                    timestamps.append(t_center)
            except Exception as e:
                print(f"HRV error at window {start}: {e}")
                continue

        return np.array(features, dtype=np.float64), timestamps

    def detect_anomalies_from_hrv_features(
        features: np.ndarray,
        timestamps: List[float],
        subsequence_length: int,
        ground_truth_intervals: Optional[List[List[float]]] = None,
        sampling_rate: int = 256,
        top_k_percent: float = 5.0
    ) -> Tuple[List[int], int, int, np.ndarray]:
        """
        Detect anomalies in HRV feature space using multivariate matrix profiling
        and compare detected anomalies to annotated seizure intervals.

        Parameters:
        ----------
        features : np.ndarray
            HRV feature matrix (n_windows, n_features)
        timestamps : List[float]
            List of center timestamps (in seconds) for each HRV window
        subsequence_length : int
            Window size (in HRV steps) for matrix profile
        ground_truth_intervals : Optional[List[List[float]]]
            List of [start_sample, end_sample] intervals (256 Hz) indicating seizure events
        sampling_rate : int
            Sampling rate of original EKG signal
        top_k_percent : float
            Percentage of top anomalies to consider (e.g., 5.0 for top 5%)

        Returns:
        -------
        Tuple:
            - List of anomaly sample indices (int)
            - Number of true positives (within any GT interval)
            - Number of false positives (outside all GT intervals)
            - Matrix profile mean (np.ndarray)
        """
        if features.shape[0] < subsequence_length + 1:
            raise ValueError("Not enough feature vectors for given subsequence_length.")

        # Compute matrix profile
        matrix_profile, _ = stumpy.mstump(features.T, m=subsequence_length)
        profile_mean = matrix_profile.mean(axis=0)

        n_scores = len(profile_mean)
        top_k = max(1, int(np.floor(top_k_percent / 100 * n_scores)))

        top_indices = np.argsort(profile_mean)[-top_k:][::-1]

        # Project anomaly positions to original EKG sample domain (256Hz)
        anomaly_sample_indices = [
            int(timestamps[i + subsequence_length // 2] * sampling_rate)
            for i in top_indices
            if (i + subsequence_length // 2) < len(timestamps)
        ]

        # Evaluate anomaly matches
        tp = 0
        fp = 0
        if ground_truth_intervals is not None:
            for idx in anomaly_sample_indices:
                matched = any(start <= idx <= end for start, end in ground_truth_intervals)
                if matched:
                    tp += 1
                else:
                    fp += 1

        return anomaly_sample_indices, tp, fp, profile_mean

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