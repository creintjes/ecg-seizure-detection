import numpy as np
import stumpy
import matplotlib.pyplot as plt
from typing import List

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