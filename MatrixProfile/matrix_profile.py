import numpy as np
import stumpy
import matplotlib.pyplot as plt

class MatrixProfile:
    @staticmethod
    def calculate_matrix_profile_for_sample(sample:np.ndarray, subsequence_length:int):
        return stumpy.stump(sample, subsequence_length)
    
    @staticmethod
    def get_annomaly_index_from_matrix_profile(matrix_profile:np.ndarray) -> int:
        return np.argsort(matrix_profile[:, 0])[-1]
    
    @staticmethod
    def plot_matrix_profile(sample: np.ndarray, matrix_profile: np.ndarray) -> None:
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