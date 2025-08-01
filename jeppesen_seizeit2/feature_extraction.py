"""
Jeppesen Feature-Extraktion adaptiert für SeizeIT2
Basiert auf Original-Implementierung mit scipy-Abhängigkeiten
"""

import numpy as np
import pandas as pd
import scipy.signal
from typing import Tuple, List
from config import ELGENDI_PARAMS

def peak_detection_elgendi(ecg_data: np.ndarray, sampling_rate: int, **kwargs) -> List[int]:
    """
    Detects R-peaks in ECG data using the Elgendi method.
    Adaptiert für SeizeIT2 (250Hz statt 512Hz).
    
    Parameters:
        ecg_data (np.ndarray): The ECG signal data.
        sampling_rate (int): The sampling rate of the ECG data in Hz.
        **kwargs: Additional parameters (überschreibt config defaults)
        
    Returns:
        List[int]: List of indices representing the detected R-peaks.
    """
    # Merge config parameters with kwargs
    params = {**ELGENDI_PARAMS, **kwargs}
    
    def _filter_peaks(ecg_data, foundpeaks, sampling_rate, min_rr_distance=0.25):
        """
        Filters detected peaks based on minimum RR interval distance.
        """
        filtered_peaks = []
        jumpnextone = False
        min_rr_samples = int(min_rr_distance * sampling_rate)
        
        for i in range(len(foundpeaks) - 1):
            if jumpnextone:
                jumpnextone = False
                continue
            
            dist = foundpeaks[i + 1] - foundpeaks[i]
            
            # forwards block proximity filter
            if dist > min_rr_samples:
                # backwards block proximity filter
                if len(filtered_peaks) == 0 or (foundpeaks[i] - filtered_peaks[-1]) > min_rr_samples:
                    filtered_peaks.append(foundpeaks[i])
            else:
                if ecg_data[foundpeaks[i]] > ecg_data[foundpeaks[i + 1]]:
                    # backwards block proximity filter
                    if len(filtered_peaks) == 0 or (foundpeaks[i] - filtered_peaks[-1]) > min_rr_samples:
                        filtered_peaks.append(foundpeaks[i])
                    jumpnextone = True
                else:
                    # backwards block proximity filter
                    if len(filtered_peaks) == 0 or (foundpeaks[i + 1] - filtered_peaks[-1]) > min_rr_samples:
                        filtered_peaks.append(foundpeaks[i + 1])
                    jumpnextone = True
        
        # Check the last peak
        if len(foundpeaks) > 0 and (len(filtered_peaks) == 0 or (foundpeaks[-1] - filtered_peaks[-1]) > min_rr_samples):
            filtered_peaks.append(foundpeaks[-1])

        return filtered_peaks
    
    # Bandpass Filter
    nyquist = 0.5 * sampling_rate
    low = params['low'] / nyquist
    high = params['high'] / nyquist
    
    # Stelle sicher, dass Grenzfrequenzen im gültigen Bereich sind
    if low >= 1.0 or high >= 1.0:
        raise ValueError(f"Grenzfrequenzen zu hoch für Sampling-Rate {sampling_rate}Hz")
    
    coeffs = scipy.signal.butter(params['order'], [low, high], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], ecg_data)
    
    # First Derivative (QRS enhancement)
    diff = np.diff(filtered)
    diff = np.append(diff, diff[-1])
    
    # Squaring (QRS enhancement)
    squared = diff ** 2
    
    # Normalization
    filtered_norm = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
    peaks = np.zeros(len(filtered_norm))
    
    w1 = int(params['w1factor'] * sampling_rate)
    w2 = int(params['w2factor'] * sampling_rate)
    
    # Moving averages
    maqrs = np.convolve(squared, np.ones(w1), mode="same") / w1
    mabeat = np.convolve(squared, np.ones(w2), mode="same") / w2
    
    alpha = params['beta'] * np.mean(squared)
    thr1 = mabeat + alpha

    # Determination of Blocks of Interest
    blocksofinterest = maqrs > thr1
    blocksofinterest = np.append(blocksofinterest, False)
    boi = False
    
    for i, boi_val in enumerate(blocksofinterest):
        if boi_val and not boi:
            boi = True
            boiarea = i
        elif not boi_val and boi:
            boi = False
            # Block width filter
            if (i - boiarea) >= w1:
                peak = boiarea + np.argmax(filtered_norm[boiarea:i])
                peaks[peak] = 1
                
    foundpeaks = np.where(peaks == 1)[0]
    
    return _filter_peaks(ecg_data, foundpeaks, sampling_rate)

def median_filter(rr_intervals: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Applies a median filter to the R-R interval series (tachogram).
    
    Parameters:
        rr_intervals (np.ndarray): The R-R intervals in milliseconds.
        window_size (int): Size of the median filter window.
        
    Returns:
        np.ndarray: The filtered R-R intervals.
    """
    return pd.Series(rr_intervals).rolling(window=window_size, center=False, min_periods=1).median().to_numpy()

def calculate_sd1_sd2(rr_intervals: np.ndarray) -> Tuple[float, float]:
    """
    Calculates SD1 and SD2 for a series of R-R intervals.
    
    Parameters:
        rr_intervals (np.ndarray): Sequence of R-R intervals.
        
    Returns:
        Tuple[float, float]: (SD1, SD2) values.
    """
    rr_intervals = np.asarray(rr_intervals)
    x1 = rr_intervals[:-1]
    x2 = rr_intervals[1:]
    
    diff1 = (x2 - x1) / np.sqrt(2)
    sum1 = (x2 + x1 - 2 * np.mean(rr_intervals)) / np.sqrt(2)
    
    sd1 = np.std(diff1, ddof=1)
    sd2 = np.std(sum1, ddof=1)
    
    return sd1, sd2

def calculate_csi(rr_intervals: np.ndarray, window_size: int = 50, smoothing_window_size: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates CSI and Modified CSI retrospectively using sliding windows over RR intervals.
    
    Parameters:
        rr_intervals (np.ndarray): Sequence of R-R intervals.
        window_size (int): Window size for the sliding evaluation.
        smoothing_window_size (int): Size of the smoothing window for rolling mean.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (CSI values, Modified CSI values)
    """
    if len(rr_intervals) < window_size:
        raise ValueError("Length of rr_intervals must be greater than or equal to window_size.")
        
    rr_intervals = np.asarray(rr_intervals)
    csi_values = [np.nan] * (window_size - 1)
    modcsi_values = [np.nan] * (window_size - 1)

    for i in range(window_size - 1, len(rr_intervals)):
        start_idx = max(0, i - window_size + 1)
        window = rr_intervals[start_idx:i + 1]
        sd1, sd2 = calculate_sd1_sd2(window)
        
        T = 4 * sd1
        L = 4 * sd2
        
        if T == 0:
            csi = np.nan
            modcsi = np.nan
        else:
            csi = L / T
            modcsi = (L ** 2) / T
        
        csi_values.append(csi)
        modcsi_values.append(modcsi)
    
    smoothing_window_size = int(smoothing_window_size)
    if smoothing_window_size > 0:
        csi_values = pd.Series(csi_values).rolling(window=smoothing_window_size, min_periods=1).mean().to_numpy()
        modcsi_values = pd.Series(modcsi_values).rolling(window=smoothing_window_size, min_periods=1).mean().to_numpy()

    return np.array(csi_values), np.array(modcsi_values)

def calculate_hr_diff(rr_intervals: np.ndarray, window_size: int = 50, smoothing_window_size: int = -1) -> np.ndarray:
    """
    Calculates the HR-diff feature using second-order central differences 
    over sliding windows of R-R intervals.
    
    Parameters:
        rr_intervals (np.ndarray): Sequence of R-R intervals.
        window_size (int): Number of intervals in each window.
        smoothing_window_size (int): Size of the smoothing window for rolling mean.
    
    Returns:
        np.ndarray: HR-diff values, padded with NaNs for alignment.
    """
    if len(rr_intervals) < window_size:
        raise ValueError("Length of rr_intervals must be >= window_size.")
    
    rr_intervals = np.asarray(rr_intervals)
    hr_diff_values = [np.nan] * (window_size - 1)

    all_diffs = np.zeros(len(rr_intervals))
    for i in range(1, len(rr_intervals)-1):
        all_diffs[i] = rr_intervals[i+1] - rr_intervals[i-1]
    
    for i in range(window_size - 1, len(rr_intervals)):
        start_idx = i - window_size + 1
        window_sum = np.sum(all_diffs[max(1, start_idx):i+1])
        hr_diff_values.append(window_sum)

    smoothing_window_size = int(smoothing_window_size)
    if smoothing_window_size > 0:
        hr_diff_values = pd.Series(hr_diff_values).rolling(window=smoothing_window_size, min_periods=1).mean().to_numpy()

    return np.array(hr_diff_values)

def calculate_relative_tachogram_slope(rr_intervals: np.ndarray, window_size: int = 50, smoothing_window_size: int = -1) -> np.ndarray:
    """
    Calculates the relative slope of the tachogram using the Least Squares method
    over sliding windows of RR intervals, normalized by the mean RR interval.

    Parameters:
        rr_intervals (np.ndarray): Sequence of R-R intervals.
        window_size (int): Number of intervals in each window.
        smoothing_window_size (int): Size of the smoothing window for rolling mean.

    Returns:
        np.ndarray: Normalized slope magnitudes, aligned with RR timeline.
    """
    if len(rr_intervals) < window_size:
        raise ValueError("Length of rr_intervals must be >= window_size.")
    
    rr_intervals = np.asarray(rr_intervals)
    slope_values = [np.nan] * (window_size - 1)

    for i in range(window_size - 1, len(rr_intervals)):
        start_idx = i - window_size + 1
        window = rr_intervals[start_idx:i + 1]

        x = np.arange(len(window))
        y = window
        
        # Get mean RR for normalization
        mean_rr = np.mean(window)

        # Calculate slope and normalize by mean RR
        cov = np.cov(x, y, ddof=1)[0, 1]
        var_x = np.var(x, ddof=1)
        
        slope = cov / var_x if var_x != 0 else 0.0
        # Normalize by mean RR to get relative change
        norm_slope = slope / mean_rr if mean_rr != 0 else 0.0
        
        slope_values.append(abs(norm_slope))

    smoothing_window_size = int(smoothing_window_size)
    if smoothing_window_size > 0:
        slope_values = pd.Series(slope_values).rolling(window=smoothing_window_size, min_periods=1).mean().to_numpy()

    return np.array(slope_values)

def add_feature_combinations(peak_dataframe: pd.DataFrame, *feature_groups: List[str]) -> pd.DataFrame:
    """
    Adds combinations of features to a DataFrame.

    Parameters:
        peak_dataframe (pd.DataFrame): DataFrame containing the features.
        *feature_groups (List[str]): Variable number of lists, each containing feature names to calculate combinations for.

    Returns:
        pd.DataFrame: DataFrame containing the calculated combinations.
    """
    df = peak_dataframe.copy()
    for feature_group in feature_groups:
        for i in range(len(feature_group)):
            for j in range(i + 1, len(feature_group)):
                feature1 = feature_group[i]
                feature2 = feature_group[j]
                if "x" in feature1 or "x" in feature2:
                    df[f"&{feature1}x{feature2}"] = df[feature1] * df[feature2]
                else:
                    df[f"{feature1}x{feature2}"] = df[feature1] * df[feature2]
    return df

def get_peak_dataframe(ecg_df: pd.DataFrame, peak_detection_method: str = "elgendi", 
                      sampling_rate: int = None) -> pd.DataFrame:
    """
    Computes R-R intervals and returns a DataFrame with the results.
    Adaptiert für SeizeIT2-Format.
    
    Parameters:
        ecg_df (pd.DataFrame): DataFrame containing ECG data with a 'ecg' column.
        peak_detection_method (str): The method to use for peak detection.
        sampling_rate (int): The sampling rate of the ECG data in Hz.
        
    Returns:
        pd.DataFrame: A DataFrame containing the R-R intervals.
    """
    if sampling_rate is None:
        sampling_rate = ecg_df.attrs.get('sampling_rate', 250)
    
    if peak_detection_method == 'elgendi':
        r_peaks = peak_detection_elgendi(ecg_data=ecg_df['ecg'].values, sampling_rate=sampling_rate)
    else:
        raise ValueError(f"Unbekannte Peak-Detection-Methode: {peak_detection_method}")
    
    # Konvertiere Sample-Indices zu Zeitpunkten
    r_peaks_times = ecg_df.index[r_peaks]

    # Berechne RR-Intervalle in Millisekunden
    rr_series = (r_peaks_times.to_series().diff().dropna().dt.total_seconds() * 1000).values
    rr_series_filtered = median_filter(rr_series)

    # Erstelle Peak-DataFrame mit RR-Intervallen
    peak_data = pd.DataFrame({
        'rr_intervals': rr_series,
        'rr_intervals_filtered': rr_series_filtered,
    }, index=r_peaks_times[1:])  # Erste RR-Intervall startet beim zweiten Peak
    
    peak_data.index.name = 'r_peak'
    
    # Kopiere Seizure-Spalten falls vorhanden
    if 'seizure' in ecg_df.columns:
        peak_data['seizure'] = 0
        peak_data['seizure_unfiltered'] = 0
        
        for peak_time in peak_data.index:
            # Finde nächsten ECG-Zeitpunkt für Seizure-Label
            closest_idx = ecg_df.index.get_indexer([peak_time], method='nearest')[0]
            peak_data.at[peak_time, 'seizure'] = ecg_df.iloc[closest_idx]['seizure']
            if 'seizure_unfiltered' in ecg_df.columns:
                peak_data.at[peak_time, 'seizure_unfiltered'] = ecg_df.iloc[closest_idx]['seizure_unfiltered']
            else:
                peak_data.at[peak_time, 'seizure_unfiltered'] = ecg_df.iloc[closest_idx]['seizure']

    # Kopiere Metadaten
    for attr_name, attr_value in ecg_df.attrs.items():
        peak_data.attrs[attr_name] = attr_value
    
    peak_data.attrs['peak_detection_method'] = peak_detection_method
    peak_data.attrs['n_peaks'] = len(r_peaks)
    
    return peak_data