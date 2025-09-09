"""
Evaluation-Utilities für Jeppesen SeizeIT2 Anpassung
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def apply_seizure_padding(record_df: pd.DataFrame, seizure_padding: Tuple[int, int], 
                         seizure_col: str, new_col_name: str) -> pd.DataFrame:
    """
    Applies seizure padding to the DataFrame based on the specified column name.
    
    Parameters:
        record_df (pd.DataFrame): DataFrame containing the seizure column.
        seizure_padding (Tuple[int, int]): Number of peaks before and after the seizure to include.
        seizure_col (str): Name of the existing seizure column.
        new_col_name (str): Name of the new seizure column with applied padding.

    Returns:
        pd.DataFrame: DataFrame with applied seizure padding.
    """
    record_df_copy = record_df.copy()
    if seizure_col not in record_df.columns:
        raise ValueError(f"The DataFrame must contain a '{seizure_col}' column")

    record_df_copy[new_col_name] = 0

    record_df_copy['seizure_group'] = (record_df_copy[seizure_col] != record_df_copy[seizure_col].shift()).cumsum()
    seizure_groups = record_df_copy[record_df_copy[seizure_col] == 1].groupby('seizure_group')

    # Apply padding for each seizure group
    for _, group in seizure_groups:
        start_idx = group.index[0]
        end_idx = group.index[-1]

        start_padding_idx = max(0, record_df_copy.index.get_loc(start_idx) - seizure_padding[0])
        end_padding_idx = min(len(record_df_copy), record_df_copy.index.get_loc(end_idx) + seizure_padding[1] + 1)
        record_df_copy.iloc[start_padding_idx:end_padding_idx, record_df_copy.columns.get_loc(new_col_name)] = 1

    record_df_copy[new_col_name] = record_df_copy[new_col_name].astype(int)
    record_df_copy.drop(columns=['seizure_group'], inplace=True)

    return record_df_copy

def compute_cutoff_value(peak_dataframes: List[pd.DataFrame], metric: str, 
                        seizure_col: str = 'seizure_for_cutoff') -> float:
    """
    Compute the cutoff value for a given metric from a list of peak dataframes.
    
    For recordings > 48 hours: uses first 24 hours as baseline
    For recordings <= 48 hours: uses first half of recording as baseline

    Parameters:
        peak_dataframes (List[pd.DataFrame]): List of DataFrames containing peak data.
        metric (str): Name of the metric to compute the cutoff value for.
        seizure_col (str): Name of the seizure column to exclude from baseline.
        
    Returns:
        float: The cutoff value (maximum of the metric in baseline period * 1.05)
    """
    combined_time = sum((df.index[-1] - df.index[0]).total_seconds() for df in peak_dataframes)
    combined_time = combined_time / 3600  # Convert to hours
    
    if combined_time > 48:
        baseline_period = pd.Timedelta(hours=24)
    else:
        baseline_period = pd.Timedelta(hours=combined_time / 2)

    processed_time = pd.Timedelta(0)
    max_metric_value = float('-inf')
    
    for peak_df in peak_dataframes:
        if processed_time >= baseline_period:
            break
            
        # Filter out seizure periods for baseline calculation
        baseline_df = peak_df[peak_df[seizure_col] == 0]
        baseline_df = baseline_df.loc[baseline_df.index[0]:baseline_df.index[0] + baseline_period - processed_time]
        processed_time += baseline_df.index[-1] - baseline_df.index[0]
        
        if not baseline_df.empty and metric in baseline_df.columns:
            current_max = baseline_df[metric].max()
            if not np.isnan(current_max):
                max_metric_value = max(max_metric_value, current_max)
    
    return max_metric_value * 1.05 if max_metric_value != float('-inf') else 0.0

def compute_cutoff_values(peak_dataframes: List[pd.DataFrame], metrics: List[str], 
                         seizure_col: str = "seizure_for_eval") -> Dict[str, float]:
    """
    Calculate cutoff values for multiple metrics from a list of peak dataframes.
    
    Parameters:
        peak_dataframes (List[pd.DataFrame]): List of DataFrames containing peak data.
        metrics (List[str]): List of metric names to calculate cutoff values for.
        seizure_col (str): Name of the seizure column to exclude from baseline.
        
    Returns:
        Dict[str, float]: Dictionary with metric names as keys and cutoff values as values.
    """
    cutoff_values = {}
    for metric in metrics:
        cutoff_values[metric] = compute_cutoff_value(peak_dataframes=peak_dataframes, 
                                                   metric=metric, seizure_col=seizure_col)
    return cutoff_values

def apply_refractory_prediction(peaks_df: pd.DataFrame, parameter: str, cutoff: float, 
                               refractory_minutes: int = 3) -> pd.DataFrame:
    """
    Apply threshold-based prediction with refractory period.
    
    Parameters:
        peaks_df (pd.DataFrame): DataFrame with time index and parameter column
        parameter (str): Column name to threshold
        cutoff (float): Threshold value for prediction
        refractory_minutes (int): Minutes for refractory period
        
    Returns:
        pd.DataFrame: DataFrame with prediction column added
    """
    result = peaks_df.copy()
    
    # Initialize columns
    result['refractory'] = 0
    result[f'{parameter}_pred'] = 0
    
    # Get indices exceeding threshold
    above_threshold = result[parameter] > cutoff
    
    if not above_threshold.any():
        result.drop(columns=['refractory'], inplace=True)
        return result
    
    # Process indices in time order
    threshold_times = result.index[above_threshold].sort_values()
    
    for detection_time in threshold_times:
        # Skip if already in refractory period
        if result.at[detection_time, 'refractory'] == 1:
            continue
            
        # Mark as prediction
        result.at[detection_time, f'{parameter}_pred'] = 1
        
        # Calculate refractory period end
        refractory_end = detection_time + pd.Timedelta(minutes=refractory_minutes)
        
        # Get all indices in refractory period
        refractory_mask = (result.index >= detection_time) & (result.index <= refractory_end)
        
        # Set refractory flag for all affected rows at once
        result.loc[refractory_mask, 'refractory'] = 1
    
    result.drop(columns=['refractory'], inplace=True)
    return result

def apply_refractory_ensemble_prediction(peaks_df: pd.DataFrame, parameters: List[str], 
                                       cutoffs: List[float], refractory_minutes: int = 3) -> pd.DataFrame:
    """
    Apply ensemble prediction with refractory period in a vectorized way
    
    Parameters:
        peaks_df (pd.DataFrame): DataFrame with time index and parameter columns
        parameters (List[str]): Column names to threshold
        cutoffs (List[float]): Threshold values for each parameter
        refractory_minutes (int): Minutes for refractory period
        
    Returns:
        pd.DataFrame: DataFrame with prediction column added
    """
    result = peaks_df.copy()

    pred_col = f"en-{parameters[0]}x{parameters[1]}_pred"
    
    # Initialize columns
    result['refractory'] = 0
    result[pred_col] = 0
    
    # Get indices exceeding thresholds for each parameter
    above_thresholds = [result[param] > cutoff for param, cutoff in zip(parameters, cutoffs)]
    
    # Check if any Series has any True values
    if not any(series.any() for series in above_thresholds):
        result.drop(columns=['refractory'], inplace=True)
        return result
    
    # Process indices in time order
    threshold_times = result.index[np.any(np.vstack([series.values for series in above_thresholds]), axis=0)].sort_values()
    
    for detection_time in threshold_times:
        # Skip if already in refractory period
        if result.at[detection_time, 'refractory'] == 1:
            continue
            
        # Mark as prediction
        result.at[detection_time, pred_col] = 1
        
        # Calculate refractory period end
        refractory_end = detection_time + pd.Timedelta(minutes=refractory_minutes)
        
        # Get all indices in refractory period
        refractory_mask = (result.index >= detection_time) & (result.index <= refractory_end)
        
        # Set refractory flag for all affected rows at once
        result.loc[refractory_mask, 'refractory'] = 1
    
    result.drop(columns=['refractory'], inplace=True)
    return result

def calculate_prediction_count(peaks: pd.DataFrame, pred_col: str, 
                             excluded_seizure_padding: Tuple[int, int]) -> int:
    """
    Calculates the number of prediction events in the peaks DataFrame.
    Excludes predictions made during seizures that were filtered out.
    
    Parameters:
        peaks (pd.DataFrame): DataFrame containing the peaks data.
        pred_col (str): The name of the column indicating predicted seizure events.
        excluded_seizure_padding (Tuple[int, int]): Number of peaks to exclude around filtered-out seizures.
        
    Returns:
        int: The number of prediction events.
    """
    peaks_copy = peaks.copy()
    
    if pred_col not in peaks_copy.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in the DataFrame.")
    
    # If both seizure columns don't exist, return simple count
    if 'seizure' not in peaks_copy.columns or 'seizure_unfiltered' not in peaks_copy.columns:
        return int(peaks_copy[pred_col].sum())
    
    # Create a mask for filtered-out seizures (seizure_unfiltered=1 but seizure=0)
    filtered_out_seizure_mask = (peaks_copy['seizure_unfiltered'] == 1) & (peaks_copy['seizure'] == 0)
    
    # Group consecutive filtered-out seizures
    peaks_copy['filtered_out_group'] = (filtered_out_seizure_mask != filtered_out_seizure_mask.shift()).cumsum()
    filtered_out_groups = peaks_copy[filtered_out_seizure_mask].groupby('filtered_out_group')
    
    exclusion_mask = pd.Series(False, index=peaks_copy.index)
    for _, group in filtered_out_groups:
        # Get the start and end indices of the filtered-out seizure group
        start_idx = group.index[0]
        end_idx = group.index[-1]
        
        # Get the positions of these indices in the original DataFrame
        start_pos = peaks_copy.index.get_loc(start_idx)
        end_pos = peaks_copy.index.get_loc(end_idx)
        
        # Calculate the positions before and after
        before_pos = max(0, start_pos - excluded_seizure_padding[0])
        after_pos = min(len(peaks_copy) - 1, end_pos + excluded_seizure_padding[1])
        
        # Get the corresponding indices
        before_idx = peaks_copy.index[before_pos]
        after_idx = peaks_copy.index[after_pos]
        
        # Mark the entire range for exclusion
        exclusion_mask.loc[before_idx:after_idx] = True
    
    # Get all rows where pred_col is 1 and not in the extended exclusion zone
    valid_pred_rows = peaks_copy.loc[(peaks_copy[pred_col] == 1) & (~exclusion_mask)]
    
    return len(valid_pred_rows)

def calculate_correct_predictions(peaks: pd.DataFrame, pred_col: str) -> int:
    """
    Calculates the number of correct predictions based on pred_col and "seizure_for_eval".

    Parameters:
        peaks (pd.DataFrame): DataFrame containing the peaks data with seizure and prediction columns.
        pred_col (str): Name of the prediction column.

    Returns:
        int: The number of correct predictions.
    """
    peaks_copy = peaks.copy()
    
    if pred_col not in peaks_copy.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in the DataFrame.")
    
    # Use seizure_for_eval if available, otherwise fall back to seizure
    seizure_col = 'seizure_for_eval' if 'seizure_for_eval' in peaks_copy.columns else 'seizure'
    
    if seizure_col not in peaks_copy.columns:
        return 0
    
    peaks_copy['seizure_group'] = (peaks_copy[seizure_col] != peaks_copy[seizure_col].shift()).cumsum()
    
    # Count correctly identified seizure episodes
    seizure_groups = peaks_copy[peaks_copy[seizure_col] == 1].groupby('seizure_group')
    
    # Handle case where there are no seizure groups
    if len(seizure_groups) == 0:
        return 0

    results = seizure_groups.apply(lambda group: group[pred_col].eq(1).any())
    
    # Ensure result is a scalar value
    try:
        correct_predictions = int(results.sum())
    except (TypeError, ValueError):
        correct_predictions = float(results.sum()) if hasattr(results, 'sum') else 0

    return correct_predictions

def get_seizure_count_from_peaks(peaks: pd.DataFrame, seizure_col: str = 'seizure') -> int:
    """
    Counts the number of distinct seizure episodes in peaks DataFrame.
    
    Parameters:
        peaks (pd.DataFrame): DataFrame with seizure column
        seizure_col (str): Name of seizure column
        
    Returns:
        int: Number of seizure episodes
    """
    if seizure_col not in peaks.columns:
        return 0
    
    # Group consecutive seizure periods
    seizure_groups = (peaks[seizure_col] != peaks[seizure_col].shift()).cumsum()
    seizure_episodes = peaks[peaks[seizure_col] == 1].groupby(seizure_groups)
    
    return len(seizure_episodes)