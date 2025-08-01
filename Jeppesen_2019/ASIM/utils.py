from nptdms import TdmsFile
import pandas as pd
import numpy as np
import aarhus.config.config as cfg
import os


def peak_detection_elgendi(ecg_data, sampling_rate, low = 8, high = 20, order = 3, w1factor=0.12, w2factor=0.65, beta=0.08):
    """
    Detects R-peaks in ECG data using the Elgendi method.
    Parameters:
        ecg_data (list or np.array): The ECG signal data.
        sampling_rate (float): The sampling rate of the ECG data in Hz.
        low (float, optional): Low cutoff frequency for bandpass filter. Default is 8 Hz.
        high (float, optional): High cutoff frequency for bandpass filter. Default is 20 Hz.
        order (int, optional): Order of the Butterworth filter. Default is 3.
        w1factor (float, optional): Factor for window size w1. Default is 0.12.
        w2factor (float, optional): Factor for window size w2. Default is 0.65.
        beta (float, optional): Scaling factor for threshold calculation. Default is 0.08.
    Returns:
        list: A list of indices representing the detected R-peaks.
    """

    def _filter_peaks(ecg_data, foundpeaks, sampling_rate, min_rr_distance=0.25):
        """
        Filters detected peaks in ECG data based on minimum RR interval distance.
        Parameters:
            data (list or np.array): The ECG signal data.
            foundpeaks (list or np.array): Indices of detected peaks in the ECG data.
            sampling_rate (float): The sampling rate of the ECG data in Hz.
            min_rr_distance (float, optional): The minimum RR interval distance in seconds. Peaks closer than this distance will be filtered out. Default is 0.25 seconds.
        Returns:
            list: A list of indices representing the filtered peaks.
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
    
    # Bandpass
    nyquist = 0.5 * sampling_rate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band") # 3rd order butterworth filter
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], ecg_data) # remove filter delay
    
    # First Derivative (QRS enhancement)
    diff = np.diff(filtered)
    diff = np.append(diff, diff[-1])
    
    # Squaring (QRS enhancement)
    squared = diff ** 2
    
    # Normalization
    filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
    peaks = np.zeros(len(filtered))
    w1 = int(w1factor * sampling_rate)
    w2 = int(w2factor * sampling_rate)
    maqrs = np.convolve(squared, np.ones(w1), mode="same") / w1 # array where each element is the average of the w1 neighboring elements in the squared array
    mabeat = np.convolve(squared, np.ones(w2), mode="same") / w2 # array where each element is the average of the w1 neighboring elements in the squared array
    alpha = beta * np.mean(squared)
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
                peak = boiarea + np.argmax(filtered[boiarea:i])
                peaks[peak] = 1
    foundpeaks = np.where(peaks == 1)[0]
    
    return _filter_peaks(ecg_data, foundpeaks, sampling_rate)

def get_all_patient_records_as_dict(patient, seizure_labels=None, cutoff=False) -> dict:
    """
    Get all records for a given patient.

    Args:
        patient (int): Patient number.
        seizure_labels (bool): The seizure label type to include. Defaults to None (no seizure labels).
        cutoff (bool): Whether to apply cutoff times (defaults to False).

    Returns:
        dict: Dictionary containing DataFrames for each recording.
    """
    data_dir = os.path.join(
        config["DATAPATH"]["recordings_dir"],
        f"Patient {patient}"
    )
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    enrollment_dirs = [d for d in subdirs if d.lower().startswith("enrollment") or d.lower().startswith("enrolment")]

    # If no enrollment directories exist, assume recording files are directly under the patient's directory
    if not enrollment_dirs:
        enrollment_dirs = [""]

    records = {}
    for enrollment_dir in enrollment_dirs:
        enrollment_path = os.path.join(data_dir, enrollment_dir) if enrollment_dir else data_dir
        recording_dirs = [d for d in os.listdir(enrollment_path) if os.path.isdir(os.path.join(enrollment_path, d))]

        # If no recording directories exist, look for .tdms files directly in the enrollment directory
        if not recording_dirs:
            tdms_file_paths = [
                os.path.join(enrollment_path, file)
                for file in os.listdir(enrollment_path)
                if file.endswith('.tdms')
            ]
            for tdms_file_path in tdms_file_paths:
                kwargs = {
                    "patient": patient,
                    "enrollment": enrollment_dir.split()[-1] if enrollment_dir else None,
                    "recording": None,
                    "cutoff": cutoff
                }
                if seizure_labels is not None:
                    kwargs["type"] = seizure_labels
                    record_df = get_record_with_seizure_labels(**kwargs)
                else:
                    record_df = get_record(**kwargs)
                record_id = str(patient)
                record_id += f"_{enrollment_dir.split()[-1]}" if enrollment_dir != "" else ""
                record_id += "_1"
                records[record_id] = record_df
        else:
            for recording_dir in recording_dirs:
                recording_path = os.path.join(enrollment_path, recording_dir)
                tdms_file_path = next(
                    (os.path.join(recording_path, file) for file in os.listdir(recording_path) if file.endswith('.tdms')),
                    None
                )
                if not tdms_file_path:
                    continue
                kwargs = {
                    "patient": patient,
                    "enrollment": enrollment_dir.split()[-1] if enrollment_dir else None,
                    "recording": int(recording_dir.split()[-1]),
                    "cutoff": cutoff
                }
                if seizure_labels is not None:
                    kwargs["type"] = seizure_labels
                    record_df = get_record_with_seizure_labels(**kwargs)
                else:
                    record_df = get_record(**kwargs)
                
                record_id = str(patient)
                record_id += f"_{enrollment_dir.split()[-1]}" if enrollment_dir != "" else ""
                record_id += f"_{recording_dir.split()[-1]}"
                records[record_id] = record_df
    records = dict(sorted(records.items()))
    return records


def median_filter(rr_intervals):
    """
    Applies a 7-point median filter to the R-R interval series (tachogram).
    
    This filter smooths the R-R intervals by replacing each value with the median
    of the previous seven values (if available).

    Parameters:
        rr_intervals (list or np.array): The R-R intervals in milliseconds.

    Returns:
        np.array: The filtered R-R intervals.
    """

    return pd.Series(rr_intervals).rolling(window=7, center=False, min_periods=1).median().to_numpy()


def apply_seizure_padding(record_df: pd.DataFrame, seizure_padding: tuple, seizure_col: str, new_col_name: str) -> pd.DataFrame:
    """
    Applies seizure padding to the DataFrame based on the specified column name.
    
    Parameters:
        record_df (pd.DataFrame): DataFrame containing the seizure column.
        seizure_padding (tuple): A tuple of two integers indicating the number of peaks before and after the seizure to include.
        new_col_name (str): The name of the new seizure column with the applied padding.

    Returns:
        pd.DataFrame: The DataFrame with applied seizure padding.
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


def get_peak_dataframe(ecg_df: pd.DataFrame, peak_detection_method: str="elgendi", correct_peaks: bool=False, sampling_rate: int=512, seizure_type: str="analysis") -> pd.DataFrame:
    """
    Computes R-R intervals and returns a DataFrame with the results.
    Parameters:
        ecg_df (pd.DataFrame): DataFrame containing ECG data with a 'ecg' column.
        peak_detection_method (str): The method to use for peak detection.
        correct_peaks (bool): Whether to correct the detected peaks.
        sampling_rate (float): The sampling rate of the ECG data in Hz.
        use_median_filter (bool): Whether to apply a median filter to the R-R intervals.
        seizure_type (str): The type of seizures to include in the "seizure" column.
        seizure_padding (tuple): A tuple of two integers indicating the number of peaks before and after the seizure to include in the "seizure_for_cutoff" column. If None, no padding is applied.
    Returns:
        pd.DataFrame: A DataFrame containing the R-R intervals.
    """
    if peak_detection_method in ['elgendi']:
        r_peaks = peak_detection_elgendi(ecg_data=ecg_df['ecg'].values, sampling_rate=sampling_rate)
    r_peaks = ecg_df.index[r_peaks]

    rr_series = (r_peaks.to_series().diff().dropna().dt.total_seconds() * 1000).values
    # rr_series = filter_nn_series(rr_series)
    rr_series_filtered = median_filter(rr_series)

    peak_data = pd.DataFrame({
        'rr_intervals': rr_series,
        'rr_intervals_filtered': rr_series_filtered,
    }, index=r_peaks[1:])
    
    peak_data.index.name = 'r_peak'
    
    peak_data['seizure'] = 0
    sids = get_record_seizure_ids(patient=ecg_df.attrs['patient'], enrollment=ecg_df.attrs['enrollment'], recording=ecg_df.attrs['recording'], type=seizure_type)
    for sid in sids:
        seizure_info = get_seizure_info(sid)
        if seizure_info is None:
            raise ValueError(f"Seizure info not found for patient: {ecg_df.attrs['patient']}, enrollment: {ecg_df.attrs['enrollment']}, recording: {ecg_df.attrs['recording']}, seizure_id: {sid}")
        start_time = seizure_info['start_time']
        end_time = seizure_info['end_time']
        peak_data.loc[(peak_data.index >= start_time) & (peak_data.index <= end_time), 'seizure'] = 1
    peak_data['seizure'] = peak_data['seizure'].astype(int)

    peak_data['seizure_unfiltered'] = 0
    sids = get_record_seizure_ids(patient=ecg_df.attrs['patient'], enrollment=ecg_df.attrs['enrollment'], recording=ecg_df.attrs['recording'], type="usable")
    for sid in sids:
        seizure_info = get_seizure_info(sid)
        if seizure_info is None:
            raise ValueError(f"Seizure info not found for patient: {ecg_df.attrs['patient']}, enrollment: {ecg_df.attrs['enrollment']}, recording: {ecg_df.attrs['recording']}, seizure_id: {sid}")
        start_time = seizure_info['start_time']
        end_time = seizure_info['end_time']
        peak_data.loc[(peak_data.index >= start_time) & (peak_data.index <= end_time), 'seizure_unfiltered'] = 1
    peak_data['seizure_unfiltered'] = peak_data['seizure_unfiltered'].astype(int)

    peak_data.attrs['patient'] = ecg_df.attrs['patient']
    peak_data.attrs['enrollment'] = ecg_df.attrs['enrollment']
    peak_data.attrs['recording'] = ecg_df.attrs['recording']
    peak_data.attrs['record_id'] = ecg_df.attrs['record_id']
    peak_data.attrs['seizure_type'] = seizure_type
    peak_data.attrs['peak_detection_method'] = peak_detection_method
    peak_data.attrs['sampling_rate'] = sampling_rate

    return peak_data


def calculate_sd1_sd2(rr_intervals)-> tuple:
    """
    Calculates SD1 and SD2 for a series of R-R intervals.
    
    Parameters:
        rr_intervals (array-like): Sequence of R-R intervals.
        
    Returns:
        (sd1, sd2): Tuple of SD1 (transverse) and SD2 (longitudinal) values.
    """
    rr_intervals = np.asarray(rr_intervals)
    x1 = rr_intervals[:-1]
    x2 = rr_intervals[1:]
    
    diff1 = (x2 - x1) / np.sqrt(2)
    sum1 = (x2 + x1 - 2 * np.mean(rr_intervals)) / np.sqrt(2)
    
    sd1 = np.std(diff1, ddof=1)
    sd2 = np.std(sum1, ddof=1)
    
    return sd1, sd2


def calculate_csi(rr_intervals, window_size=50, smoothing_window_size=-1)-> tuple:
    """
    Calculates CSI and Modified CSI retrospectively using sliding windows over RR intervals.
    
    Parameters:
        rr_intervals (array-like): Sequence of R-R intervals.
        window_size (int): Window size for the sliding evaluation (e.g., 50 or 100).
        smoothing_window_size (int): Size of the smoothing window for rolling mean. (default: -1, no smoothing)
            If set to -1, no smoothing is applied.
            If set to a positive integer, a rolling mean is applied with that window size.
        
    Returns:
        csi_values: List of CSI values for each window.
        modcsi_values: List of Modified CSI values for each window.
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


def calculate_hr_diff(rr_intervals, window_size=50, smoothing_window_size=-1)-> np.ndarray:
    """
    Calculates the HR-diff feature using second-order central differences 
    over sliding windows of R-R intervals.
    
    Parameters:
        rr_intervals (array-like): Sequence of R-R intervals.
        window_size (int): Number of intervals in each window (e.g., 50 or 100).
        smoothing_window_size (int): Size of the smoothing window for rolling mean. (default: -1, no smoothing)
            If set to -1, no smoothing is applied.
            If set to a positive integer, a rolling mean is applied with that window size.
    
    Returns:
        hr_diff_values: Numpy array of HR-diff values, padded with NaNs for alignment.
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


def calculate_relative_tachogram_slope(rr_intervals, window_size=50, smoothing_window_size=-1)-> np.ndarray:
    """
    Calculates the relative slope of the tachogram using the Least Squares method
    over sliding windows of RR intervals, normalized by the mean RR interval.

    Parameters:
        rr_intervals (array-like): Sequence of R-R intervals.
        window_size (int): Number of intervals in each window (e.g., 50 or 100).
        smoothing_window_size (int): Size of the smoothing window for rolling mean.

    Returns:
        slope_values: Numpy array of normalized slope magnitudes, aligned with RR timeline.
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



def add_feature_combinations(peak_dataframe: pd.DataFrame, *feature_groups: list) -> pd.DataFrame:
    """
    Adds combinations of features to a DataFrame.

    Parameters:
        peak_dataframe (pd.DataFrame): DataFrame containing the features.
        *feature_groups (list): Variable number of lists, each containing feature names to calculate combinations for.

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


def compute_cutoff_value(peak_dataframes: list, metric: str, seizure_col:str='seizure_for_cutoff')-> float:
    """
    Compute the cutoff value for a given metric from a list of peak dataframes.
    
    For recordings > 48 hours: uses first 24 hours as baseline
    For recordings <= 48 hours: uses first half of recording as baseline

    Parameters:
        peak_dataframes (list): List of DataFrames containing peak data.
        metric (str): Name of the metric to compute the cutoff value for.
        
    Returns:
        float: The cutoff value (maximum of the metric in baseline period)
    """
    combined_time = sum((df.index[-1] - df.index[0]).total_seconds() for df in peak_dataframes)
    combined_time = combined_time / 3600
    
    if combined_time > 48:
        baseline_period = pd.Timedelta(hours=24)
    else:
        baseline_period = pd.Timedelta(hours=combined_time / 2)

    processed_time = pd.Timedelta(0)
    max_metric_value = float('-inf')
    for peak_df in peak_dataframes:
        if processed_time >= baseline_period:
            break
        peak_df = peak_df[peak_df[seizure_col] == 0]
        peak_df = peak_df.loc[peak_df.index[0]:peak_df.index[0] + baseline_period - processed_time]
        processed_time += peak_df.index[-1] - peak_df.index[0]
        
        max_metric_value = max(max_metric_value, peak_df[metric].max())
    return max_metric_value * 1.05


def apply_refractory_prediction(peaks_df, parameter, cutoff, refractory_minutes=3):
    """
    Apply threshold-based prediction with refractory period.
    
    Parameters:
        peaks_df: DataFrame with time index and parameter column
        parameter: Column name to threshold
        cutoff: Threshold value for prediction
        refractory_minutes: Minutes for refractory period
        
    Returns:
        DataFrame with prediction and refractory columns added
    """
    # Create copy to avoid modifying original
    result = peaks_df.copy()
    
    # Initialize columns
    result['refractory'] = 0
    result[f'{parameter}_pred'] = 0
    
    # Get indices exceeding threshold
    above_threshold = result[parameter] > cutoff
    
    if not above_threshold.any():
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


def apply_refractory_ensemble_prediction(peaks_df, parameters, cutoffs, refractory_minutes=3):
    """
    Apply ensemble prediction with refractory period in a vectorized way
    
    Parameters:
        peaks_df: DataFrame with time index and parameter columns
        parameters: Column names to threshold
        cutoffs: Threshold values for each parameter
        refractory_minutes: Minutes for refractory period
        
    Returns:
        DataFrame with prediction and refractory columns added
    """
    # Create copy to avoid modifying original
    result = peaks_df.copy()

    pred_col = f"en-{parameters[0]}x{parameters[1]}_pred"
    
    # Initialize columns
    result['refractory'] = 0
    result[pred_col] = 0
    
    # Get indices exceeding thresholds for each parameter
    above_thresholds = [result[param] > cutoff for param, cutoff in zip(parameters, cutoffs)]
    
    # Check if any Series has any True values
    if not any(series.any() for series in above_thresholds):
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

def calculate_prediction_count(peaks: pd.DataFrame, pred_col: str, excluded_seizure_padding: tuple) -> int:
    """
    Calculates the number of prediction events in the peaks DataFrame.
    Excludes predictions made during seizures that were filtered out, 
    as well as a specified amount of peaks around those seizures.
    
    Parameters:
        peaks (pd.DataFrame): DataFrame containing the peaks data.
        pred_col (str): The name of the column indicating predicted seizure events.
        excluded_seizure_padding (tuple): A tuple specifying the number of peaks to exclude around filtered-out seizures.
        
    Returns:
        int: The number of prediction events.
    """
    peaks_copy = peaks.copy()
    
    if pred_col is None:
        pred_cols = [col for col in peaks_copy.columns if 'pred' in col]
        if len(pred_cols) == 0:
            raise ValueError("No prediction columns found in the DataFrame.")
        if len(pred_cols) > 1:
            raise ValueError("Multiple prediction columns found.")
        pred_col = pred_cols[0]

    if pred_col not in peaks_copy.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in the DataFrame.")
    
    # If both seizure columns don't exist, raise error
    if 'seizure' not in peaks_copy.columns or 'seizure_unfiltered' not in peaks_copy.columns:
        raise ValueError("Both 'seizure' and 'seizure_unfiltered' columns are required.")
    
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
    
    if len(valid_pred_rows) == 0:
        return 0
    
    return len(valid_pred_rows)


def calculate_correct_predictions(peaks: pd.DataFrame, pred_col: str) -> int:
    """
    Calculates the number of correct predictions in the peaks DataFrame based on pred_col and the column "seizure_for_eval".

    Parameters:
        peaks (pd.DataFrame): DataFrame containing the peaks data with a seizure column
                             and a prediction column.
        time_period (str): The time period to consider for counting correct predictions.
                        Options are 'combined', 'day', or 'night'.
                        'combined' counts all correct predictions,
                        'day' counts only during the day (7:00-19:00),
                        'night' counts only during the night (19:00-7:00).
                        Default is 'combined'.

    Returns:
        int: The number of correct predictions.
    """
    # Create a proper copy to avoid the SettingWithCopyWarning
    peaks_copy = peaks.copy()
    
    if pred_col is None:
        pred_cols = [col for col in peaks_copy.columns if 'pred' in col]
        if len(pred_cols) == 0:
            raise ValueError("No prediction columns found in the DataFrame.")
        if len(pred_cols) > 1:
            raise ValueError("Multiple prediction columns found.")
        pred_col = pred_cols[0]
    if pred_col not in peaks_copy.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in the DataFrame.")
    
    # Filter by time period first, before creating the seizure_group column
    # if time_period == "day":
    #     # Define day period (7:00-19:00)
    #     day_mask = (peaks_copy.index.hour >= 7) & (peaks_copy.index.hour < 19)
    #     peaks_copy = peaks_copy[day_mask].copy()  # Create another explicit copy after filtering
    # elif time_period == "night":
    #     # Define night period (19:00-7:00)
    #     night_mask = ~((peaks_copy.index.hour >= 7) & (peaks_copy.index.hour < 19))
    #     peaks_copy = peaks_copy[night_mask].copy()  # Create another explicit copy after filtering

    peaks_copy.loc[:, 'seizure_group'] = (peaks_copy['seizure_for_eval'] != peaks_copy['seizure_for_eval'].shift()).cumsum()
    
    # Count correctly identified seizure episodes
    seizure_groups = peaks_copy[peaks_copy['seizure_for_eval'] == 1].groupby('seizure_group')
    
    # Handle case where there are no seizure groups
    if len(seizure_groups) == 0:
        return 0

    results = seizure_groups.apply(lambda group: group[pred_col].eq(1).any())
    
    # Ensure TP is a scalar value
    try:
        TP = int(results.sum())
    except (TypeError, ValueError):
        TP = float(results.sum()) if hasattr(results, 'sum') else 0

    return TP


def compute_cutoff_values(peak_dataframes: list, metrics: list, seizure_col: str = "seizure_for_eval")-> dict:
    """
    Calculate cutoff values for multiple metrics from a list of peak dataframes.
    
    Parameters:
        peak_dataframes (list): List of DataFrames containing peak data.
        metrics (list): List of metric names to calculate cutoff values for.
        
    Returns:
        cutoff_values (dict): Dictionary with metric names as keys and cutoff values as values.
    """
    cutoff_values = {}
    for metric in metrics:
        cutoff_values[metric] = compute_cutoff_value(peak_dataframes=peak_dataframes, metric=metric, seizure_col=seizure_col)
    return cutoff_values