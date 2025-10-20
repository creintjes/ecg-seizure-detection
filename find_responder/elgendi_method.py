def peak_detection_elgendi(ecg_data, sampling_rate, low = 8, high = 20, order = 3, w1factor=0.12, w2factor=0.65, beta=0.08):
    """
    Detects R-peaks in ECG data using the Elgendi method.
    Parameters:
        ecg_data (list or np.array): The ECG signal data.
        sampling_rate (int): The sampling rate of the ECG data in Hz.
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
            sampling_rate (int): The sampling rate of the ECG data in Hz.
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