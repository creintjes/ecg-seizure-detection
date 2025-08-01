import os
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['OPENBLAS_NUM_THREADS'] = '6'
os.environ['BLAS_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
import utils


run_prefix = "40_no_prefiltering"

peak_detection_method = 'elgendi'
seizure_type = "analysis"

seizure_padding_cutoff = (120, 120)
seizure_padding_eval = (120, 100)

run_name = f"{run_prefix}_detection_{peak_detection_method}_padding_{seizure_padding_cutoff[0]}-{seizure_padding_cutoff[1]}-{seizure_padding_eval[0]}-{seizure_padding_eval[1]}.csv"


def process_patient(patient):
    gc.collect()
    patient_records = utils.get_all_patient_records_as_dict(patient, seizure_labels=True) # im Format {record_id: record_dataframe}
    peak_dataframes = {}

    # Feature calculation
    for record_id, record in patient_records.items():
        gc.collect()

        # Cutoff the record to the start and end cutoffs
        start_cutoff, end_cutoff = (0, 0) # no cutoff if records do not need to be cut
        record = record.loc[record.index[0] + pd.Timedelta(minutes=start_cutoff):record.index[-1] - pd.Timedelta(minutes=end_cutoff)]

        # Peak detection + seizure columns
        peak_dataframe = utils.get_peak_dataframe(ecg_df=record, sampling_rate=512, peak_detection_method=peak_detection_method, seizure_type=seizure_type)
        peak_dataframe = utils.apply_seizure_padding(record_df=peak_dataframe, seizure_padding=seizure_padding_cutoff, seizure_col='seizure_unfiltered', new_col_name='seizure_for_cutoff')
        peak_dataframe = utils.apply_seizure_padding(record_df=peak_dataframe, seizure_padding=seizure_padding_eval, seizure_col='seizure', new_col_name='seizure_for_eval')
        

        for window_length in [50, 100]:
            csi, modcsi = utils.calculate_csi(rr_intervals=peak_dataframe['rr_intervals'].values, window_size=window_length)
            peak_dataframe[f'csi_{window_length}'] = csi
            peak_dataframe[f'modcsi_{window_length}'] = modcsi
            del csi, modcsi

            csi_filtered, modcsi_filtered = utils.calculate_csi(rr_intervals=peak_dataframe['rr_intervals_filtered'].values, window_size=window_length)
            peak_dataframe[f'csi_{window_length}_filtered'] = csi_filtered
            peak_dataframe[f'modcsi_{window_length}_filtered'] = modcsi_filtered
            del csi_filtered, modcsi_filtered

            peak_dataframe[f'hr_diff_{window_length}'] = utils.calculate_hr_diff(peak_dataframe['rr_intervals'].values, window_size=window_length)
            peak_dataframe[f'hr_diff_{window_length}_filtered'] = utils.calculate_hr_diff(peak_dataframe['rr_intervals_filtered'].values, window_size=window_length)

            peak_dataframe[f'slope_{window_length}'] = utils.calculate_relative_tachogram_slope(peak_dataframe['rr_intervals_filtered'].values, window_size=window_length)

        # feature combinations
        peak_dataframe = utils.add_feature_combinations(peak_dataframe, ['modcsi_100', 'slope_100'], ['csi_100', 'slope_100'], ['modcsi_100_filtered', 'slope_100'], ['csi_100_filtered', 'slope_100'], ['modcsi_50', 'slope_50'], ['csi_50', 'slope_50'], ['modcsi_50_filtered', 'slope_50'], ['csi_50_filtered', 'slope_50'])
        peak_dataframes[record_id] = peak_dataframe

    # Evaluation
    parameter_metrics = {}
    parameters_to_evaluate = [col for col in peak_dataframe.columns if col.startswith('csi_') or col.startswith('modcsi_') or col.startswith('hr_diff_') or col.startswith('&')]
    for parameter in parameters_to_evaluate:
        gc.collect()

        patient_seizure_prediction_count = 0
        patient_correct_prediction_count = 0
        patient_seizure_count = 0


        cutoff = utils.compute_cutoff_value(peak_dataframes=list(peak_dataframes.values()), metric=parameter, seizure_col='seizure_for_cutoff')

        for record_id, peak_dataframe in peak_dataframes.items():
            peaks = peak_dataframe[[parameter, 'seizure', 'seizure_unfiltered', 'seizure_for_cutoff', 'seizure_for_eval']].copy()

            patient_seizure_count += len(utils.get_record_seizure_ids(peaks.attrs['patient'], peaks.attrs['enrollment'], peaks.attrs['recording'], seizure_type))

            peaks = utils.apply_refractory_prediction(peaks_df=peaks, parameter=parameter, cutoff=cutoff, refractory_minutes=3)
            
            patient_seizure_prediction_count += utils.calculate_prediction_count(peaks=peaks, pred_col=f"{parameter}_pred", excluded_seizure_padding=seizure_padding_eval)
            patient_correct_prediction_count += utils.calculate_correct_predictions(peaks=peaks, pred_col=f"{parameter}_pred")



            
        
        combined_record_time_days = sum([(record.index[-1] - record.index[0]).total_seconds() / (24*60*60) for record in peak_dataframes.values()])
        false_alarms = patient_seizure_prediction_count - patient_correct_prediction_count
        combined_fad = false_alarms / combined_record_time_days if combined_record_time_days > 0 else 0

        combined_sensitivity = patient_correct_prediction_count / patient_seizure_count if patient_seizure_count > 0 else 0
        combined_precision = patient_correct_prediction_count / patient_seizure_prediction_count if patient_seizure_prediction_count > 0 else 0
        

        parameter_metrics[parameter] = {
            "cutoff": cutoff,
            "sensitivity": combined_sensitivity,
            "precision": combined_precision,
            "F1_score": 2 * (combined_precision * combined_sensitivity) / (combined_precision + combined_sensitivity) if (combined_precision + combined_sensitivity) > 0 else 0,
            "FAD": combined_fad,
            "num_pred_seizures": patient_seizure_prediction_count,
            "num_correct_pred_seizures": patient_correct_prediction_count,
            "num_seizures": patient_seizure_count,
        }

    # ensemble evaluation
    ensemble_combinations = [['modcsi_100xslope_100', 'modcsi_100_filteredxslope_100'], ['modcsi_100xslope_100', 'csi_100xslope_100'], ['modcsi_100xslope_100', 'csi_100_filteredxslope_100'], ['modcsi_100_filteredxslope_100', 'csi_100xslope_100'], ['modcsi_100_filteredxslope_100', 'csi_100_filteredxslope_100'], ['csi_100xslope_100', 'csi_100_filteredxslope_100']]
    for combination in ensemble_combinations:
        gc.collect()

        patient_seizure_prediction_count = 0
        patient_correct_prediction_count = 0
        patient_seizure_count = 0


        cutoffs = utils.compute_cutoff_values(peak_dataframes=list(peak_dataframes.values()), metrics=combination, seizure_col='seizure_for_eval')

        for record_id, peak_dataframe in peak_dataframes.items():
            peaks = peak_dataframe[[*combination, 'seizure', 'seizure_unfiltered', 'seizure_for_cutoff', 'seizure_for_eval']].copy()

            patient_seizure_count += len(utils.get_record_seizure_ids(peaks.attrs['patient'], peaks.attrs['enrollment'], peaks.attrs['recording'], seizure_type))

            peaks = utils.apply_refractory_ensemble_prediction(peaks_df=peaks, parameters=combination, cutoffs=list(cutoffs.values()), refractory_minutes=3)
            
            patient_seizure_prediction_count += utils.calculate_prediction_count(peaks=peaks, pred_col=f"en-{combination[0]}x{combination[1]}_pred", excluded_seizure_padding=seizure_padding_eval)
            patient_correct_prediction_count += utils.calculate_correct_predictions(peaks=peaks, pred_col=f"en-{combination[0]}x{combination[1]}_pred")
            
        
        combined_record_time_days = sum([(record.index[-1] - record.index[0]).total_seconds() / (24*60*60) for record in peak_dataframes.values()])
        false_alarms = patient_seizure_prediction_count - patient_correct_prediction_count
        combined_fad = false_alarms / combined_record_time_days if combined_record_time_days > 0 else 0

        combined_sensitivity = patient_correct_prediction_count / patient_seizure_count if patient_seizure_count > 0 else 0
        combined_precision = patient_correct_prediction_count / patient_seizure_prediction_count if patient_seizure_prediction_count > 0 else 0
        

        parameter_metrics[f"en-{combination[0]}x{combination[1]}"] = {
            "cutoff": cutoff,
            "sensitivity": combined_sensitivity,
            "precision": combined_precision,
            "F1_score": 2 * (combined_precision * combined_sensitivity) / (combined_precision + combined_sensitivity) if (combined_precision + combined_sensitivity) > 0 else 0,
            "FAD": combined_fad,
            "num_pred_seizures": patient_seizure_prediction_count,
            "num_correct_pred_seizures": patient_correct_prediction_count,
            "num_seizures": patient_seizure_count,
        }

    patient_metrics = []
    for parameter, metrics in parameter_metrics.items():
        patient_metrics.append({
            "patient": patient,
            "responder": 1 if metrics['sensitivity'] >= (2/3) else 0,
            "parameter": parameter,
            "cutoff": metrics["cutoff"],
            "sensitivity": metrics["sensitivity"],
            "precision": metrics["precision"],
            "F1_score": metrics["F1_score"],
            "FAD": metrics["FAD"],
            "num_pred_seizures": metrics["num_pred_seizures"],
            "num_correct_pred_seizures": metrics["num_correct_pred_seizures"],
            "num_seizures": metrics["num_seizures"],
            "seizure_prediction_distances": metrics["seizure_prediction_distances"],
            "seizure_prediction_onset_distances": metrics["seizure_prediction_onset_distances"]
        })
    return patient_metrics


if __name__ == "__main__":
    tqdm.write(f"Run {run_name}")
    tqdm.write(f"Peak detection method: {peak_detection_method}")
    tqdm.write(f"Seizure type: {seizure_type}")
    tqdm.write(f"Seizure padding for cutoff: {seizure_padding_cutoff}")
    tqdm.write(f"Seizure padding for evaluation: {seizure_padding_eval}")
    tqdm.write("")

    with ProcessPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(process_patient, range(1, 44)), total=43, unit="patient", desc="Calculating metrics", position=0))

    all_patient_metrics = []
    for patient_metrics in results:
        all_patient_metrics.extend(patient_metrics)
    

    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)

    metrics_df = pd.DataFrame(all_patient_metrics)
    file_path = os.path.join(result_dir, run_name)
    metrics_df.to_csv(file_path, index=False)
    tqdm.write(f"Saved results to {file_path}")
