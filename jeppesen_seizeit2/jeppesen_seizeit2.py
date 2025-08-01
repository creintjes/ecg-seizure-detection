"""
Jeppesen 2019 Ansatz adaptiert für SeizeIT2 Dataset
Hauptskript für ECG-basierte Epilepsie-Vorhersage
"""

import os
import gc
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

# Lokale Imports
from config import *
from seizeit2_utils import get_all_patient_records_as_dict, get_all_subjects, validate_subject_data
from feature_extraction import get_peak_dataframe, calculate_csi, calculate_hr_diff, calculate_relative_tachogram_slope, add_feature_combinations
from evaluation_utils import *

# Multiprocessing Setup
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['OPENBLAS_NUM_THREADS'] = '6' 
os.environ['BLAS_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'

def process_subject(subject: str) -> List[Dict]:
    """
    Verarbeitet einen einzelnen Subject aus dem SeizeIT2 Dataset.
    
    Args:
        subject (str): Subject-ID (z.B. 'sub-001')
        
    Returns:
        List[Dict]: Liste der Metriken für alle Parameter dieses Subjects
    """
    gc.collect()
    
    try:
        # Lade alle Recordings für diesen Subject
        subject_records = get_all_patient_records_as_dict(subject)  
        
        if not subject_records:
            tqdm.write(f"⚠️ Keine gültigen Recordings für {subject}")
            return []
            
        peak_dataframes = {}
        
        # Feature-Extraktion für jedes Recording
        for record_id, record in subject_records.items():
            gc.collect()
            
            try:
                # R-Peak Detection + RR-Intervalle
                peak_dataframe = get_peak_dataframe(
                    ecg_df=record, 
                    peak_detection_method=PEAK_DETECTION_METHOD
                )
                
                # Seizure Padding anwenden
                peak_dataframe = apply_seizure_padding(
                    record_df=peak_dataframe, 
                    seizure_padding=SEIZURE_PADDING_CUTOFF, 
                    seizure_col='seizure_unfiltered', 
                    new_col_name='seizure_for_cutoff'
                )
                peak_dataframe = apply_seizure_padding(
                    record_df=peak_dataframe, 
                    seizure_padding=SEIZURE_PADDING_EVAL, 
                    seizure_col='seizure', 
                    new_col_name='seizure_for_eval'
                )
                
                # Feature-Berechnung für verschiedene Window-Größen
                for window_length in WINDOW_LENGTHS:
                    # CSI und ModCSI
                    csi, modcsi = calculate_csi(
                        rr_intervals=peak_dataframe['rr_intervals'].values, 
                        window_size=window_length
                    )
                    peak_dataframe[f'csi_{window_length}'] = csi
                    peak_dataframe[f'modcsi_{window_length}'] = modcsi
                    del csi, modcsi

                    # CSI und ModCSI für gefilterte RR-Intervalle
                    csi_filtered, modcsi_filtered = calculate_csi(
                        rr_intervals=peak_dataframe['rr_intervals_filtered'].values, 
                        window_size=window_length
                    )
                    peak_dataframe[f'csi_{window_length}_filtered'] = csi_filtered
                    peak_dataframe[f'modcsi_{window_length}_filtered'] = modcsi_filtered
                    del csi_filtered, modcsi_filtered

                    # HR-diff
                    peak_dataframe[f'hr_diff_{window_length}'] = calculate_hr_diff(
                        peak_dataframe['rr_intervals'].values, 
                        window_size=window_length
                    )
                    peak_dataframe[f'hr_diff_{window_length}_filtered'] = calculate_hr_diff(
                        peak_dataframe['rr_intervals_filtered'].values, 
                        window_size=window_length
                    )

                    # Relative Tachogram Slope
                    peak_dataframe[f'slope_{window_length}'] = calculate_relative_tachogram_slope(
                        peak_dataframe['rr_intervals_filtered'].values, 
                        window_size=window_length
                    )

                # Feature-Kombinationen
                feature_combinations = [
                    ['modcsi_100', 'slope_100'], 
                    ['csi_100', 'slope_100'], 
                    ['modcsi_100_filtered', 'slope_100'], 
                    ['csi_100_filtered', 'slope_100'],
                    ['modcsi_50', 'slope_50'], 
                    ['csi_50', 'slope_50'], 
                    ['modcsi_50_filtered', 'slope_50'], 
                    ['csi_50_filtered', 'slope_50']
                ]
                peak_dataframe = add_feature_combinations(peak_dataframe, *feature_combinations)
                peak_dataframes[record_id] = peak_dataframe
                
            except Exception as e:
                tqdm.write(f"⚠️ Fehler bei Recording {record_id}: {e}")
                continue

        if not peak_dataframes:
            tqdm.write(f"⚠️ Keine erfolgreichen Recordings für {subject}")
            return []

        # Evaluation - Einzelne Parameter
        parameter_metrics = {}
        parameters_to_evaluate = [col for col in peak_dataframe.columns 
                                if col.startswith('csi_') or col.startswith('modcsi_') 
                                or col.startswith('hr_diff_') or col.startswith('&')]

        for parameter in parameters_to_evaluate:
            gc.collect()

            try:
                subject_seizure_prediction_count = 0
                subject_correct_prediction_count = 0
                subject_seizure_count = 0

                # Cutoff-Wert berechnen
                cutoff = compute_cutoff_value(
                    peak_dataframes=list(peak_dataframes.values()), 
                    metric=parameter, 
                    seizure_col='seizure_for_cutoff'
                )

                for record_id, peak_dataframe in peak_dataframes.items():
                    peaks = peak_dataframe[[parameter, 'seizure', 'seizure_unfiltered', 'seizure_for_cutoff', 'seizure_for_eval']].copy()

                    subject_seizure_count += get_seizure_count_from_peaks(peaks, 'seizure')

                    # Refractory Prediction anwenden
                    peaks = apply_refractory_prediction(
                        peaks_df=peaks, 
                        parameter=parameter, 
                        cutoff=cutoff, 
                        refractory_minutes=REFRACTORY_PERIOD_MINUTES
                    )
                    
                    subject_seizure_prediction_count += calculate_prediction_count(
                        peaks=peaks, 
                        pred_col=f"{parameter}_pred", 
                        excluded_seizure_padding=SEIZURE_PADDING_EVAL
                    )
                    subject_correct_prediction_count += calculate_correct_predictions(
                        peaks=peaks, 
                        pred_col=f"{parameter}_pred"
                    )

                # Metriken berechnen
                combined_record_time_days = sum([
                    (record.index[-1] - record.index[0]).total_seconds() / (24*60*60) 
                    for record in peak_dataframes.values()
                ])
                false_alarms = subject_seizure_prediction_count - subject_correct_prediction_count
                combined_fad = false_alarms / combined_record_time_days if combined_record_time_days > 0 else 0

                combined_sensitivity = subject_correct_prediction_count / subject_seizure_count if subject_seizure_count > 0 else 0
                combined_precision = subject_correct_prediction_count / subject_seizure_prediction_count if subject_seizure_prediction_count > 0 else 0

                parameter_metrics[parameter] = {
                    "cutoff": cutoff,
                    "sensitivity": combined_sensitivity,
                    "precision": combined_precision,
                    "F1_score": 2 * (combined_precision * combined_sensitivity) / (combined_precision + combined_sensitivity) if (combined_precision + combined_sensitivity) > 0 else 0,
                    "FAD": combined_fad,
                    "num_pred_seizures": subject_seizure_prediction_count,
                    "num_correct_pred_seizures": subject_correct_prediction_count,
                    "num_seizures": subject_seizure_count,
                }
                
            except Exception as e:
                tqdm.write(f"⚠️ Fehler bei Parameter {parameter} für {subject}: {e}")
                continue

        # Ensemble Evaluation
        for combination in ENSEMBLE_COMBINATIONS:
            gc.collect()

            try:
                subject_seizure_prediction_count = 0
                subject_correct_prediction_count = 0
                subject_seizure_count = 0

                # Cutoffs für Ensemble berechnen
                cutoffs = compute_cutoff_values(
                    peak_dataframes=list(peak_dataframes.values()), 
                    metrics=combination, 
                    seizure_col='seizure_for_eval'
                )

                for record_id, peak_dataframe in peak_dataframes.items():
                    peaks = peak_dataframe[[*combination, 'seizure', 'seizure_unfiltered', 'seizure_for_cutoff', 'seizure_for_eval']].copy()

                    subject_seizure_count += get_seizure_count_from_peaks(peaks, 'seizure')

                    # Ensemble Refractory Prediction
                    peaks = apply_refractory_ensemble_prediction(
                        peaks_df=peaks, 
                        parameters=combination, 
                        cutoffs=list(cutoffs.values()), 
                        refractory_minutes=REFRACTORY_PERIOD_MINUTES
                    )
                    
                    pred_col = f"en-{combination[0]}x{combination[1]}_pred"
                    subject_seizure_prediction_count += calculate_prediction_count(
                        peaks=peaks, 
                        pred_col=pred_col, 
                        excluded_seizure_padding=SEIZURE_PADDING_EVAL
                    )
                    subject_correct_prediction_count += calculate_correct_predictions(
                        peaks=peaks, 
                        pred_col=pred_col
                    )
                
                # Ensemble-Metriken berechnen
                combined_record_time_days = sum([
                    (record.index[-1] - record.index[0]).total_seconds() / (24*60*60) 
                    for record in peak_dataframes.values()
                ])
                false_alarms = subject_seizure_prediction_count - subject_correct_prediction_count
                combined_fad = false_alarms / combined_record_time_days if combined_record_time_days > 0 else 0

                combined_sensitivity = subject_correct_prediction_count / subject_seizure_count if subject_seizure_count > 0 else 0
                combined_precision = subject_correct_prediction_count / subject_seizure_prediction_count if subject_seizure_prediction_count > 0 else 0

                parameter_metrics[f"en-{combination[0]}x{combination[1]}"] = {
                    "cutoff": list(cutoffs.values()),  # Liste für Ensemble
                    "sensitivity": combined_sensitivity,
                    "precision": combined_precision,
                    "F1_score": 2 * (combined_precision * combined_sensitivity) / (combined_precision + combined_sensitivity) if (combined_precision + combined_sensitivity) > 0 else 0,
                    "FAD": combined_fad,
                    "num_pred_seizures": subject_seizure_prediction_count,
                    "num_correct_pred_seizures": subject_correct_prediction_count,
                    "num_seizures": subject_seizure_count,
                }
                
            except Exception as e:
                tqdm.write(f"⚠️ Fehler bei Ensemble {combination} für {subject}: {e}")
                continue

        # Ergebnisse für diesen Subject zusammenstellen
        subject_metrics = []
        for parameter, metrics in parameter_metrics.items():
            subject_metrics.append({
                "subject": subject,
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
            })
            
        return subject_metrics
        
    except Exception as e:
        tqdm.write(f"❌ Kritischer Fehler bei {subject}: {e}")
        return []

def main():
    """Hauptfunktion"""
    # Konfiguration validieren
    try:
        validate_config()
    except Exception as e:
        print(f"❌ Konfigurationsfehler: {e}")
        return
    
    # Run-Name generieren
    run_name = f"{RUN_PREFIX}_detection_{PEAK_DETECTION_METHOD}_padding_{SEIZURE_PADDING_CUTOFF[0]}-{SEIZURE_PADDING_CUTOFF[1]}-{SEIZURE_PADDING_EVAL[0]}-{SEIZURE_PADDING_EVAL[1]}.csv"
    
    print(f"🚀 Starte Jeppesen SeizeIT2 Analyse")
    print(f"📊 Run: {run_name}")
    print(f"🔍 Peak Detection: {PEAK_DETECTION_METHOD}")
    print(f"⚕️  Seizure Padding Cutoff: {SEIZURE_PADDING_CUTOFF}")
    print(f"📈 Seizure Padding Eval: {SEIZURE_PADDING_EVAL}")
    print()

    # Alle Subjects laden und validieren
    all_subjects = get_all_subjects()
    valid_subjects = [subj for subj in all_subjects if validate_subject_data(subj)]
    
    print(f"📂 Gefunden: {len(all_subjects)} Subjects, {len(valid_subjects)} gültig")
    
    if not valid_subjects:
        print("❌ Keine gültigen Subjects gefunden!")
        return

    # Parallelverarbeitung
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_subject, valid_subjects), 
            total=len(valid_subjects), 
            unit="subject", 
            desc="Verarbeite Subjects", 
            position=0
        ))

    # Ergebnisse zusammenführen
    all_subject_metrics = []
    for subject_metrics in results:
        if subject_metrics:  # Nur erfolgreiche Subjects
            all_subject_metrics.extend(subject_metrics)

    if not all_subject_metrics:
        print("❌ Keine erfolgreichen Ergebnisse!")
        return

    # Ergebnisse speichern
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_df = pd.DataFrame(all_subject_metrics)
    file_path = RESULTS_DIR / run_name
    metrics_df.to_csv(file_path, index=False)
    
    print(f"✅ Ergebnisse gespeichert: {file_path}")
    print(f"📊 Verarbeitete Subjects: {len(set(metrics_df['subject']))}")
    print(f"📋 Gesamte Datenpunkte: {len(metrics_df)}")
    
    # Kurze Zusammenfassung
    if len(metrics_df) > 0:
        print("\n📈 Kurze Zusammenfassung:")
        print(f"   Mittlere Sensitivität: {metrics_df['sensitivity'].mean():.3f}")
        print(f"   Mittlere FAD: {metrics_df['FAD'].mean():.3f}")
        print(f"   Responder-Rate: {metrics_df['responder'].mean():.3f}")

if __name__ == "__main__":
    main()