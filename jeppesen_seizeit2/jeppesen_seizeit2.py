"""
Jeppesen 2019 Ansatz adaptiert für SeizeIT2 Dataset
Hauptskript für ECG-basierte Epilepsie-Vorhersage
"""

import os
import gc
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from datetime import datetime
import time

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
    start_time = time.time()
    gc.collect()
    
    try:
        tqdm.write(f"🔄 Beginne Verarbeitung von {subject}")
        
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
            
        elapsed_time = time.time() - start_time
        tqdm.write(f"✅ {subject} erfolgreich verarbeitet in {elapsed_time:.1f}s ({len(subject_metrics)} Parameter)")
        return subject_metrics
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        tqdm.write(f"❌ Kritischer Fehler bei {subject} nach {elapsed_time:.1f}s: {e}")
        import traceback
        tqdm.write(f"🔍 Traceback: {traceback.format_exc()}")
        return []
    finally:
        gc.collect()

def load_checkpoint(checkpoint_path):
    """Lädt vorhandene Checkpoint-Daten"""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data.get('completed_subjects', []), checkpoint_data.get('results', [])
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Checkpoint-Datei: {e}")
            return [], []
    return [], []

def save_checkpoint(checkpoint_path, completed_subjects, results):
    """Speichert Checkpoint-Daten"""
    checkpoint_data = {
        'completed_subjects': completed_subjects,
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'total_completed': len(completed_subjects)
    }
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️ Fehler beim Speichern der Checkpoint-Datei: {e}")

def process_subjects_with_checkpoints(valid_subjects, checkpoint_path, final_csv_path):
    """Verarbeitet Subjects mit Checkpoint-Funktionalität"""
    # Checkpoint laden
    completed_subjects, existing_results = load_checkpoint(checkpoint_path)
    
    # Noch zu verarbeitende Subjects ermitteln
    remaining_subjects = [s for s in valid_subjects if s not in completed_subjects]
    
    if not remaining_subjects:
        print("✅ Alle Subjects bereits verarbeitet!")
        return existing_results
    
    print(f"📊 Fortschritt: {len(completed_subjects)}/{len(valid_subjects)} Subjects bereits verarbeitet")
    print(f"🔄 Verbleibend: {len(remaining_subjects)} Subjects")
    
    all_results = existing_results.copy()
    
    # Batch-Verarbeitung mit kleineren Gruppen für häufigere Checkpoints
    batch_size = min(MAX_WORKERS, CHECKPOINT_BATCH_SIZE)  # Batch-Größe aus Konfiguration
    
    for i in range(0, len(remaining_subjects), batch_size):
        batch_subjects = remaining_subjects[i:i+batch_size]
        
        print(f"\n🔄 Verarbeite Batch {i//batch_size + 1}/{(len(remaining_subjects)-1)//batch_size + 1}: {batch_subjects}")
        
        # Batch parallel verarbeiten
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(batch_subjects))) as executor:
            future_to_subject = {executor.submit(process_subject, subject): subject for subject in batch_subjects}
            
            batch_results = []
            for future in as_completed(future_to_subject):
                subject = future_to_subject[future]
                try:
                    result = future.result(timeout=SUBJECT_TIMEOUT_MINUTES * 60)  # Timeout aus Konfiguration
                    if result:  # Nur erfolgreiche Ergebnisse
                        batch_results.extend(result)
                        completed_subjects.append(subject)
                        print(f"✅ {subject} erfolgreich verarbeitet ({len(result)} Datenpunkte)")
                    else:
                        print(f"⚠️ {subject} ohne Ergebnisse")
                        completed_subjects.append(subject)  # Trotzdem als "verarbeitet" markieren
                        
                except Exception as e:
                    print(f"❌ Fehler bei {subject}: {e}")
                    # Subject nicht zu completed_subjects hinzufügen, damit es beim nächsten Durchlauf erneut versucht wird
            
            # Batch-Ergebnisse zu Gesamtergebnissen hinzufügen
            all_results.extend(batch_results)
            
            # Checkpoint nach jedem Batch speichern
            save_checkpoint(checkpoint_path, completed_subjects, all_results)
            
            # Zwischenergebnis als CSV speichern
            if all_results:
                try:
                    interim_df = pd.DataFrame(all_results)
                    interim_csv_path = final_csv_path.parent / f"interim_{final_csv_path.name}"
                    interim_df.to_csv(interim_csv_path, index=False)
                    print(f"💾 Zwischenergebnis gespeichert: {interim_csv_path}")
                except Exception as e:
                    print(f"⚠️ Fehler beim Speichern des Zwischenergebnisses: {e}")
            
            print(f"📊 Fortschritt: {len(completed_subjects)}/{len(valid_subjects)} Subjects verarbeitet")
    
    return all_results

def main():
    """Hauptfunktion mit Checkpoint-Unterstützung"""
    # Konfiguration validieren
    try:
        validate_config()
    except Exception as e:
        print(f"❌ Konfigurationsfehler: {e}")
        return
    
    # Run-Name generieren
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{RUN_PREFIX}_detection_{PEAK_DETECTION_METHOD}_padding_{SEIZURE_PADDING_CUTOFF[0]}-{SEIZURE_PADDING_CUTOFF[1]}-{SEIZURE_PADDING_EVAL[0]}-{SEIZURE_PADDING_EVAL[1]}_{timestamp}.csv"
    checkpoint_name = f"{RUN_PREFIX}_checkpoint_{PEAK_DETECTION_METHOD}_padding_{SEIZURE_PADDING_CUTOFF[0]}-{SEIZURE_PADDING_CUTOFF[1]}-{SEIZURE_PADDING_EVAL[0]}-{SEIZURE_PADDING_EVAL[1]}.json"
    
    print(f"🚀 Starte Jeppesen SeizeIT2 Analyse (mit Checkpoint-Unterstützung)")
    print(f"📊 Run: {run_name}")
    print(f"💾 Checkpoint: {checkpoint_name}")
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

    # Ergebnisordner erstellen
    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint_path = RESULTS_DIR / checkpoint_name
    final_csv_path = RESULTS_DIR / run_name
    
    print(f"💾 Checkpoint-Datei: {checkpoint_path}")
    print(f"📄 Finale CSV-Datei: {final_csv_path}")
    
    try:
        # Subjects verarbeiten (mit oder ohne Checkpoints)
        if ENABLE_CHECKPOINTS:
            all_subject_metrics = process_subjects_with_checkpoints(
                valid_subjects, checkpoint_path, final_csv_path
            )
        else:
            print("⚠️ Checkpoint-Funktionalität deaktiviert - Klassische Verarbeitung")
            # Klassische Parallelverarbeitung ohne Checkpoints
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

        # Finale Ergebnisse speichern
        metrics_df = pd.DataFrame(all_subject_metrics)
        metrics_df.to_csv(final_csv_path, index=False)
        
        print(f"\n✅ Finale Ergebnisse gespeichert: {final_csv_path}")
        print(f"📊 Verarbeitete Subjects: {len(set(metrics_df['subject']))}")
        print(f"📋 Gesamte Datenpunkte: {len(metrics_df)}")
        
        # Kurze Zusammenfassung
        if len(metrics_df) > 0:
            print("\n📈 Kurze Zusammenfassung:")
            print(f"   Mittlere Sensitivität: {metrics_df['sensitivity'].mean():.3f}")
            print(f"   Mittlere FAD: {metrics_df['FAD'].mean():.3f}")
            print(f"   Responder-Rate: {metrics_df['responder'].mean():.3f}")
        
        # Checkpoint-Datei nach erfolgreichem Abschluss löschen
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"🗑️ Checkpoint-Datei gelöscht: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Fehler beim Löschen der Checkpoint-Datei: {e}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Verarbeitung durch Benutzer unterbrochen")
        print(f"💾 Checkpoint gespeichert - Verarbeitung kann später fortgesetzt werden")
        print(f"🔄 Zum Fortsetzen einfach das Skript erneut ausführen")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        print(f"💾 Checkpoint möglicherweise gespeichert - prüfen Sie: {checkpoint_path}")
        raise

if __name__ == "__main__":
    main()