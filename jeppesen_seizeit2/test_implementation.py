"""
Test-Skript für Jeppesen SeizeIT2 Implementierung
Testet die Pipeline mit einem einzelnen Subject
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from config import *
from seizeit2_utils import get_all_subjects, validate_subject_data, get_all_patient_records_as_dict
from feature_extraction import get_peak_dataframe
from evaluation_utils import apply_seizure_padding

def test_data_loading():
    """Testet das Laden der SeizeIT2-Daten"""
    print("🔍 Teste Daten-Loading...")
    
    try:
        subjects = get_all_subjects()
        print(f"   ✅ {len(subjects)} Subjects gefunden")
        
        if not subjects:
            print("   ❌ Keine Subjects gefunden!")
            return False
            
        # Teste ersten Subject
        test_subject = subjects[0]
        print(f"   🧪 Teste Subject: {test_subject}")
        
        is_valid = validate_subject_data(test_subject)
        print(f"   {'✅' if is_valid else '❌'} Subject-Validierung: {is_valid}")
        
        if is_valid:
            records = get_all_patient_records_as_dict(test_subject)
            print(f"   ✅ {len(records)} Recordings geladen")
            
            if records:
                first_record = list(records.values())[0]
                print(f"   ✅ Erstes Recording: {len(first_record)} Samples")
                print(f"   ✅ Columns: {list(first_record.columns)}")
                print(f"   ✅ Sampling Rate: {first_record.attrs.get('sampling_rate', 'unbekannt')}Hz")
                
                # Prüfe Seizure-Labels
                seizure_count = first_record['seizure'].sum() if 'seizure' in first_record.columns else 0
                print(f"   ✅ Seizure-Samples: {seizure_count}")
                
        return is_valid
        
    except Exception as e:
        print(f"   ❌ Fehler beim Daten-Loading: {e}")
        return False

def test_feature_extraction():
    """Testet die Feature-Extraktion"""
    print("\n🔧 Teste Feature-Extraktion...")
    
    try:
        subjects = get_all_subjects()
        test_subject = subjects[0] if subjects else None
        
        if not test_subject or not validate_subject_data(test_subject):
            print("   ❌ Kein gültiger Test-Subject verfügbar")
            return False
            
        records = get_all_patient_records_as_dict(test_subject)
        if not records:
            print("   ❌ Keine Recordings verfügbar")
            return False
            
        first_record = list(records.values())[0]
        print(f"   🧪 Verwende Recording mit {len(first_record)} Samples")
        
        # R-Peak Detection
        peak_dataframe = get_peak_dataframe(first_record, peak_detection_method='elgendi')
        print(f"   ✅ {len(peak_dataframe)} R-Peaks erkannt")
        print(f"   ✅ RR-Intervall Bereich: {peak_dataframe['rr_intervals'].min():.1f} - {peak_dataframe['rr_intervals'].max():.1f} ms")
        
        # Seizure Padding testen
        if 'seizure' in peak_dataframe.columns:
            padded_df = apply_seizure_padding(
                peak_dataframe, 
                seizure_padding=(10, 10), 
                seizure_col='seizure', 
                new_col_name='seizure_padded'
            )
            original_seizure_count = peak_dataframe['seizure'].sum()
            padded_seizure_count = padded_df['seizure_padded'].sum()
            print(f"   ✅ Seizure Padding: {original_seizure_count} → {padded_seizure_count} Samples")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fehler bei Feature-Extraktion: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_pipeline():
    """Testet eine minimale Pipeline"""
    print("\n⚙️ Teste Minimale Pipeline...")
    
    try:
        from feature_extraction import calculate_csi
        
        # Einfacher Test mit synthetischen RR-Intervallen
        test_rr = np.random.normal(800, 50, 200)  # 200 RR-Intervalle um 800ms
        
        csi, modcsi = calculate_csi(test_rr, window_size=50)
        print(f"   ✅ CSI berechnet: {len(csi)} Werte")
        print(f"   ✅ CSI Bereich: {np.nanmin(csi):.3f} - {np.nanmax(csi):.3f}")
        print(f"   ✅ ModCSI Bereich: {np.nanmin(modcsi):.3f} - {np.nanmax(modcsi):.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fehler bei Minimal-Pipeline: {e}")
        return False

def main():
    """Hauptfunktion für Tests"""
    print("🧪 Jeppesen SeizeIT2 Implementation Test")
    print("=" * 50)
    
    # Konfiguration prüfen
    try:
        validate_config()
        print("✅ Konfiguration validiert")
    except Exception as e:
        print(f"❌ Konfigurationsfehler: {e}")
        print("💡 Bitte SEIZEIT2_DATA_PATH in config.py anpassen!")
        return
    
    # Tests ausführen
    tests = [
        ("Daten-Loading", test_data_loading),
        ("Feature-Extraktion", test_feature_extraction), 
        ("Minimale Pipeline", test_minimal_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' fehlgeschlagen: {e}")
            results.append((test_name, False))
    
    # Zusammenfassung
    print("\n📊 Test-Zusammenfassung:")
    print("=" * 50)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 Ergebnis: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("🚀 Implementierung ist bereit!")
        print("💡 Nächster Schritt: python jeppesen_seizeit2.py")
    else:
        print("🔧 Bitte Fehler beheben vor dem Hauptlauf")

if __name__ == "__main__":
    main()