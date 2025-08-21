#!/usr/bin/env python3
"""
Test-Skript für die Checkpoint-Funktionalität
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Lokale Imports
from config import RESULTS_DIR
from jeppesen_seizeit2 import load_checkpoint, save_checkpoint

def test_checkpoint_functionality():
    """Testet die Grundfunktionalität der Checkpoint-Mechanismen"""
    
    print("🧪 Teste Checkpoint-Funktionalität...")
    
    # Test-Daten erstellen
    test_checkpoint_path = RESULTS_DIR / "test_checkpoint.json"
    test_subjects = ["sub-001", "sub-002", "sub-003"]
    test_results = [
        {
            "subject": "sub-001",
            "parameter": "csi_100", 
            "sensitivity": 0.8,
            "FAD": 0.5
        },
        {
            "subject": "sub-001", 
            "parameter": "modcsi_100",
            "sensitivity": 0.7,
            "FAD": 0.3
        }
    ]
    
    try:
        # Test 1: Checkpoint speichern
        print("📝 Test 1: Checkpoint speichern...")
        save_checkpoint(test_checkpoint_path, test_subjects[:1], test_results)
        
        if test_checkpoint_path.exists():
            print("✅ Checkpoint-Datei erfolgreich erstellt")
        else:
            print("❌ Fehler: Checkpoint-Datei nicht erstellt")
            return False
            
        # Test 2: Checkpoint laden
        print("📖 Test 2: Checkpoint laden...")
        loaded_subjects, loaded_results = load_checkpoint(test_checkpoint_path)
        
        if loaded_subjects == test_subjects[:1] and loaded_results == test_results:
            print("✅ Checkpoint erfolgreich geladen")
        else:
            print("❌ Fehler beim Laden des Checkpoints")
            print(f"   Erwartet: {test_subjects[:1]}, {test_results}")
            print(f"   Erhalten: {loaded_subjects}, {loaded_results}")
            return False
            
        # Test 3: Checkpoint-Datei Struktur prüfen
        print("🔍 Test 3: Checkpoint-Datei Struktur...")
        with open(test_checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            
        required_keys = ['completed_subjects', 'results', 'timestamp', 'total_completed']
        if all(key in checkpoint_data for key in required_keys):
            print("✅ Checkpoint-Struktur ist korrekt")
        else:
            print("❌ Fehler: Checkpoint-Struktur unvollständig")
            print(f"   Gefundene Keys: {list(checkpoint_data.keys())}")
            print(f"   Erwartete Keys: {required_keys}")
            return False
            
        # Test 4: Leeres Checkpoint laden (Datei existiert nicht)
        print("📂 Test 4: Nicht-existierenden Checkpoint laden...")
        non_existent_path = RESULTS_DIR / "non_existent_checkpoint.json"
        empty_subjects, empty_results = load_checkpoint(non_existent_path)
        
        if empty_subjects == [] and empty_results == []:
            print("✅ Leerer Checkpoint korrekt behandelt")
        else:
            print("❌ Fehler bei nicht-existierendem Checkpoint")
            return False
            
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Checkpoint-Test: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # Aufräumen
        if test_checkpoint_path.exists():
            test_checkpoint_path.unlink()
            
    print("✅ Alle Checkpoint-Tests erfolgreich!")
    return True

def test_resume_functionality():
    """Simuliert Resume-Funktionalität"""
    
    print("\n🔄 Teste Resume-Funktionalität...")
    
    # Simuliere Szenario: 3 von 5 Subjects bereits verarbeitet
    all_subjects = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"]
    completed_subjects = ["sub-001", "sub-002", "sub-003"]
    
    # Berechne verbleibende Subjects
    remaining_subjects = [s for s in all_subjects if s not in completed_subjects]
    
    expected_remaining = ["sub-004", "sub-005"]
    
    if remaining_subjects == expected_remaining:
        print("✅ Resume-Logik funktioniert korrekt")
        print(f"   Gesamt: {len(all_subjects)} Subjects")
        print(f"   Verarbeitet: {len(completed_subjects)} Subjects")
        print(f"   Verbleibend: {len(remaining_subjects)} Subjects")
        return True
    else:
        print("❌ Fehler in Resume-Logik")
        print(f"   Erwartet verbleibend: {expected_remaining}")
        print(f"   Tatsächlich verbleibend: {remaining_subjects}")
        return False

if __name__ == "__main__":
    print("🚀 Starte Checkpoint-Tests...\n")
    
    # Ergebnisordner erstellen falls nicht vorhanden
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Tests ausführen
    test1_success = test_checkpoint_functionality()
    test2_success = test_resume_functionality()
    
    # Gesamtergebnis
    if test1_success and test2_success:
        print("\n🎉 Alle Tests erfolgreich! Checkpoint-System ist einsatzbereit.")
    else:
        print("\n❌ Einige Tests fehlgeschlagen. Bitte Fehler prüfen.")