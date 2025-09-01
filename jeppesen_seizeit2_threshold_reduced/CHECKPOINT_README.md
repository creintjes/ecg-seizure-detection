# Checkpoint-Funktionalität für jeppesen_seizeit2.py

## Übersicht

Das Skript `jeppesen_seizeit2.py` wurde um eine robuste Checkpoint-Funktionalität erweitert, um Datenverlust bei Abstürzen zu vermeiden und die Verarbeitung nach Unterbrechungen fortzusetzen.

## Hauptfunktionen

### 1. Automatisches Speichern von Zwischenergebnissen
- **Batch-Verarbeitung**: Subjects werden in kleinen Batches verarbeitet (standardmäßig 3 pro Batch)
- **Checkpoint nach jedem Batch**: Nach jedem verarbeiteten Batch werden die Ergebnisse gespeichert
- **Interim-CSV**: Zusätzlich wird nach jedem Batch eine `interim_*.csv` Datei erstellt

### 2. Resume-Funktionalität
- **Automatische Erkennung**: Beim Start wird geprüft, ob bereits verarbeitete Subjects existieren
- **Fortsetzen**: Nur noch nicht verarbeitete Subjects werden bearbeitet
- **Fortschrittsmeldung**: Klare Anzeige des Verarbeitungsfortschritts

### 3. Robuste Fehlerbehandlung
- **Subject-Timeout**: Konfigurierbarer Timeout pro Subject (Standard: 60 Minuten)
- **Detaillierte Logging**: Ausführliche Fehlermeldungen mit Traceback
- **Graceful Degradation**: Fehler bei einzelnen Subjects stoppt nicht die gesamte Verarbeitung

## Konfiguration

In `config.py` können folgende Parameter angepasst werden:

```python
# === CHECKPOINT KONFIGURATION ===
ENABLE_CHECKPOINTS = True              # Aktiviert/Deaktiviert Checkpoints
CHECKPOINT_BATCH_SIZE = 3              # Subjects pro Batch
SUBJECT_TIMEOUT_MINUTES = 60           # Timeout pro Subject
```

## Verwendung

### Normale Ausführung
```bash
python3 jeppesen_seizeit2.py
```

### Nach einem Absturz
Einfach das Skript erneut ausführen - es erkennt automatisch den vorherigen Fortschritt:

```bash
python3 jeppesen_seizeit2.py
```

**Ausgabe bei Resume:**
```
📊 Fortschritt: 15/50 Subjects bereits verarbeitet
🔄 Verbleibend: 35 Subjects
```

### Ohne Checkpoints (klassische Verarbeitung)
Setzen Sie in `config.py`:
```python
ENABLE_CHECKPOINTS = False
```

## Dateien

### Checkpoint-Dateien
- **Format**: `seizeit2_jeppesen_checkpoint_[parameter].json`
- **Inhalt**: Bereits verarbeitete Subjects und deren Ergebnisse
- **Speicherort**: `results/` Ordner
- **Automatisches Löschen**: Nach erfolgreichem Abschluss

### Ausgabe-Dateien
- **Finale CSV**: `seizeit2_jeppesen_detection_[parameter]_[timestamp].csv`
- **Interim CSV**: `interim_seizeit2_jeppesen_detection_[parameter]_[timestamp].csv`

## Vorteile

✅ **Schutz vor Datenverlust**: Zwischenergebnisse gehen bei Abstürzen nicht verloren

✅ **Flexibilität**: Verarbeitung kann jederzeit unterbrochen und fortgesetzt werden

✅ **Überwachung**: Detaillierte Fortschrittsanzeige und Logging

✅ **Performance**: Optimierte Batch-Verarbeitung

✅ **Robustheit**: Einzelne Subject-Fehler stoppen nicht die gesamte Verarbeitung

## Fehlerbehebung

### Problem: Checkpoint-Datei beschädigt
**Lösung**: Löschen Sie die Checkpoint-Datei im `results/` Ordner und starten neu

### Problem: Skript hängt bei einem Subject
**Lösung**: Nach dem konfigurierten Timeout wird automatisch zum nächsten Subject gewechselt

### Problem: Nicht genügend Speicher
**Lösung**: Reduzieren Sie `CHECKPOINT_BATCH_SIZE` und `MAX_WORKERS` in der Konfiguration

### Problem: Zu häufige Checkpoints
**Lösung**: Erhöhen Sie `CHECKPOINT_BATCH_SIZE` für größere Batches

## Monitoring

Das Skript gibt detaillierte Informationen aus:
- Verarbeitungszeit pro Subject
- Anzahl extrahierter Parameter
- Batch-Fortschritt
- Speicherort der Zwischenergebnisse

## Technische Details

- **Checkpoint-Format**: JSON mit Metadaten (Timestamp, Anzahl, etc.)
- **Parallelverarbeitung**: Prozess-Pool mit konfigurierbarer Worker-Anzahl
- **Memory Management**: Explizite Garbage Collection nach jedem Subject
- **Exception Handling**: Vollständige Traceback-Erfassung für Debugging