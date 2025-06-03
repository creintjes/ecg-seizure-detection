from pathlib import Path
import shutil

# Pfad zum Dataset
base_path = Path("ds005873-download")

# Alle Teilnehmerordner wie sub-001, sub-002, ...
for subject in base_path.glob("sub-*"):
    ses_path = subject / "ses-01"

    if not ses_path.exists():
        continue  # Überspringen, wenn ses-01 nicht existiert

    # Alle Unterordner in ses-01 durchgehen
    for modality in ses_path.iterdir():
        if modality.is_dir():
            if modality.name.lower() not in ["ecg", "eeg"]:
                # Lösche alle anderen Modalitäten komplett
                print(f"Lösche Ordner: {modality}")
                shutil.rmtree(modality)
            elif modality.name.lower() == "eeg":
                # Im EEG-Ordner nur Event-TSV-Dateien behalten
                eeg_path = modality
                for file in eeg_path.iterdir():
                    if file.is_file() and not file.name.endswith("_events.tsv"):
                        print(f"Lösche Nicht-Event-Datei: {file}")
                        file.unlink()
