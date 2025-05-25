from pathlib import Path
import re
from collections import defaultdict

# Basisverzeichnis, z. B. dein Projektpfad
base_path = Path("C:/Users/Reintjes/Documents/aD/UoC/Semester2/ASIM/Projekt/ecg-seizure-detection/ds005873-download")

# Regex zum Extrahieren von run-Nummern
run_pattern = re.compile(r"run-(\d+)")

# Ergebnisse als Dictionary: {sub-xxx: Anzahl Runs}
run_counts = defaultdict(int)

# Alle sub-xxx Ordner durchsuchen
for subject in base_path.glob("sub-*"):
    ecg_path = subject / "ses-01" / "ecg"
    if not ecg_path.exists():
        continue

    # Alle passenden .edf-Dateien durchsuchen
    for file in ecg_path.glob("*.edf"):
        if match := run_pattern.search(file.name):
            run_counts[subject.name] += 1

# Ausgabe
for subject, count in sorted(run_counts.items()):
    print(f"{subject}: {count} run(s)")
