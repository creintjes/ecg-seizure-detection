import numpy as np
import pickle
from pathlib import Path

def has_constant_region(sig: np.ndarray, min_len: int, tol: float = 1e-6) -> bool:
    if len(sig) < min_len:
        return False  # zu kurz zum Prüfen
    var = np.convolve((sig - sig.mean())**2, np.ones(min_len), 'valid') / min_len
    return (var < tol).any()

def check_directory_for_constants(folder: Path, min_len: int = 320, tol: float = 1e-6):
    files = list(folder.glob("*.pkl"))
    if not files:
        print(f"⚠️  Keine .pkl-Dateien gefunden in {folder}")
        return

    print(f"🔍 Untersuche {len(files)} Dateien in: {folder}")
    count_problematic = 0

    for file in sorted(files):
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
                sig = data["channels"][0]["data"]
                if has_constant_region(sig, min_len, tol):
                    print(f"⚠️  {file.name} enthält konstante Region")
                    count_problematic += 1
        except Exception as e:
            print(f"❌ Fehler beim Laden von {file.name}: {e}")

    print(f"\n✅ Fertig. {count_problematic} von {len(files)} Dateien enthalten konstante Regionen.")

# Beispiel-Aufruf:
check_directory_for_constants(
    folder=Path("/home/swolf/asim_shared/preprocessed_data/seizure_only/32hz_30min/downsample_32hz_context_30min"),
    min_len=320,
    tol=1e-6
)
