#!/usr/bin/env python3
"""
Fast MADRID m-Parameter Analysis (Seizure Detection)

Dieses Skript untersucht, wie sich verschiedene Subsequenzlängen (m) auf die
Sensitivität der MADRID-Seizure-Erkennung auswirken – jetzt deutlich effizienter,
weil jede Datei nur einmal durch MADRID läuft.

python madrid_m_analysis_v2.py --data-dir /home/swolf/asim_shared/preprocessed_data/seizure_only/32hz_30min/downsample_32hz_context_30min --percentage 40 --m-range 320 3200 --step 320 --workers 4

"""

from __future__ import annotations
import os, sys, time, warnings, random, argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------------------------------------------- #
# 1) Robustes Import-Handling für die MADRID-Klasse                           #
# --------------------------------------------------------------------------- #
def _import_madrid():
    """
    Versucht nacheinander:
      1. from madrid import MADRID       (gleicher Ordner)
      2. from models.madrid import MADRID
    """
    try:
        from madrid import MADRID  # noqa: F401
        return MADRID
    except Exception:
        try:
            from models.madrid import MADRID  # noqa: F401
            return MADRID
        except Exception as e:
            raise ImportError(
                "MADRID konnte nicht importiert werden. "
                "Lege madrid.py entweder in das gleiche Verzeichnis oder nach models/."
            ) from e

MADRID = _import_madrid()
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# 2) Helferklassen                                                            #
# --------------------------------------------------------------------------- #
class MultiLengthRunner:
    """
    Ruft MADRID pro Datei **genau einmal** auf (min_m .. max_m) und
    cached das Ergebnis für beliebige m-Werte.
    """

    def __init__(self,
                 min_m: int,
                 max_m: int,
                 step: int,
                 train_ratio: float = 1 / 3,
                 use_gpu: bool = True):
        self.min_m, self.max_m, self.step = min_m, max_m, step
        self.train_ratio = train_ratio
        self.use_gpu = use_gpu

    # --------------------------------------------------------------------- #
    def run(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gibt (multi_length_table, bsf, bsf_loc) auf CPU-Numpy zurück.
        """
        train_test_split = int(len(ts) * self.train_ratio)
        madrid = MADRID(use_gpu=self.use_gpu, enable_output=False)
        ml_table, bsf, bsf_loc = madrid.fit(
            T=ts,
            min_length=self.min_m,
            max_length=self.max_m,
            step_size=self.step,
            train_test_split=train_test_split,
        )
        return ml_table, bsf, bsf_loc

    # --------------------------------------------------------------------- #
    def sensitivity_for_m(
        self,
        m: int,
        bsf: np.ndarray,
        bsf_loc: np.ndarray,
        true_regions: List[Tuple[int, int]],
        fs: int,
        overlap_thresh: float,
    ) -> float:
        """
        Berechnet, ob der top-Discord bei m einen Anfall trifft (0/1).
        true_regions: Liste[(start_sample, end_sample)]
        """
        idx = 0  # Index im bsf-Vektor
        if len(bsf) > 1:
            # bsf ist in gleicher Reihenfolge wie Range(min_m,max_m,step)
            idx = (m - self.min_m) // self.step
        loc = int(bsf_loc[idx]) if not np.isnan(bsf_loc[idx]) else None
        if loc is None:
            return 0.0

        # Zeitbereich des gefundenen Discords
        start = loc
        end = loc + m

        for s, e in true_regions:
            inter = max(0, min(end, e) - max(start, s))
            if inter / min(m, (e - s)) >= overlap_thresh:
                return 1.0
        return 0.0


# --------------------------------------------------------------------------- #
# 3) Loader & Seizure-Region-Extractor                                        #
# --------------------------------------------------------------------------- #
def load_pickle(path: Path) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
    """
    Erwartet das von dir beschriebene .pkl-Format und gibt zurück:
      signal (np.ndarray), [(start,end), ...] in Samples, sampling_rate
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    ch = data["channels"][0]
    signal = ch["data"].astype(np.float64)
    labels = ch["labels"]
    fs = data.get("sampling_rate", 125)

    ictal_idx = np.where(labels == "ictal")[0]
    if len(ictal_idx) == 0:
        return signal, [], fs

    # Blöcke gruppieren
    regions = []
    start = ictal_idx[0]
    for i in range(1, len(ictal_idx)):
        if ictal_idx[i] != ictal_idx[i - 1] + 1:
            regions.append((start, ictal_idx[i - 1] + 1))
            start = ictal_idx[i]
    regions.append((start, ictal_idx[-1] + 1))
    return signal, regions, fs


# --------------------------------------------------------------------------- #
# 4) Kern-Analyse-Routine                                                     #
# --------------------------------------------------------------------------- #
def analyse_directory(
    data_dir: Path,
    m_values: List[int],
    step: int,
    overlap_thresh: float,
    percentage: float,
    workers: int,
    use_gpu: bool,
) -> Dict[int, float]:
    """
    Durchläuft alle Dateien, ruft MADRID pro Datei genau einmal auf
    und berechnet pro m die Trefferquote (Sensitivity).
    """
    # -------- Dateien auswählen --------
    files = sorted(list(data_dir.glob("*.pkl")))
    n_select = max(1, int(len(files) * percentage / 100))
    random.shuffle(files)
    files = files[:n_select]
    print(f"📂 Verwende {len(files)} Dateien ({percentage} % von {len(list(data_dir.glob('*.pkl')))}).")

    # -------- Runner vorbereiten --------
    runner = MultiLengthRunner(
        min_m=min(m_values),
        max_m=max(m_values),
        step=step,
        use_gpu=use_gpu,
    )

    # -------- Verarbeitung (option. parallel) --------
    if workers > 1:
        from joblib import Parallel, delayed
        def _process(path):
            sig, regions, fs = load_pickle(path)
            try:
                ml_table, bsf, bsf_loc = runner.run(sig)
                return sig, regions, fs, bsf, bsf_loc
            except ValueError as e:
                if "constant" in str(e):
                    print(f"⚠️  Datei übersprungen wegen konstanter Region: {path.name}")
                    return None  # später filtern
                else:
                    raise

        results = Parallel(n_jobs=workers)(
            delayed(_process)(p) for p in files
        )
    else:
        results = []
        for p in files:
            try:
                sig, regions, fs = load_pickle(p)
                ml_table, bsf, bsf_loc = runner.run(sig)
                results.append((sig, regions, fs, bsf, bsf_loc))
            except ValueError as e:
                if "constant" in str(e):
                    print(f"⚠️  Datei übersprungen wegen konstanter Region: {p.name}")
                else:
                    raise

    results = [r for r in results if r is not None]


    # -------- Sensitivität für jede m --------
    sens_dict = defaultdict(list)

    for sig, regions, fs, bsf, bsf_loc in results:
        for m in m_values:
            hit = runner.sensitivity_for_m(
                m, bsf, bsf_loc, regions, fs, overlap_thresh
            )
            sens_dict[m].append(hit)

    # Mittelwert pro m
    return {m: float(np.mean(vals)) for m, vals in sens_dict.items()}


# --------------------------------------------------------------------------- #
# 5) CLI-Interface                                                            #
# --------------------------------------------------------------------------- #
def cli():
    parser = argparse.ArgumentParser(
        description="Schnelle MADRID-m-Analyse mit nur einem Fit pro Datei"
    )
    parser.add_argument("--data-dir", required=True, help="Verzeichnis mit *.pkl")
    parser.add_argument("--m-range", nargs=2, type=int, metavar=("MIN", "MAX"), default=[250, 3000],
                        help="m-Bereich in Samples (Default: 250-3000)")
    parser.add_argument("--step", type=int, default=250, help="Schrittweite zwischen m-Werten")
    parser.add_argument("--percentage", type=float, default=100.0,
                        help="Prozentsatz der Dateien, die genutzt werden (Default 100)")
    parser.add_argument("--overlap", type=float, default=0.1,
                        help="Min. Überlappung (Anteil) zum Zählen als Treffer")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel-Jobs (joblib), Default 1 = kein Parallelismus")
    parser.add_argument("--no-gpu", action="store_true", help="GPU deaktivieren")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"❌ Ordner {data_dir} existiert nicht.")

    min_m, max_m = args.m_range
    m_values = list(range(min_m, max_m + 1, args.step))

    print(f"Teste {len(m_values)} m-Werte von {min_m} bis {max_m} in {args.step}er-Schritten.")
    start = time.time()
    sensitivities = analyse_directory(
        data_dir=data_dir,
        m_values=m_values,
        step=args.step,
        overlap_thresh=args.overlap,
        percentage=args.percentage,
        workers=args.workers,
        use_gpu=not args.no_gpu,
    )
    elapsed = time.time() - start

    # ---- Ergebnisübersicht ----
    best_m = max(sensitivities, key=sensitivities.get)
    print("\n╔═══════════════════════════════════════════════╗")
    print("║        m-Analyse – Ergebnisübersicht          ║")
    print("╚═══════════════════════════════════════════════╝")
    for m in sorted(sensitivities):
        print(f"m={m:5d}: Sensitivität = {sensitivities[m]:.3f}")
    print(f"\n🏆 Bestes m = {best_m} (Sensitivität {sensitivities[best_m]:.3f})")
        # ---- Ergebnis speichern ----
    # Beispiel: "32hz_30min" aus dem letzten Ordner extrahieren
    suffix = data_dir.name  # z. B. "downsample_32hz_context_30min"
    hz_match = next((part for part in suffix.split("_") if "hz" in part), "unknownhz")
    min_match = next((part for part in suffix.split("_") if "min" in part), "unknownmin")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"ergebnisse_{hz_match}_{min_match}_{timestamp}.txt"

    with open(out_path, "w") as f:
        f.write("MADRID Sensitivitätsanalyse – Ergebnisse\n")
        f.write(f"Verwendete Daten: {data_dir}\n")
        f.write(f"Samplingrate: {hz_match}, Fenster: {min_match}\n")
        f.write(f"Prozent der Daten: {args.percentage}%, GPU: {'Ja' if not args.no_gpu else 'Nein'}\n")
        f.write(f"m-Werte: {min_m} bis {max_m} in {args.step}er-Schritten\n")
        f.write(f"Gesamtzeit: {elapsed:.1f} Sekunden\n\n")
        f.write("m\tSensitivität\n")
        for m in sorted(sensitivities):
            f.write(f"{m}\t{sensitivities[m]:.3f}\n")
        f.write(f"\nBestes m: {best_m} (Sensitivität {sensitivities[best_m]:.3f})\n")

    print(f"💾 Ergebnisse gespeichert in: {out_path}")

    print(f"⏱  Gesamtzeit: {elapsed:.1f}s  (GPU={'Ja' if not args.no_gpu else 'Nein'})")


if __name__ == "__main__":
    cli()
