import numpy as np
from clustering import analyze_detection_mask

# --- Inputs you provide ---
fs = 256.0                     # samplerate (Hz)
duration_s = 60                # 1 minute recording
n = int(fs * duration_s)

# Boolean detection mask (True where your model flags anomaly)
mask = np.zeros(n, dtype=bool)
# Two example anomaly segments: [12.0–13.2]s and [37.5–38.0]s
seg1 = (int(12.0*fs), int(13.2*fs))
seg2 = (int(37.5*fs), int(38.0*fs))
mask[seg1[0]:seg1[1]] = True
mask[seg2[0]:seg2[1]] = True

# Optional per-sample scores aligned to mask (same length)
scores = np.zeros(n, dtype=float)
scores[seg1[0]:seg1[1]] = 0.7
scores[seg2[0]:seg2[1]] = 0.9

# Optional ground-truth intervals in seconds (start, end)
gt_intervals = [(12.1, 13.0)]   # overlaps seg1; seg2 will count as FP if GT is only this

# --- Run analysis ---
results = analyze_detection_mask(
    mask=mask,
    fs=8,
    file_id="rec_001",
    scores=scores,                # optional
    subject_id="subj_A",          # optional
    gt_intervals=gt_intervals,    # optional
    output_folder="rec_001_smart_clustered"
)

print("Best strategy:", results["best_strategy"])
print("Clusters in best strategy:", results["best_results"]["clusters"])
print("Total representatives:", len(results["best_results"]["representatives"]))