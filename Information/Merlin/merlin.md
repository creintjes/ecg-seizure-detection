# MERLIN: Anomalie-Detektion in Zeitreihen

Der **MERLIN-Algorithmus** ist ein **parameterfreier Anomalie-Detektor für Zeitreihen**, der sogenannte **Discords** findet – also Teilsequenzen, die sich **stark von allen anderen unterscheiden**.  
Hier ist eine verständliche Erklärung, wie die Implementierung funktioniert:

---

## 🧠 Grundidee von MERLIN

MERLIN durchsucht eine Zeitreihe mit einem **Sliding-Window-Ansatz** und sucht nach Subsequenzen, die **keinen ähnlichen Nachbarn haben**.  
Solche Subsequenzen gelten als **Anomalien (Discords)**.

---

## 🔍 Algorithmus-Schritte (aus `_predict()`)

1. **Validierung der Eingabeparameter**
    - Prüft, ob `min_length ≥ 4` und `max_length ≤` Hälfte der Zeitreihe
    - Gibt Warnung bei konstanten Bereichen aus

2. **Generiere alle Fensterlängen `L` zwischen `min_length` und `max_length`**

3. **Für jede Fensterlänge `L`:**
    - Setze Distanzschwellwert `r = 2√L`
    - Suche Subsequenz, die keinen Nachbarn mit Abstand `< r` hat → das ist ein Discord
    - Falls kein Discord gefunden wird, verringere `r` iterativ:
        - `r = r * 0.99` oder `r = μ - 2σ`

4. **Für gefundene Discords:**
    - Speichere deren Startposition
    - Markiere als Anomalie

---

## 🔧 Zentrale Funktion `_drag()`

Diese Funktion:

- z-normalisiert jede Subsequenz
- prüft für jede neue Subsequenz, ob sie ein Discord-Kandidat ist (d. h. kein naher Nachbar existiert)
- speichert für alle Kandidaten den **größten minimalen Abstand**
- wählt am Ende den mit dem **höchsten Abstand** → **am meisten anomal**

---

## 📈 Beispielhafte Anwendung

```python
from aeon.anomaly_detection.distance_based import MERLIN
import numpy as np

X = np.array([1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4])
detector = MERLIN(min_length=4, max_length=5)
anomalies = detector.fit_predict(X)
