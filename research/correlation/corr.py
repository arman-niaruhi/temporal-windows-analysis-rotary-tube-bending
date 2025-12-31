import numpy as np
import matplotlib.pyplot as plt
from dtw import accelerated_dtw

np.random.seed(42)
t = np.linspace(0, 40*np.pi, 4000)
signal = np.sin(t) + 0.3*np.sin(3*t) + 0.2*np.random.randn(len(t))

start, end = 1000, 1200
ref_segment = signal[start:end]
window_size = len(ref_segment)

distances = []
indices = range(0, len(signal) - window_size, 50)  
for i in indices:
    window = signal[i:i + window_size]
    dist, _, _, _ = accelerated_dtw(ref_segment, window, dist='euclidean')
    distances.append(dist)

distances = np.array(distances)

best_idx = indices[np.argmin(distances)]

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(signal, label="Signal")
plt.axvspan(start, end, color='orange', alpha=0.3, label="Marked Region")
plt.axvspan(best_idx, best_idx + window_size, color='green', alpha=0.3, label="Most Similar Region")
plt.legend()
plt.title("Signal and Most Similar Region (by DTW)")

plt.subplot(2, 1, 2)
plt.plot(indices, distances)
plt.title("DTW Distance Between Marked Region and Other Windows")
plt.xlabel("Start Index of Window")
plt.ylabel("DTW Distance (lower = more similar)")
plt.tight_layout()
plt.show()
