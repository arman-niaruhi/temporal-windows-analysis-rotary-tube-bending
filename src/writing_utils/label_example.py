import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 1, 1000)
signal = np.sin(3 * np.pi * 2 * t) + 0.8 * np.random.randn(len(t))

patch_size = 100
step_non_overlap = patch_size
step_overlap = 50

non_overlap_starts = np.arange(0, len(signal) - patch_size + 1, step_non_overlap)
overlap_starts = np.arange(0, len(signal) - patch_size + 1, step_overlap)

sig_min = np.min(signal)
sig_max = np.max(signal)
sig_range = sig_max - sig_min

y_arrow_non = sig_min - 0.05 * sig_range

plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal, color='grey')
for i, start in enumerate(non_overlap_starts[:3]):
    end = start + patch_size - 1
    plt.axvspan(t[start], t[end], color='blue', alpha=0.2)
    plt.annotate(
        '', xy=(t[end], y_arrow_non), xytext=(t[start], y_arrow_non),
        arrowprops=dict(arrowstyle='<->', color='blue', lw=2)
    )
    plt.text((t[start]+t[end])/2, y_arrow_non - 0.02*sig_range, f'Patch {i+1}',
             color='blue', ha='center', va='top')
plt.title("Non-overlapping: First 3 Patches")
plt.xlabel("Time (s)")
plt.ylabel("Signal")
plt.ylim(y_arrow_non - 0.1*sig_range, sig_max + 0.1*sig_range)

plt.subplot(2, 1, 2)
plt.plot(t, signal, color='grey')

for i, start in enumerate(overlap_starts[:3]):
    end = start + patch_size - 1
    plt.axvspan(t[start], t[end], color='red', alpha=0.2)
    y_arrow = sig_min + 0.3*sig_range - i*0.2*sig_range
    plt.annotate(
        '', xy=(t[end], y_arrow), xytext=(t[start], y_arrow),
        arrowprops=dict(arrowstyle='<->', color='red', lw=2)
    )
    plt.text((t[start]+t[end])/2, y_arrow - 0.01*sig_range, f'Patch {i+1}',
             color='red', ha='center', va='top')

plt.title("Overlapping: First 3 Patches")
plt.xlabel("Time (s)")
plt.ylabel("Sensor Data")
plt.ylim(sig_min - 0.5*sig_range, sig_max + 0.1*sig_range)

plt.tight_layout()
plt.savefig("patches_plot.pdf") 