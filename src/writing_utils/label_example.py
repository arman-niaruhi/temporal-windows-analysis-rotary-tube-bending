import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# -------------------------------------------------
# 1️⃣ Time axis: 0 to 40
# -------------------------------------------------
t = np.linspace(0, 40, 1000)

# -------------------------------------------------
# 2️⃣ Generate signal
#    Use (t/40) inside sine to keep original frequency
# -------------------------------------------------
raw_signal = np.sin(3 * np.pi * 2 * (t / 40)) + 0.8 * np.random.randn(len(t))

# Normalize to [0,1]
signal_norm = (raw_signal - raw_signal.min()) / (raw_signal.max() - raw_signal.min())

# Scale to [0,40]
signal = signal_norm * 40.0

# -------------------------------------------------
# 3️⃣ Window parameters
# -------------------------------------------------
patch_size = 100
step_non_overlap = patch_size
step_overlap = 50

non_overlap_starts = np.arange(0, len(signal) - patch_size + 1, step_non_overlap)
overlap_starts = np.arange(0, len(signal) - patch_size + 1, step_overlap)

# Signal statistics
sig_min = np.min(signal)
sig_max = np.max(signal)
sig_range = sig_max - sig_min
y_arrow_non = sig_min - 0.05 * sig_range

plt.figure(figsize=(14, 6))

# =================================================
# (1) Non-overlapping windows
# =================================================
plt.subplot(2, 1, 1)
plt.plot(t, signal, color='#7a7a7a')

for i, start in enumerate(non_overlap_starts[:3]):
    end = start + patch_size - 1
    plt.axvspan(t[start], t[end], color="#f80c14", alpha=0.2)
    plt.annotate(
        '',
        xy=(t[end], y_arrow_non),
        xytext=(t[start], y_arrow_non),
        arrowprops=dict(arrowstyle='<->', lw=2)
    )
    plt.text(
        (t[start] + t[end]) / 2,
        y_arrow_non - 0.02 * sig_range,
        f'Window {i+1}',
        ha='center',
        color="#f80c14",
        va='top'
    )

plt.xlabel("Time")
plt.ylabel("Sensor Data")
plt.ylim(y_arrow_non - 0.1 * sig_range, sig_max + 0.1 * sig_range)

# =================================================
# (2) Overlapping windows
# =================================================
plt.subplot(2, 1, 2)
plt.plot(t, signal, color='#7a7a7a')

for i, start in enumerate(overlap_starts[:3]):
    end = start + patch_size - 1
    plt.axvspan(t[start], t[end],color="#0421fd", alpha=0.2)
    y_arrow = sig_min + 0.3 * sig_range - i * 0.2 * sig_range
    plt.annotate(
        '',
        xy=(t[end], y_arrow),
        xytext=(t[start], y_arrow),
        color="#0421fd",
        arrowprops=dict(arrowstyle='<->', lw=2)
    )
    plt.text(
        (t[start] + t[end]) / 2,
        y_arrow - 0.01 * sig_range,
        f'Window {i+1}',
        ha='center',
        color="#0421fd",
        va='top'
    )

plt.xlabel("Time")
plt.ylabel("Sensor Data")
plt.ylim(sig_min - 0.5 * sig_range, sig_max + 0.1 * sig_range)

plt.tight_layout()
plt.savefig("window-plot.pdf")