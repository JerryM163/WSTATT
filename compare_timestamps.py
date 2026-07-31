import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

history_18 = np.load("Wstatt-18-0-2-4-5_history.npz")
history_24 = np.load("Wstatt-24-0-2-4-5_history.npz")

train_18 = history_18["train_loss"]
val_18 = history_18["val_loss"]

train_24 = history_24["train_loss"]
val_24 = history_24["val_loss"]

epochs_18 = np.arange(1, len(train_18) + 1)
epochs_24 = np.arange(1, len(train_24) + 1)

# Training and Validation Loss for 18 vs. 24 timestamps
plt.figure(figsize=(10, 6))

sns.lineplot(
    x=epochs_18,
    y=train_18,
    marker="o",
    label="18 timestamps - Training"
)

sns.lineplot(
    x=epochs_18,
    y=val_18,
    marker="o",
    linestyle="--",
    label="18 timestamps - Validation"
)

sns.lineplot(
    x=epochs_24,
    y=train_24,
    marker="s",
    label="24 timestamps - Training"
)

sns.lineplot(
    x=epochs_24,
    y=val_24,
    marker="s",
    linestyle="--",
    label="24 timestamps - Validation"
)

plt.title("Sun Bands: 18 vs. 24 Timestamps")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    "sun_bands_18_vs_24_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Generalization gap

gap_18 = val_18 - train_18
gap_24 = val_24 - train_24

plt.figure(figsize=(10,6))

sns.lineplot(
    x=epochs_18,
    y=gap_18,
    marker="o",
    label="18 timestamps"
)

sns.lineplot(
    x=epochs_24,
    y=gap_24,
    marker="s",
    label="24 timestamps"
)

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)

plt.title("Sun Bands: Generalization Gap")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss - Training Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    "sun_bands_18_vs_24_generalization_gap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
print("Comparison plots saved.")