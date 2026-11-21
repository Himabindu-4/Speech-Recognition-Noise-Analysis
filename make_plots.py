from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yaml

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

results_path = Path(cfg["paths"]["results"]) / "experiment_results.csv"
fig_dir = Path(cfg["paths"]["figures"])
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(results_path)
noisy = df[df["condition"] == "noisy_baseline"].copy()
den = df[df["condition"] == "denoised"].copy()

def plot_metric(metric, ylabel, filename):
    n = noisy.groupby("snr_db")[metric].mean()
    d = den.groupby("snr_db")[metric].mean()

    x = sorted(n.index.astype(int))
    plt.figure(figsize=(7, 5))
    plt.plot(x, [n.loc[i] for i in x], marker="o", label="Noisy baseline")
    plt.plot(x, [d.loc[i] for i in x], marker="o", label="Spectral subtraction")
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs. SNR")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / filename, dpi=200)
    plt.close()

plot_metric("wer", "Word Error Rate (WER)", "wer_vs_snr.png")
plot_metric("cer", "Character Error Rate (CER)", "cer_vs_snr.png")

lat = (
    df[df["condition"].isin(["noisy_baseline", "denoised"])]
    .groupby("condition")["latency_s"]
    .mean()
)

plt.figure(figsize=(7, 5))
plt.bar(lat.index, lat.values)
plt.ylabel("Mean inference latency (s)")
plt.title("Inference latency by processing condition")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(fig_dir / "system_metrics.png", dpi=200)
plt.close()

print(f"Figures written to {fig_dir}")
