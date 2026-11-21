from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import yaml

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

manifest = pd.read_csv(cfg["paths"]["manifest"])
row = manifest.iloc[0]

audio, sr = librosa.load(row["audio_path"], sr=cfg["sample_rate"], mono=True)

n_fft = cfg["spectral_subtraction"]["n_fft"]
hop = cfg["spectral_subtraction"]["hop_length"]

S = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop,
    n_mels=80,
)
S_db = librosa.power_to_db(S, ref=np.max)

fig_dir = Path(cfg["paths"]["figures"])
fig_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))
librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=hop,
    x_axis="time",
    y_axis="mel",
)
plt.colorbar(format="%+2.0f dB")
plt.title("Annotated log-Mel spectrogram of a clean LibriSpeech sample")
plt.axvline(0.25, linestyle="--", label="0.25 s noise-estimation boundary")
plt.xlabel("Time (s)")
plt.ylabel("Mel frequency")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "annotated_spectrogram.png", dpi=200)
plt.close()

print(f"Saved {fig_dir / 'annotated_spectrogram.png'}")
