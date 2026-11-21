from pathlib import Path
import argparse
import random
import sys
import time

import pandas as pd
import torch
import whisper
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio_utils import load_mono, mix_at_snr, save_audio
from src.denoise import spectral_subtraction
from src.metrics import score

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--save-audio", action="store_true")
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

random.seed(cfg["experiment"]["seed"])
torch.manual_seed(cfg["experiment"]["seed"])

manifest = pd.read_csv(cfg["paths"]["manifest"])
if args.limit is not None:
    manifest = manifest.head(args.limit)

noise_root = Path(cfg["paths"]["musan_root"]) / cfg["experiment"]["noise_category"]
noise_files = sorted(noise_root.rglob("*.wav"))
if not noise_files:
    raise FileNotFoundError(
        f"No WAV files found in {noise_root}. "
        "Download/extract MUSAN and place it under data/musan/."
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = cfg["model"]["name"]
print(f"Loading Whisper model: {model_name} on {device}")
model = whisper.load_model(model_name, device=device)

out_dir = Path(cfg["paths"]["output_audio"])
results_dir = Path(cfg["paths"]["results"])
out_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

rows = []

for row_i, row in manifest.iterrows():
    clean = load_mono(row["audio_path"], cfg["sample_rate"])
    reference = row["reference"]

    # Clean baseline
    t0 = time.perf_counter()
    clean_pred = model.transcribe(
        clean,
        language=cfg["model"]["language"],
        task=cfg["model"]["task"],
        fp16=(device == "cuda"),
        verbose=False,
    )["text"].strip()
    clean_latency = time.perf_counter() - t0

    clean_score = score(reference, clean_pred)
    duration = len(clean) / cfg["sample_rate"]

    rows.append({
        "dataset_index": row["dataset_index"],
        "speaker_id": row["speaker_id"],
        "snr_db": "clean",
        "condition": "clean_baseline",
        "reference": reference,
        "prediction": clean_pred,
        "wer": clean_score["wer"],
        "cer": clean_score["cer"],
        "latency_s": clean_latency,
        "audio_duration_s": duration,
        "rtf": clean_latency / max(duration, 1e-8),
    })

    for snr in cfg["experiment"]["snrs_db"]:
        noise_path = random.choice(noise_files)
        noise = load_mono(noise_path, cfg["sample_rate"])
        noisy = mix_at_snr(clean, noise, snr)

        if args.save_audio:
            save_audio(
                out_dir / f"{int(row['dataset_index'])}_{snr}dB_noisy.wav",
                noisy,
                cfg["sample_rate"],
            )

        t0 = time.perf_counter()
        noisy_pred = model.transcribe(
            noisy,
            language=cfg["model"]["language"],
            task=cfg["model"]["task"],
            fp16=(device == "cuda"),
            verbose=False,
        )["text"].strip()
        noisy_latency = time.perf_counter() - t0

        noisy_score = score(reference, noisy_pred)

        denoised = spectral_subtraction(
            noisy,
            sr=cfg["sample_rate"],
            **cfg["spectral_subtraction"],
        )

        if args.save_audio:
            save_audio(
                out_dir / f"{int(row['dataset_index'])}_{snr}dB_denoised.wav",
                denoised,
                cfg["sample_rate"],
            )

        t0 = time.perf_counter()
        denoised_pred = model.transcribe(
            denoised,
            language=cfg["model"]["language"],
            task=cfg["model"]["task"],
            fp16=(device == "cuda"),
            verbose=False,
        )["text"].strip()
        denoised_latency = time.perf_counter() - t0

        denoised_score = score(reference, denoised_pred)

        rows.extend([
            {
                "dataset_index": row["dataset_index"],
                "speaker_id": row["speaker_id"],
                "snr_db": snr,
                "condition": "noisy_baseline",
                "reference": reference,
                "prediction": noisy_pred,
                "wer": noisy_score["wer"],
                "cer": noisy_score["cer"],
                "latency_s": noisy_latency,
                "audio_duration_s": duration,
                "rtf": noisy_latency / max(duration, 1e-8),
                "noise_file": str(noise_path),
            },
            {
                "dataset_index": row["dataset_index"],
                "speaker_id": row["speaker_id"],
                "snr_db": snr,
                "condition": "denoised",
                "reference": reference,
                "prediction": denoised_pred,
                "wer": denoised_score["wer"],
                "cer": denoised_score["cer"],
                "latency_s": denoised_latency,
                "audio_duration_s": duration,
                "rtf": denoised_latency / max(duration, 1e-8),
                "noise_file": str(noise_path),
            },
        ])

        print(
            f"[{row_i+1}/{len(manifest)}] "
            f"SNR={snr:>3} | noisy WER={noisy_score['wer']:.3f} | "
            f"denoised WER={denoised_score['wer']:.3f}"
        )

results = pd.DataFrame(rows)
results.to_csv(results_dir / "experiment_results.csv", index=False)

summary = (
    results.groupby(["snr_db", "condition"], dropna=False)
    .agg(
        samples=("wer", "count"),
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
        mean_latency_s=("latency_s", "mean"),
        mean_rtf=("rtf", "mean"),
    )
    .reset_index()
)
summary.to_csv(results_dir / "summary_by_snr.csv", index=False)

print(f"\nSaved: {results_dir / 'experiment_results.csv'}")
print(f"Saved: {results_dir / 'summary_by_snr.csv'}")
