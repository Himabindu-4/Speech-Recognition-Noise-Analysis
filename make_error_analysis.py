from pathlib import Path
import argparse
import pandas as pd
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=5)
args = parser.parse_args()

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

path = Path(cfg["paths"]["results"]) / "experiment_results.csv"
df = pd.read_csv(path)

# Choose the most severe noisy-baseline errors from 0 dB.
subset = df[
    (df["condition"] == "noisy_baseline") &
    (df["snr_db"].astype(str) == "0")
].copy()

subset = subset.sort_values("wer", ascending=False).head(args.n)

out = subset[
    ["dataset_index", "speaker_id", "snr_db", "reference", "prediction", "wer", "cer"]
].copy()

out["diagnosis"] = "Fill after listening to the corresponding clean/noisy audio. Do not infer the acoustic cause from text alone."

output_path = Path(cfg["paths"]["results"]) / "error_cases.csv"
out.to_csv(output_path, index=False)

print(out.to_string(index=False))
print(f"\nSaved {output_path}")
