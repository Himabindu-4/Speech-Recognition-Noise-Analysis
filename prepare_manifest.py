from pathlib import Path
import argparse
import random
import pandas as pd
import torchaudio
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

root = Path(cfg["paths"]["librispeech_root"])
limit = args.limit if args.limit is not None else cfg["experiment"]["limit"]

ds = torchaudio.datasets.LIBRISPEECH(
    root=str(root),
    url="test-clean",
    download=False,
)

indices = list(range(len(ds)))
random.Random(cfg["experiment"]["seed"]).shuffle(indices)
indices = indices[:limit]

rows = []
for idx in indices:
    path, sr, transcript, speaker_id, chapter_id, utterance_id = ds.get_metadata(idx)
    rows.append({
        "dataset_index": idx,
        "audio_path": str(Path(path).resolve()),
        "sample_rate": sr,
        "reference": transcript,
        "speaker_id": speaker_id,
        "chapter_id": chapter_id,
        "utterance_id": utterance_id,
    })

out = Path(cfg["paths"]["manifest"])
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)

print(f"Wrote {len(rows)} samples to {out}")
print(f"Unique speakers: {pd.DataFrame(rows)['speaker_id'].nunique()}")
