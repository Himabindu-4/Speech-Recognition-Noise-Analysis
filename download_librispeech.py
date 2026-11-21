from pathlib import Path
import argparse
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data/librispeech")
args = parser.parse_args()

root = Path(args.root)
root.mkdir(parents=True, exist_ok=True)

print("Downloading LibriSpeech test-clean if it is not already present...")
ds = torchaudio.datasets.LIBRISPEECH(
    root=str(root),
    url="test-clean",
    download=True,
)
print(f"LibriSpeech test-clean is available. Samples found: {len(ds)}")
