from pathlib import Path
import random
import numpy as np
import librosa
import soundfile as sf

def load_mono(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)

def rms(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))

def mix_at_snr(clean, noise, snr_db):
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    start = 0
    if len(noise) > len(clean):
        start = random.randint(0, len(noise) - len(clean))
    noise = noise[start:start + len(clean)].copy()

    clean_rms = rms(clean)
    noise_rms = rms(noise)

    if noise_rms < 1e-8:
        raise ValueError("Selected noise is effectively silent.")

    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / noise_rms)

    mixed = clean + noise
    peak = np.max(np.abs(mixed))
    if peak > 0.999:
        mixed = mixed / peak * 0.999

    return mixed.astype(np.float32)

def save_audio(path, audio, sr=16000):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)
