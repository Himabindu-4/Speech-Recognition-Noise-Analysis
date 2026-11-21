import numpy as np
from scipy.signal import stft, istft

def spectral_subtraction(
    audio,
    sr=16000,
    n_fft=512,
    hop_length=128,
    win_length=512,
    alpha=2.0,
    oversubtraction_floor=0.02,
    noise_estimate_seconds=0.25,
):
    """
    Basic magnitude spectral subtraction.

    The first noise_estimate_seconds are treated as a noise-only estimate.
    This is a controlled classroom baseline, not a production-grade denoiser.
    """
    _, _, z = stft(
        audio,
        fs=sr,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )

    mag = np.abs(z)
    phase = np.angle(z)

    noise_samples = max(1, int(noise_estimate_seconds * sr))
    noise_mag = mag[:, :max(1, int(noise_samples / hop_length))]
    noise_power = np.mean(noise_mag ** 2, axis=1, keepdims=True)

    signal_power = mag ** 2
    clean_power = signal_power - alpha * noise_power
    clean_power = np.maximum(clean_power, oversubtraction_floor * signal_power)

    enhanced_mag = np.sqrt(clean_power)
    enhanced = enhanced_mag * np.exp(1j * phase)

    _, out = istft(
        enhanced,
        fs=sr,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        input_onesided=True,
    )

    out = out[:len(audio)]
    peak = np.max(np.abs(out)) + 1e-12
    if peak > 0.999:
        out = out / peak * 0.999

    return out.astype(np.float32)
