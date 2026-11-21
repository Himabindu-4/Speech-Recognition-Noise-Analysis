# Speech Recognition Robustness in Noisy Environments

## Project Overview

This case study evaluates the robustness of the **Whisper Base Automatic Speech Recognition (ASR) model** under different levels of synthetic Gaussian noise. The study also investigates whether **spectral-gating-based noise reduction** can improve speech recognition accuracy in noisy environments.

The experiment compares the performance of Whisper Base on clean speech, noisy speech, and noise-reduced speech using Word Error Rate (WER), Character Error Rate (CER), and inference time.

## Dataset

- **Dataset:** LibriSpeech ASR – `test-clean`
- **Number of audio samples:** 5
- **Sampling rate:** 16 kHz
- **SNR levels:** 0 dB, 10 dB, and 20 dB
- **Noise type:** Synthetic Gaussian white noise
- **Language:** English

## Technologies Used

- Python
- Google Colab
- OpenAI Whisper Base
- PyTorch
- Librosa
- noisereduce
- JiWER
- NumPy
- Pandas
- Matplotlib

## Methodology

The experiment follows these steps:

1. Load clean speech samples from the LibriSpeech `test-clean` dataset.
2. Transcribe the clean speech using Whisper Base.
3. Generate synthetic Gaussian white noise.
4. Mix the noise with the clean speech at 0 dB, 10 dB, and 20 dB SNR.
5. Transcribe the noisy speech using Whisper Base.
6. Apply spectral-gating noise reduction using `noisereduce`.
7. Transcribe the denoised speech using Whisper Base.
8. Calculate WER and CER for each condition.
9. Measure inference time and compare the performance of the noisy and denoised speech.

## Experimental Pipeline

```text
LibriSpeech Clean Speech
          │
          ▼
   Whisper Base
   Clean Baseline
          │
          ▼
     Ground Truth
          │
          │
          ▼
  Add Gaussian Noise
          │
     ┌────┼────┐
     ▼    ▼    ▼
   20 dB 10 dB 0 dB
     │    │    │
     └────┼────┘
          ▼
    Whisper Base
          │
          ▼
    Noisy Prediction
          │
          ▼
     WER / CER
          
          AND

     Noisy Audio
          │
          ▼
 Spectral-Gating
 Noise Reduction
          │
          ▼
    Denoised Audio
          │
          ▼
    Whisper Base
          │
          ▼
  Denoised Prediction
          │
          ▼
     WER / CER
```

## Experimental Results

The following values represent the **mean per-sample error rates** obtained from the experiment.

| **SNR (dB)** | **Noisy WER (%)** | **Denoised WER (%)** | **Noisy CER (%)** | **Denoised CER (%)** |
|---:|---:|---:|---:|---:|
| 0 | 23.72 | 32.52 | 11.53 | 20.11 |
| 10 | 5.70 | 9.13 | 2.69 | 3.53 |
| 20 | 7.25 | 7.09 | 2.84 | 2.43 |

## Key Findings

- Speech recognition performance degraded substantially at **0 dB SNR**, where the noise level was comparable to the speech signal.
- Spectral-gating noise reduction resulted in **higher WER and CER at 0 dB and 10 dB SNR** in this experiment.
- At **20 dB SNR**, denoising produced a small improvement in both WER and CER.
- The results indicate that noise reduction does not always improve ASR performance.
- Spectral gating can potentially suppress useful speech components along with unwanted background noise.
- The effectiveness of preprocessing depends on the severity of the noise and the characteristics of the speech signal.

## Limitations

This is a preliminary experiment conducted using only **five speech samples** and synthetic Gaussian white noise.

Therefore, the results should not be generalized to all speakers, real-world acoustic environments, noise types, or ASR systems.

The experiment also evaluates only the Whisper Base model and one noise-reduction technique.

## Repository Contents

```text
Speech-Recognition-Noise-Analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Speech_Recognition_Noise_Analysis.ipynb
│
├── results/
│   ├── clean_audio_baseline_results.csv
│   ├── noisy_audio_experiment_results.csv
│   └── snr_performance_summary.csv
│
├── figures/
│   ├── wer_vs_snr.png
│   ├── cer_vs_snr.png
│   └── spectrogram.png
│
└── report/
    └── REPORT_TEMPLATE.md
```

## Reproducibility

The complete experimental implementation is provided in the Google Colab notebook.

The notebook contains the procedures for:

- Loading the LibriSpeech samples
- Generating Gaussian noise
- Creating different SNR conditions
- Whisper Base transcription
- Spectral-gating noise reduction
- WER and CER calculation
- Inference-time measurement
- Result generation

## Author

**Rachepalli Hima Bindu**  
B.Tech – Artificial Intelligence and Machine Learning  
Jain University, Bengaluru

## Course

**Speech Recognition and Conversational Systems**

**Instructor:** Prof. Shobha H R
