# Speech Recognition Robustness in Noisy Environments

## Project Overview

This case study evaluates the robustness of the **Whisper Base Automatic Speech Recognition (ASR) model** under different levels of synthetic background noise. The study investigates how increasing noise levels affect speech recognition performance and whether **spectral-gating-based noise reduction** can improve transcription accuracy.

The experiment compares the performance of Whisper Base on clean speech, noisy speech, and noise-reduced speech using **Word Error Rate (WER), Character Error Rate (CER), and inference time**.

## Dataset

- **Dataset:** LibriSpeech ASR – `test-clean`
- **Number of audio samples:** 5
- **Sampling rate:** 16 kHz
- **Noise type:** Synthetic Gaussian white noise
- **SNR levels:** 0 dB, 10 dB, and 20 dB

The LibriSpeech samples are used as clean reference speech. Synthetic Gaussian white noise is added to the speech recordings at controlled Signal-to-Noise Ratio (SNR) levels.

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

## ASR Model

The project uses the **Whisper Base** pretrained speech recognition model.

Whisper Base is evaluated under three experimental conditions:

1. Clean speech baseline
2. Noisy speech without noise reduction
3. Noisy speech after spectral-gating noise reduction

The same ASR model is used across the experimental conditions to ensure that the comparison focuses on the effect of noise and preprocessing.

## Methodology

The experimental pipeline consists of the following steps:

1. Load clean speech samples from the LibriSpeech `test-clean` dataset.
2. Transcribe the clean speech using Whisper Base to establish a baseline.
3. Generate synthetic Gaussian white noise.
4. Add the noise to the clean speech at **0 dB, 10 dB, and 20 dB SNR**.
5. Transcribe the noisy audio using Whisper Base.
6. Apply spectral-gating noise reduction using the `noisereduce` library.
7. Transcribe the denoised audio using Whisper Base.
8. Calculate WER and CER for noisy and denoised speech.
9. Measure inference time for the ASR system.
10. Compare the performance across different SNR levels.

## Experimental Pipeline

```text
                LibriSpeech
               Clean Speech
                    |
                    v
             Whisper Base
              Clean Baseline
                    |
                    v
              Ground Truth
                    |
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
      20 dB       10 dB        0 dB
        |           |           |
        +-----------+-----------+
                    |
                    v
          Synthetic Gaussian
              White Noise
                    |
                    v
              Noisy Speech
                    |
          +---------+---------+
          |                   |
          v                   v
    Whisper Base        Spectral Gating
     Baseline          Noise Reduction
          |                   |
          |                   v
          |             Denoised Speech
          |                   |
          |                   v
          |             Whisper Base
          |                   |
          +---------+---------+
                    |
                    v
              WER / CER
                    +
             Inference Time
```

## Experimental Results

The following values represent the **mean per-sample error rates** obtained from the experiment.

| **SNR (dB)** | **Noisy WER (%)** | **Denoised WER (%)** | **Noisy CER (%)** | **Denoised CER (%)** |
|---:|---:|---:|---:|---:|
| 0 | 23.72 | 32.52 | 11.53 | 20.11 |
| 10 | 5.70 | 9.13 | 2.69 | 3.53 |
| 20 | 7.25 | 7.09 | 2.84 | 2.43 |

## Key Findings

The experimental results show that background noise can significantly affect ASR performance.

- At **0 dB SNR**, the noisy speech produced a substantially higher WER of **23.72%**, showing significant degradation under strong noise.
- At **10 dB SNR**, the noisy speech achieved a WER of **5.70%**.
- At **20 dB SNR**, the noisy speech achieved a WER of **7.25%**.
- Spectral-gating noise reduction increased WER at **0 dB and 10 dB SNR**.
- At **20 dB SNR**, spectral gating produced a small improvement, reducing WER from **7.25% to 7.09%**.
- A similar pattern was observed in CER, where denoising improved the result at 20 dB but degraded performance at lower SNR levels.

These results suggest that noise reduction does not always improve ASR performance. In some conditions, the preprocessing stage may remove or distort speech components that are useful to the recognition model.

## Interpretation

The results indicate that the effectiveness of spectral-gating noise reduction depends on the severity of the noise.

At lower SNR levels, the noise-reduction algorithm may have difficulty distinguishing speech components from background noise. As a result, useful speech information can also be suppressed, leading to higher transcription errors.

At 20 dB SNR, the noise is less dominant, and spectral gating provides a small improvement in both WER and CER.

## Evaluation Metrics

### Word Error Rate (WER)

Word Error Rate measures the difference between the reference transcription and the predicted transcription at the word level.

```text
WER = (S + D + I) / N
```

where:

- **S** = Substitutions
- **D** = Deletions
- **I** = Insertions
- **N** = Number of words in the reference

### Character Error Rate (CER)

Character Error Rate measures transcription errors at the character level.

### Inference Time

Inference time measures the time required by Whisper Base to process an audio sample and generate its transcription.

## Limitations

This is a preliminary experimental study using only **five speech samples** and synthetic Gaussian white noise.

The limited sample size means that the results should not be generalized to all speakers, speech patterns, real-world acoustic environments, or ASR systems.

The experiment also uses synthetic white noise rather than naturally occurring environmental sounds such as traffic, café conversations, music, or vehicle noise.

## Repository Contents

The repository contains the implementation and supporting files for the experiment:

- Google Colab notebook containing the experimental implementation
- Clean speech baseline evaluation
- Noisy speech evaluation
- Spectral-gating noise-reduction experiment
- WER and CER calculations
- SNR performance analysis
- Experimental result files
- Visualization scripts
- Report documentation

## Conclusion

This case study demonstrates the effect of background noise on automatic speech recognition performance using Whisper Base.

The experiment shows that stronger noise conditions can increase transcription errors. However, spectral-gating noise reduction did not consistently improve recognition accuracy. It improved performance slightly at 20 dB SNR but reduced performance at 0 dB and 10 dB SNR.

Therefore, the results indicate that **noise reduction should be carefully evaluated before being used as a preprocessing stage for ASR**, since removing background noise can also affect speech information required by the recognition model.

## Author

**Name:** YOUR NAME  
**Roll Number:** YOUR ROLL NUMBER  
**Program:** B.Tech Artificial Intelligence and Machine Learning  
**University:** Jain University, Bengaluru  

## Course

**Speech Recognition and Conversational Systems**

**Instructor:** Prof. Shobha H R
