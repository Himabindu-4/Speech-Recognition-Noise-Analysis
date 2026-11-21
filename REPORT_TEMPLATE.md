# Speech Recognition Robustness in Noisy Environments

## Title Page

**Project Title:** Speech Recognition Robustness in Noisy Environments  
**Case Study:** Option A – Robustness in Noisy Environments  
**Student:** YOUR NAME  
**Roll Number:** YOUR ROLL NUMBER  
**Course:** Speech Recognition and Conversational Systems  
**Instructor:** Prof. Shobha H R  
**Date:** 24 November 2026  

---

# 1. Executive Summary

Write approximately half a page.

Include:
- problem
- dataset
- Whisper Base model
- SNR conditions
- spectral subtraction
- WER/CER
- latency/RTF
- main conclusion

Use only values from `outputs/results/summary_by_snr.csv`.

# 2. Problem Statement & Real-World Context

Explain why background noise is important in automotive and smart-home speech interfaces.

State the experimental constraints:
- background noise
- fixed sampling rate
- pretrained model
- computational hardware
- controlled SNR conditions

# 3. System Architecture & Methodology

## 3.1 Pipeline

Insert a diagram:

Clean LibriSpeech
→ Noise selection from MUSAN
→ Controlled SNR mixing
→ Noisy waveform
→ Whisper Base baseline
→ WER/CER

and in parallel:

Noisy waveform
→ Spectral subtraction
→ Denoised waveform
→ Whisper Base
→ WER/CER

## 3.2 Audio parameters

Document:
- sampling rate
- FFT size
- hop length
- window length
- SNR levels
- noise-estimation duration
- spectral subtraction alpha
- spectral floor

## 3.3 Model justification

Explain why Whisper Base was selected. Cite the official Whisper repository and paper.

# 4. Experimental Setup & Quantitative Results

## 4.1 Dataset

Report:
- exact number of samples used
- total duration
- unique speakers
- LibriSpeech split
- MUSAN category
- train/test split information

Do not copy the full dataset size if you only evaluated a subset. Clearly distinguish dataset-level facts from your experimental subset.

## 4.2 Metrics

Define:

WER = (S + D + I) / N

where:
- S = substitutions
- D = deletions
- I = insertions
- N = reference word count

Also report CER and one system metric such as latency or RTF.

## 4.3 Results

Insert:
- `wer_vs_snr.png`
- `cer_vs_snr.png`
- `system_metrics.png`

Add numbered captions and discuss the trends using the actual generated values.

# 5. Qualitative Error Analysis

Use `outputs/results/error_cases.csv`.

For 3–5 examples, create a table:

| Sample | SNR | Ground Truth | Prediction | WER | Acoustic/Linguistic Diagnosis |
|---|---:|---|---|---:|---|
| ... | ... | ... | ... | ... | ... |

Listen to the relevant audio before assigning the cause. Possible causes include masking, consonant deletion, insertion, substitution, overlapping noise, or model uncertainty, but the exact cause must be supported by the audio.

# 6. Engineering Trade-offs & Conclusion

Discuss:
- accuracy vs. denoising cost
- latency
- RTF
- model memory
- suitability for edge devices
- limitations of simple spectral subtraction

Recommendations:
1. Test stronger neural denoisers or beamforming.
2. Evaluate additional noise categories and SNR values.
3. Compare Whisper Base with a smaller/faster ASR model.

# References

Use IEEE or APA style. Include:
- Whisper paper/repository
- LibriSpeech paper/dataset
- MUSAN paper/dataset
- Librosa
- PyTorch/Torchaudio
- JiWER

# Appendix

Include:
- configuration
- hardware/software versions
- additional tables
- GitHub repository URL
