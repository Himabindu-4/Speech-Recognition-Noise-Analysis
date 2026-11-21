import re
from jiwer import wer, cer

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def score(reference, hypothesis):
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    return {
        "wer": wer(ref, hyp),
        "cer": cer(ref, hyp),
    }
