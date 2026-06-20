# Chinese Sentiment Classification — IPFA Research

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlreadyJ/chinese-sentiment-ipfa/blob/main/notebooks/chinese_sentiment_ipfa.ipynb)

**Author:** AlreadyJ  
**Research direction:** Clinical NLP · Multimodal Affective Computing

---

## Overview

This repository trains a Chinese sentiment classifier as the **text-channel baseline** for the *Illocutionary Pragmatic Force Attenuation* (IPFA) research pipeline — the central construct of my PhD proposal on AI-assisted mental health support for Mandarin-speaking patients.

IPFA describes a systematic failure mode in affective AI systems: when a patient's expressive output is suppressed through **affective flattening** (a cardinal symptom of depression and schizophrenia) combined with **face-preserving Chinese pragmatics**, standard classifiers misread suppressed distress as neutral or positive sentiment. This is a clinically dangerous false negative.

This classifier exists to **demonstrate that failure concretely** on a standard text-only model, motivating the domain-adversarial adaptation layer in the full pipeline.

---

## Research context

```
Full IPFA pipeline (PhD scope)
──────────────────────────────────────────────────────────────────────
 Source domain          Adaptation           Clinical deployment
 C-drama corpora   →   DANN transfer    →   Mandarin mental health
 (MAFW, MER 2024)      (Ganin et al.)       support system
      ↓                                           ↓
 High-affect speech                      Suppressed distress
 exaggerates IPFA cues                   detected, not missed
──────────────────────────────────────────────────────────────────────
 THIS REPO: text-channel baseline — establishes what a vanilla
 fine-tuned BERT can and cannot detect before multimodal DANN
 adaptation is applied.
```

The IPFA problem is formalised in: Illocutionary Pragmatic Force Attenuation in multimodal Chinese mental health AI

---

## Model

| Component | Detail |
|---|---|
| Base model | `bert-base-chinese` (HuggingFace) |
| Dataset | ChnSentiCorp (Tan Songbo, ICA-CAS) — 9,600 Chinese reviews, binary sentiment |
| Task | Binary sentiment classification (negative / positive) |
| Training | 3 epochs, AdamW lr=2e-5, linear warmup (10%), batch=16, max_len=128 |
| Hardware | CPU-compatible (≈ 35 min on Colab free tier) |

---

## Results

*Run the notebook to reproduce. Expected performance:*

| Metric | Score |
|---|---|
| Test accuracy | ~95% |
| Macro F1 | ~0.95 |
| Negative F1 | ~0.94 |
| Positive F1 | ~0.96 |

These figures are consistent with published BERT fine-tuning results on ChnSentiCorp (e.g. Cui et al., 2021, *MacBERT*).

### IPFA demonstration

The critical finding is **not** the aggregate accuracy — it is the classifier's behaviour on affectively attenuated text. Running `src/predict.py` in demo mode on the following examples illustrates the baseline failure mode:

| Text | Clinical label | Expected model output |
|---|---|---|
| `感觉最近做什么都没意思，也不知道为什么。` | Distress (anhedonia) | Positive / Neutral ⚠ |
| `没什么，我还好，只是有点想家而已。` | Distress (face-preserving minimisation) | Positive ⚠ |
| `我最近有点累，可能是工作太忙了。` | Distress (externalised attribution) | Positive / Neutral ⚠ |

These misclassifications are not errors in the model — they are accurate reflections of what the training signal contains. The fix requires DANN adaptation from a domain where illocutionary force is exaggerated (C-drama), not more of the same labelled data.

---

## Quickstart

### Option A — Google Colab (recommended, no setup)

Click the **Open in Colab** badge above. All dependencies install in the first cell. Runtime: ~35 min on CPU.

### Option B — Local

```bash
git clone https://github.com/AlreadyJ/chinese-sentiment-ipfa
cd chinese-sentiment-ipfa
pip install -r requirements.txt

# Train
python src/train.py --epochs 3

# Run IPFA demo predictions
python src/predict.py

# Classify a single string
python src/predict.py --text "感觉最近做什么都没意思，也不知道为什么。"

# Classify a CSV
python src/predict.py --file my_data.csv --output results/predictions.csv
```

---

## Repository structure

```
chinese-sentiment-ipfa/
├── notebooks/
│   └── chinese_sentiment_ipfa.ipynb   # Full walkthrough with IPFA analysis
├── src/
│   ├── train.py                       # Training script
│   └── predict.py                     # Inference + IPFA demo
├── results/                           # Generated: metrics.json, confusion matrix
├── model/                             # Generated: saved fine-tuned model
├── requirements.txt
└── README.md
```

---

## Limitations and scope

This repository is explicitly scoped as a **text-only baseline**. Known limitations relative to the full IPFA challenge:

- **No audio or visual modalities** — the full IPFA pipeline integrates Qwen2-Audio + OpenFace 2.0 for acoustic and facial channel analysis
- **No clinical data** — ChnSentiCorp is a product/hotel review corpus; the clinical validation will use CMDC and PDCH (held out, never in training)
- **Binary labels only** — the full pipeline uses graded IPFA scores rather than binary sentiment
- **No domain adversarial adaptation** — this is the baseline the DANN layer is evaluated against

The purpose of this repository is to demonstrate the failure mode cleanly, not to solve it.

---

## References

- Ganin et al. (2016). Domain-adversarial training of neural networks. *JMLR*.
- Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL*.
- Cui et al. (2021). Pre-training with whole word masking for Chinese BERT. *IEEE/ACM TASLP*.
- Tan Songbo. ChnSentiCorp dataset. Institute of Computing Technology, Chinese Academy of Sciences.
- Zou et al. (2023). CMDC. *IEEE Transactions on Affective Computing (TAFFC)*.

---
