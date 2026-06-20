"""
predict.py — Run sentiment inference on new Chinese text
=========================================================
Loads the fine-tuned model from model/ and classifies input text.

Usage
-----
    # Single string
    python src/predict.py --text "我今天感觉很难过，什么都不想做"

    # From a CSV file (column named 'text')
    python src/predict.py --file data/my_samples.csv --output results/predictions.csv

IPFA relevance
--------------
This script is where the IPFA research question becomes concrete:
the same text that a human clinician might read as "evasive" or
"understated" is often assigned high-confidence positive sentiment
by a vanilla fine-tuned classifier — because the surface lexical
features are neutral or politely framed, even when the pragmatic
intent is distress. Running your own examples through this predictor
and observing misclassifications is a direct demonstration of the
problem the full IPFA pipeline addresses.
"""

import argparse
import json
import os
import sys

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "model"


def load_model():
    if not os.path.exists(MODEL_DIR):
        print(f"No model found at '{MODEL_DIR}/'. Run src/train.py first.")
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(texts, tokenizer, model, max_len=128):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    id2label = model.config.id2label
    return [
        {
            "text": t,
            "predicted_label": id2label[int(p)],
            "confidence": round(float(probs[i][p]), 4),
            "prob_negative": round(float(probs[i][0]), 4),
            "prob_positive": round(float(probs[i][1]), 4),
        }
        for i, (t, p) in enumerate(zip(texts, preds))
    ]


def main(args):
    tokenizer, model = load_model()

    if args.text:
        results = predict([args.text], tokenizer, model)
        print(json.dumps(results[0], ensure_ascii=False, indent=2))

    elif args.file:
        df = pd.read_csv(args.file)
        if "text" not in df.columns:
            print("CSV must have a column named 'text'.")
            sys.exit(1)
        results = predict(df["text"].tolist(), tokenizer, model)
        out_df = pd.DataFrame(results)
        out_path = args.output or "results/predictions.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(out_df)} predictions to {out_path}")
        print(out_df[["text", "predicted_label", "confidence"]].to_string(index=False))

    else:
        # Demo mode — illustrative examples relevant to IPFA
        demo_texts = [
            "今天天气不错，心情还好。",                          # surface positive
            "我最近有点累，可能是工作太忙了。",                   # attenuated distress
            "没什么，我还好，只是有点想家而已。",                  # face-preserving circumlocution
            "这个产品非常好用，我很满意！",                       # clear positive
            "感觉最近做什么都没意思，也不知道为什么。",            # clinical IPFA example
        ]
        print("Demo predictions (IPFA-relevant examples):\n")
        results = predict(demo_texts, tokenizer, model)
        for r in results:
            label_str = r["predicted_label"].upper().ljust(8)
            conf_str  = f"{r['confidence']:.2%}"
            print(f"  [{label_str} {conf_str}]  {r['text']}")
        print(
            "\nNote: examples 2, 3, and 5 above use face-preserving indirection"
            " and affective attenuation — the core IPFA challenge. Observe whether"
            " the model correctly detects suppressed distress or defaults to"
            " surface-neutral classification."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="Single Chinese string to classify")
    group.add_argument("--file", type=str, help="Path to CSV with a 'text' column")
    parser.add_argument("--output", type=str, help="Output CSV path (with --file)")
    main(parser.parse_args())
