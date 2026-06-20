"""
Chinese Sentiment Classification — IPFA Research Foundation
============================================================
Fine-tunes bert-base-chinese on the weibo_senti_100k dataset for
binary sentiment (positive / negative).

weibo_senti_100k contains real Chinese social media posts 
 — a register close to clinical mental health discourse,
making it a strong text-channel baseline for
the IPFA research pipeline.

Usage
-----
    python src/train.py [--epochs N] [--batch_size N] [--max_len N]

Outputs
-------
    results/metrics.json     — precision, recall, F1, accuracy per class
    results/confusion.png    — confusion matrix heatmap
    model/                   — saved fine-tuned model + tokenizer
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ── Dataset wrapper ───────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_and_prepare(tokenizer, max_len):
    """Load weibo_senti_100k (all subsets), tokenise, return train/test splits."""
    print("Loading weibo_senti_100k dataset …")
    splits = {"train": [], "validation": [], "test": []}
    for subset in SUBSETS:
        ds_sub = load_dataset("dirtycomputer/weibo_senti_100k", subset)
        for split in splits:
            splits[split].append(ds_sub[split])
    ds = {split: concatenate_datasets(parts) for split, parts in splits.items()}
    print(f"  Train: {len(ds['train'])}  |  Val: {len(ds['validation'])}  |  Test: {len(ds['test'])}")

    def encode_split(split_name):
        data = ds[split_name]
        texts  = [t if isinstance(t, str) else '' for t in data['sentence']]
        labels = [1 if ex['label'] == '0POS' else 0 for ex in data]
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors=None,
        )
        return SentimentDataset(enc, labels)

    train_ds = encode_split("train")
    test_ds  = encode_split("test")
    return train_ds, test_ds


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
    return all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model + tokenizer
    model_name = "bert-base-chinese"
    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 2-class for ChnSentiCorp binary version (neg / pos)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    model.to(device)

    # Data
    train_ds, test_ds = load_and_prepare(tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    print(f"\nFine-tuning for {args.epochs} epoch(s) …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  Epoch {epoch} | step {step}/{len(train_loader)} "
                      f"| loss {total_loss/step:.4f}")
        elapsed = time.time() - t0
        print(f"Epoch {epoch} done in {elapsed:.1f}s | "
              f"avg loss {total_loss/len(train_loader):.4f}")

    # Evaluation
    print("\nEvaluating on test set …")
    preds, labels = evaluate(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds,
        target_names=["negative", "positive"],
        output_dict=True,
    )
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=["negative", "positive"]))

    # Save metrics
    os.makedirs("results", exist_ok=True)
    metrics = {"accuracy": round(acc, 4), "classification_report": report}
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved results/metrics.json")

    # Confusion matrix (text fallback — no matplotlib required)
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(
        cm,
        index=["true_negative", "true_positive"],
        columns=["pred_negative", "pred_positive"],
    )
    cm_df.to_csv("results/confusion_matrix.csv")
    print("Saved results/confusion_matrix.csv")
    print("\nConfusion matrix:\n", cm_df.to_string())

    # Save model
    os.makedirs("model", exist_ok=True)
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
    print("\nModel saved to model/")

    # Try to save a matplotlib plot too (optional dependency)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["negative", "positive"],
                    yticklabels=["negative", "positive"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — bert-base-chinese (weibo_senti_100k)")
        fig.tight_layout()
        fig.savefig("results/confusion_matrix.png", dpi=150)
        print("Saved results/confusion_matrix.png")
    except ImportError:
        print("(matplotlib/seaborn not installed — skipping PNG plot)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len",    type=int, default=128)
    main(parser.parse_args())
