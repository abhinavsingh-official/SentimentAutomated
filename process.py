"""Sentiment scoring - replaces sentiment.py and sentiment_limited.py, which
were ~95% identical (same model load, same prediction loop, same PDF report,
differing only in whether the input got sampled to 5000 rows first). One
module with a sample_size parameter instead of two near-duplicate files.

The original predict() ran one row at a time through the tokenizer in a
Python loop - on a GPU that means 74,000+ individual forward passes instead
of a handful of batched ones, which defeats most of the point of having a
GPU at all. predict_batch() below tokenizes and runs a whole batch through
the model at once."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS = ["Negative", "Neutral", "Positive"]
DEFAULT_BATCH_SIZE = 32


@dataclass
class SentimentModel:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: torch.device

    @classmethod
    def load(cls) -> "SentimentModel":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        return cls(tokenizer=tokenizer, model=model, device=device)

    @torch.no_grad()
    def predict_batch(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[tuple[str, float]]:
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=128, padding=True
            ).to(self.device)
            logits = self.model(**inputs).logits.cpu().numpy()
            probs = softmax(logits, axis=1)
            for row in probs:
                idx = int(np.argmax(row))
                results.append((LABELS[idx], float(row[idx])))
        return results


def score_dataframe(
    df: pd.DataFrame, text_column: str, sample_size: int | None = None, model: SentimentModel | None = None
) -> pd.DataFrame:
    """Scores every row in text_column. sample_size takes a random sample
    first (random_state=42, for reproducibility) instead of processing
    everything - useful for a quick look at a large dataset."""
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    model = model or SentimentModel.load()
    texts = df[text_column].astype(str).tolist()
    results = model.predict_batch(texts)

    df = df.copy()
    # A source CSV with its own "sentiment"/"confidence" column (e.g. a
    # dataset that ships human-labeled ground truth) would otherwise get
    # silently overwritten by our output columns of the same name - the
    # original two scripts had exactly this bug. Preserve it instead.
    for existing_col in ("sentiment", "confidence"):
        if existing_col in df.columns:
            df = df.rename(columns={existing_col: f"original_{existing_col}"})
    df["sentiment"], df["confidence"] = zip(*results)
    df["score"] = df["sentiment"].map({"Negative": -1, "Neutral": 0, "Positive": 1})
    return df


if __name__ == "__main__":
    import os
    import sys

    path = input("Enter full path to CSV file: ").strip()
    if not os.path.isfile(path):
        print("File not found.")
        sys.exit(1)
    df = pd.read_csv(path)

    print("Columns:", list(df.columns))
    col = input("Enter column name with text: ").strip()
    if col not in df.columns:
        print("Column not found.")
        sys.exit(1)

    sample_input = input("Sample size (blank for full dataset): ").strip()
    sample_size = int(sample_input) if sample_input else None

    print(f"Scoring {sample_size or len(df)} rows...")
    scored = score_dataframe(df, col, sample_size=sample_size)
    scored.to_csv("results.csv", index=False)
    print(f"Done. {len(scored)} rows scored, saved to results.csv")
    print(scored["sentiment"].value_counts())
