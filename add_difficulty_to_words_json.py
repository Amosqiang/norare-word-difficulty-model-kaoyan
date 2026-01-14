#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add difficulty fields to words.json and write words_with_difficulty.json.

difficulty_freq_0_100:
  From log1p(freq), clipped to p1/p99, then scaled to 0..100 (higher = harder).
difficulty_norare_pred_0_100:
  From difficulty_model_any_word.joblib (NoRaRe-based model).
difficulty_0_100:
  Weighted blend of the two (default 0.7/0.3).
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover - sklearn may be absent during static checks
    InconsistentVersionWarning = None

def main(
    words_json: Path,
    model_file: Path,
    out_json: Path,
    freq_weight: float,
    norare_weight: float
):
    data = json.loads(words_json.read_text(encoding="utf-8"))

    words = []
    freqs = []
    for row in data:
        word = row.get("word")
        words.append("" if word is None else str(word))
        freq = row.get("frequency")
        freqs.append(0.0 if freq is None else float(freq))

    freqs = np.asarray(freqs, dtype="float64")
    logf = np.log1p(freqs)
    p1 = float(np.percentile(logf, 1))
    p99 = float(np.percentile(logf, 99))
    if p99 - p1 < 1e-12:
        diff_freq = np.full_like(logf, 50.0, dtype="float64")
    else:
        clipped = np.clip(logf, p1, p99)
        diff_freq = 100.0 * (1.0 - (clipped - p1) / (p99 - p1))
    diff_freq = np.clip(diff_freq, 0.0, 100.0)

    if InconsistentVersionWarning is not None:
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    bundle = joblib.load(model_file)
    model = bundle["model"]
    X = pd.DataFrame({"word": words, "word_len": [len(w) for w in words]})
    diff_norare = np.clip(model.predict(X).astype("float64"), 0.0, 100.0)

    diff_fused = np.clip(
        freq_weight * diff_freq + norare_weight * diff_norare,
        0.0,
        100.0
    )

    for i, row in enumerate(data):
        row["difficulty_freq_0_100"] = float(diff_freq[i])
        row["difficulty_norare_pred_0_100"] = float(diff_norare[i])
        row["difficulty_0_100"] = float(diff_fused[i])

    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")
    print(f"weights: freq={freq_weight} norare={norare_weight} p1={p1:.6f} p99={p99:.6f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--words_json", default="words.json")
    ap.add_argument("--model", default="difficulty_model_any_word.joblib")
    ap.add_argument("--out_json", default="words_with_difficulty.json")
    ap.add_argument("--freq_weight", type=float, default=0.7)
    ap.add_argument("--norare_weight", type=float, default=0.3)
    args = ap.parse_args()
    main(Path(args.words_json), Path(args.model), Path(args.out_json), args.freq_weight, args.norare_weight)
