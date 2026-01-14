#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict difficulty_0_100 for ANY word using difficulty_model_any_word.joblib.

Example:
  python3 predict_any_word.py --word "abandon" --model difficulty_model_any_word.joblib
  python3 predict_any_word.py --word "abandon" --words_json words.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def load_frequency_difficulty(words_json: Path):
    data = json.loads(words_json.read_text(encoding="utf-8"))
    freq_map = {}
    for row in data:
        word = row.get("word")
        freq = row.get("frequency")
        if word is None or freq is None:
            continue
        word = str(word)
        freq_map[word] = float(freq)
        lower = word.lower()
        if lower not in freq_map:
            freq_map[lower] = float(freq)
    if not freq_map:
        return {}
    freqs = np.array(list(freq_map.values()), dtype="float64")
    logfreq = np.log10(freqs + 1.0)
    min_log = float(np.nanmin(logfreq))
    max_log = float(np.nanmax(logfreq))
    if max_log - min_log < 1e-12:
        difficulty = np.full_like(logfreq, 50.0, dtype="float64")
    else:
        difficulty = (max_log - logfreq) / (max_log - min_log) * 100.0
    return {word: float(difficulty[i]) for i, word in enumerate(freq_map.keys())}

def predict(
    word: str,
    model_file: str = "difficulty_model_any_word.joblib",
    freq_difficulty: dict | None = None,
    return_source: bool = False
):
    if freq_difficulty is not None and word in freq_difficulty:
        score, source = float(freq_difficulty[word]), "frequency"
    else:
        bundle = joblib.load(model_file)
        model = bundle["model"]
        X = pd.DataFrame([{"word": word, "word_len": len(word)}])
        score = float(model.predict(X)[0])
        score, source = float(np.clip(score, 0.0, 100.0)), "model"
    if return_source:
        return score, source
    return score

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True)
    ap.add_argument("--model", default="difficulty_model_any_word.joblib")
    ap.add_argument("--words_json", default="")
    args = ap.parse_args()
    freq_difficulty = load_frequency_difficulty(Path(args.words_json)) if args.words_json else None
    score, source = predict(args.word, args.model, freq_difficulty, return_source=True)
    if args.words_json:
        print(f"{args.word}\tdifficulty_0_100={score:.2f}\tsource={source}")
    else:
        print(f"{args.word}\tdifficulty_0_100={score:.2f}")
