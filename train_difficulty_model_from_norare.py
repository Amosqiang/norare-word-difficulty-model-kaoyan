#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train difficulty_model_any_word.joblib from NoRaRe CLDF 1.1 zip.

Input:
  - Norare CLDF 1.1.zip
Outputs (in current directory by default):
  - difficulty_model_any_word.joblib
  - norare_training_database_en.csv
  - model_manifest.json

Meaning of difficulty_0_100:
  A 0..100 latent difficulty score. Labels are created by combining multiple English lexical norms via PCA (1D).
  Optionally, words.json frequency can be blended into the label (e.g., 35% frequency, 65% PCA).
  Then a Ridge regressor learns to predict that score from word form (char n-grams) + word length.

This is designed so you can:
  (1) Inspect the "database" (norare_training_database_en.csv) to see where labels came from.
  (2) Re-train to regenerate the model package anytime.
"""
from __future__ import annotations
import argparse, zipfile, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

PREFERRED_VAR_IDS = {
    "aoa_mean": ["Kuperman-2012-AoA-ENGLISH_AOA_MEAN", "Cortese-2008-AoA-ENGLISH_AOA_MEAN"],
    "prevalence": ["Brysbaert-2019-Prevalence-ENGLISH_PREVALENCE"],
    "known_pct": ["Brysbaert-2019-Prevalence-ENGLISH_KNOWN_PERCENTAGE"],
    "freq_log": ["VanHeuven-2014-Frequency-ENGLISH_FREQUENCY_LOG", "Brysbaert-2009-Frequency-ENGLISH_FREQUENCY_LOG"],
    "cd": ["VanHeuven-2014-Frequency-ENGLISH_CD", "Brysbaert-2009-Frequency-ENGLISH_CD"],
    "concreteness": ["Brysbaert-2014-Concreteness-ENGLISH_CONCRETENESS_MEAN"],
}
METRIC_COLS = ["aoa_mean","prevalence","known_pct","freq_log","cd","concreteness"]

def choose_metric(dfwide: pd.DataFrame, var_candidates: list[str]) -> tuple[pd.Series, pd.Series]:
    chosen = pd.Series([np.nan]*len(dfwide), index=dfwide.index, dtype="float64")
    source = pd.Series([None]*len(dfwide), index=dfwide.index, dtype="object")
    for vid in var_candidates:
        if vid not in dfwide.columns:
            continue
        vals = dfwide[vid]
        mask = chosen.isna() & vals.notna()
        chosen.loc[mask] = vals.loc[mask]
        source.loc[mask] = vid
    return chosen, source

def dataset_id_from_varid(varid: str) -> str:
    return str(varid).split("_", 1)[0]

def load_words_frequency(words_json: Path) -> dict[str, float]:
    data = json.loads(words_json.read_text(encoding="utf-8"))
    freq_map: dict[str, float] = {}
    for row in data:
        word = row.get("word")
        freq = row.get("frequency")
        if word is None or freq is None:
            continue
        word = str(word).strip().lower()
        if word:
            freq_map[word] = float(freq)
        variant = row.get("variant")
        if variant:
            variant = str(variant).strip().lower()
            if variant:
                freq_map[variant] = float(freq)
    return freq_map

def frequency_bounds_from_words(freq_map: dict[str, float], clip_low: float, clip_high: float) -> tuple[float, float]:
    freqs = np.asarray(list(freq_map.values()), dtype="float64")
    logf = np.log1p(freqs)
    p1 = float(np.percentile(logf, clip_low))
    p99 = float(np.percentile(logf, clip_high))
    return p1, p99

def frequency_to_difficulty(freqs: np.ndarray, p1: float, p99: float) -> tuple[np.ndarray, np.ndarray]:
    logf = np.log1p(freqs.astype("float64"))
    if p99 - p1 < 1e-12:
        diff = np.full_like(logf, 50.0, dtype="float64")
    else:
        clipped = np.clip(logf, p1, p99)
        diff = 100.0 * (1.0 - (clipped - p1) / (p99 - p1))
    return np.clip(diff, 0.0, 100.0), logf

def main(
    norare_zip: Path,
    out_dir: Path,
    work_dir: Path,
    words_json: Path | None,
    freq_weight: float,
    freq_clip_low: float,
    freq_clip_high: float
):
    if not (0.0 <= freq_weight <= 1.0):
        raise ValueError("freq_weight must be between 0 and 1.")
    if not (0.0 <= freq_clip_low < freq_clip_high <= 100.0):
        raise ValueError("freq_clip_low/freq_clip_high must satisfy 0 <= low < high <= 100.")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(norare_zip, "r") as z:
        z.extractall(work_dir)

    # Find CLDF folder
    candidates = list(work_dir.rglob("cldf/datasets.csv"))
    if not candidates:
        raise FileNotFoundError("Could not find cldf/datasets.csv inside the zip.")
    cldf_dir = candidates[0].parent

    glosses = pd.read_csv(cldf_dir/"glosses.csv")
    datasets = pd.read_csv(cldf_dir/"datasets.csv")
    norare = pd.read_csv(cldf_dir/"norare.csv.zip", compression="zip")

    gl_en = glosses[glosses["Language_ID"]=="eng"].copy()
    gl_en["form_norm"] = gl_en["Form"].astype(str).str.lower().str.strip()

    all_var_ids = sorted({vid for vids in PREFERRED_VAR_IDS.values() for vid in vids})
    sub = norare[norare["Variable_ID"].isin(all_var_ids)].copy()
    sub["Value_num"] = pd.to_numeric(sub["Value"], errors="coerce")

    sub = sub.merge(gl_en[["ID","Form","form_norm"]], left_on="Unit_ID", right_on="ID", how="left").drop(columns=["ID"], errors="ignore")
    sub = sub[sub["form_norm"].notna()].copy()

    agg = sub.groupby(["form_norm","Form","Variable_ID"], as_index=False)["Value_num"].mean()
    wide = (agg.pivot_table(index=["form_norm","Form"], columns="Variable_ID", values="Value_num", aggfunc="mean").reset_index())

    metrics = pd.DataFrame({"form_norm": wide["form_norm"], "word": wide["Form"].astype(str)})
    for name, vids in PREFERRED_VAR_IDS.items():
        metrics[name], metrics[f"{name}_source_varid"] = choose_metric(wide, vids)

    mask = metrics[METRIC_COLS].notna().sum(axis=1) >= 3
    train_df = metrics.loc[mask].copy()

    # PCA label
    pca_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=1, random_state=0)),
    ])
    pc1 = pca_pipe.fit_transform(train_df[METRIC_COLS])[:,0]

    aoa_vals = train_df["aoa_mean"].values
    aoa_fill = np.nan_to_num(aoa_vals, nan=np.nanmedian(aoa_vals))
    corr = float(np.corrcoef(pc1, aoa_fill)[0,1])
    pca_sign = 1
    if corr < 0:
        pc1 = -pc1
        pca_sign = -1

    pc_min, pc_max = float(pc1.min()), float(pc1.max())
    train_df["difficulty_label_0_100_pca"] = (pc1 - pc_min) / (pc_max - pc_min + 1e-9) * 100.0

    freq_map: dict[str, float] = {}
    freq_p1 = freq_p99 = None
    if freq_weight > 0:
        if words_json is None or not words_json.exists():
            raise FileNotFoundError("words.json not found for frequency blending.")
        freq_map = load_words_frequency(words_json)
        if not freq_map:
            raise ValueError("words.json has no usable frequency entries.")
        freq_p1, freq_p99 = frequency_bounds_from_words(freq_map, freq_clip_low, freq_clip_high)
        train_df["frequency"] = train_df["form_norm"].map(freq_map)
        freq_mask = train_df["frequency"].notna()
        freq_vals = train_df.loc[freq_mask, "frequency"].astype(float).values
        diff_freq, logf = frequency_to_difficulty(freq_vals, freq_p1, freq_p99)
        train_df["freq_log1p"] = np.nan
        train_df.loc[freq_mask, "freq_log1p"] = logf
        train_df["difficulty_freq_0_100"] = np.nan
        train_df.loc[freq_mask, "difficulty_freq_0_100"] = diff_freq
        train_df["freq_weight_used"] = 0.0
        train_df.loc[freq_mask, "freq_weight_used"] = float(freq_weight)
        train_df["difficulty_label_0_100"] = train_df["difficulty_label_0_100_pca"]
        train_df.loc[freq_mask, "difficulty_label_0_100"] = (
            (1.0 - freq_weight) * train_df.loc[freq_mask, "difficulty_label_0_100_pca"].values
            + freq_weight * train_df.loc[freq_mask, "difficulty_freq_0_100"].values
        )
    else:
        train_df["difficulty_label_0_100"] = train_df["difficulty_label_0_100_pca"]

    # Train predictor (any word -> difficulty)
    train_df["word_len"] = train_df["word"].astype(str).str.len()
    X = train_df[["word","word_len"]]
    y = train_df["difficulty_label_0_100"].astype(float).values

    preprocess = ColumnTransformer(
        transformers=[
            ("chars", TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=2), "word"),
            ("num", "passthrough", ["word_len"]),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    reg = Ridge(alpha=3.0, random_state=0)
    model = Pipeline([("prep", preprocess), ("model", reg)])
    model.fit(X, y)

    # Save training "database"
    train_cols = (
        ["word","form_norm"]
        + METRIC_COLS
        + [f"{c}_source_varid" for c in METRIC_COLS]
        + ["difficulty_label_0_100_pca", "difficulty_label_0_100"]
    )
    if freq_map:
        train_cols += ["frequency", "freq_log1p", "difficulty_freq_0_100", "freq_weight_used"]
    out_db = out_dir/"norare_training_database_en.csv"
    train_df[train_cols].to_csv(out_db, index=False, encoding="utf-8")

    # Save model bundle
    pca_component = pca_pipe.named_steps["pca"].components_[0].tolist()
    pca_explained = float(pca_pipe.named_steps["pca"].explained_variance_ratio_[0])

    bundle = {
        "model": model,
        "labeling": {
            "method": "PCA_1D_from_lexical_norms",
            "metric_cols": METRIC_COLS,
            "preferred_variable_ids": PREFERRED_VAR_IDS,
            "pca_pipeline": pca_pipe,
            "pca_sign": int(pca_sign),
            "pca_component_loadings_ordered_as_metric_cols": pca_component,
            "pca_explained_variance_ratio": pca_explained,
            "pc_min": pc_min,
            "pc_max": pc_max,
            "aoa_corr_with_pc1_before_scaling": corr,
            "training_rows": int(len(train_df)),
            "training_rule": "English Form; keep rows with >=3 of 6 metrics; median-impute; z-score; PCA(1); align sign with AoA; min-max to 0..100; optionally blend frequency label",
            "source": {"dataset":"NoRaRe CLDF 1.1", "input_file": norare_zip.name}
        },
        "features": {"inputs_expected":["word (string)"], "features_used":["char_ngrams_2_4_tfidf","word_len"]}
    }
    if freq_map:
        bundle["labeling"]["frequency_integration"] = {
            "words_json": words_json.name if words_json else "",
            "freq_weight": float(freq_weight),
            "clip_low_percentile": float(freq_clip_low),
            "clip_high_percentile": float(freq_clip_high),
            "log_transform": "log1p",
            "p1": float(freq_p1),
            "p99": float(freq_p99),
            "matched_rows": int(train_df["frequency"].notna().sum()),
        }
    out_model = out_dir/"difficulty_model_any_word.joblib"
    joblib.dump(bundle, out_model)

    # Manifest with dataset citations
    used_var_ids = sorted({v for vids in PREFERRED_VAR_IDS.values() for v in vids})
    used_dataset_ids = sorted({dataset_id_from_varid(v) for v in used_var_ids})

    ds_map = datasets.rename(columns={"ID":"Dataset_ID"}).set_index("Dataset_ID", drop=False)
    ds_info = []
    for dsid in used_dataset_ids:
        if dsid in ds_map.index:
            row = ds_map.loc[dsid]
            ds_info.append({
                "Dataset_ID": dsid,
                "Name": str(row.get("Name","")),
                "Year": str(row.get("Year","")),
                "Citation": str(row.get("Citation","")),
            })
        else:
            ds_info.append({"Dataset_ID": dsid, "Name":"", "Year":"", "Citation":""})

    manifest = {
        "created_from": norare_zip.name,
        "cldf_dir_in_zip": str(cldf_dir),
        "training_database_file": out_db.name,
        "model_file": out_model.name,
        "what_the_score_means": "difficulty_0_100 in [0,100]; higher means harder. Predicted latent difficulty built from lexical norms, optionally blended with words.json frequency.",
        "label_provenance": {
            "metrics": METRIC_COLS,
            "metric_sources_variable_ids": PREFERRED_VAR_IDS,
            "datasets_used": ds_info,
            "pca": {
                "imputer":"median",
                "scaler":"standard (z-score)",
                "n_components":1,
                "explained_variance_ratio": pca_explained,
                "component_loadings_ordered_as_metric_cols": pca_component,
                "sign_aligned_with":"aoa_mean (higher AoA => harder)",
                "min_max_scale_to_0_100": True,
            },
            "training_rows": int(len(train_df)),
        },
        "predictor_provenance": {
            "model": "Ridge regression on TF-IDF char ngrams (2-4) + word_len",
            "can_score_any_string": True,
        }
    }
    if freq_map:
        manifest["label_provenance"]["frequency_integration"] = {
            "words_json": words_json.name if words_json else "",
            "freq_weight": float(freq_weight),
            "clip_low_percentile": float(freq_clip_low),
            "clip_high_percentile": float(freq_clip_high),
            "log_transform": "log1p",
            "p1": float(freq_p1),
            "p99": float(freq_p99),
            "matched_rows": int(train_df["frequency"].notna().sum()),
        }
    (out_dir/"model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_db}")
    print(f"Saved: {out_model}")
    print(f"Saved: {out_dir/'model_manifest.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--norare_zip", default="Norare CLDF 1.1.zip")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--work_dir", default="norare_train_model_work")
    ap.add_argument("--words_json", default="words.json")
    ap.add_argument("--freq_weight", type=float, default=0.35)
    ap.add_argument("--freq_clip_low", type=float, default=1.0)
    ap.add_argument("--freq_clip_high", type=float, default=99.0)
    args = ap.parse_args()
    main(
        Path(args.norare_zip),
        Path(args.out_dir),
        Path(args.work_dir),
        Path(args.words_json) if args.words_json else None,
        float(args.freq_weight),
        float(args.freq_clip_low),
        float(args.freq_clip_high)
    )
