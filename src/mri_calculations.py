"""
mri_calculations.py
-------------------
Metacognitive Reasoning Index (MRI) — score aggregation and model summary.

Reads judge final_scores CSV files (one per model), computes dimension scores
(Planning S_P, Monitoring S_M, Evaluation S_E), derives the MRI composite,
and produces model-level summaries with 95% confidence intervals.

Input:
    --judge_dir   Directory containing one or more final_scores.csv files
                  produced by judge.py or jury.py.
                  Each file must contain: problem_id, model_judged, sample_idx,
                  max_tokens, score_Q1 … score_Q16.

    --output_csv  Path for the model-level MRI summary CSV
                  (default: output/mri/mri_summary.csv)

Output:
    mri_summary.csv       — MRI_mean, MRI_std, MRI_ci per (model, max_tokens)
    dimension_summary.csv — mean S_P, S_M, S_E per (model, max_tokens)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ci95(x: pd.Series) -> float:
    """95% t-based confidence interval for the mean."""
    n = len(x)
    if n <= 1:
        return 0.0
    se = stats.sem(x)
    return float(se * stats.t.ppf(0.975, df=n - 1))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_mri_pipeline(judge_dir: str, output_csv: str) -> None:
    """
    Aggregate judge scores into MRI per model.

    Steps:
        1. Load and concatenate all final_scores.csv files from judge_dir.
        2. Average across judges (judge_model) per (problem, model, sample, max_tokens).
        3. Compute dimension scores S_P, S_M, S_E.
        4. Compute MRI = (S_P + S_M + S_E) / 3.
        5. Summarise per (model_judged, max_tokens).
    """
    judge_path = Path(judge_dir)
    csv_files  = sorted(judge_path.rglob("final_scores.csv"))

    if not csv_files:
        # Fall back to any judge_scores.csv
        csv_files = sorted(judge_path.rglob("judge_scores.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No final_scores.csv or judge_scores.csv found under {judge_dir}"
        )

    log.info(f"Loading {len(csv_files)} judge file(s) from {judge_dir}")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        log.info(f"  {f.relative_to(judge_path)} — {len(df)} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Combined: {len(combined)} rows, {combined['model_judged'].nunique()} models")

    score_cols = [f"score_Q{q}" for q in range(1, 17)]
    missing = [c for c in score_cols if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    # ── 1. Average across judges ───────────────────────────────────────────
    group_keys = ["problem_id", "model_judged", "sample_idx"]
    if "max_tokens" in combined.columns:
        group_keys.append("max_tokens")

    judge_avg = (
        combined
        .groupby(group_keys)[score_cols]
        .mean()
        .reset_index()
    )

    # ── 2. Dimension scores ────────────────────────────────────────────────
    P_cols = [f"score_Q{q}" for q in range(1,  7)]   # Q1–Q6   Planning     (n=6)
    M_cols = [f"score_Q{q}" for q in range(7,  12)]  # Q7–Q11  Monitoring   (n=5)
    E_cols = [f"score_Q{q}" for q in range(12, 17)]  # Q12–Q16 Evaluation   (n=5)

    judge_avg["S_P"] = judge_avg[P_cols].mean(axis=1)
    judge_avg["S_M"] = judge_avg[M_cols].mean(axis=1)
    judge_avg["S_E"] = judge_avg[E_cols].mean(axis=1)

    # ── 3. MRI composite ──────────────────────────────────────────────────
    judge_avg["MRI"] = (judge_avg["S_P"] + judge_avg["S_M"] + judge_avg["S_E"]) / 3

    # ── 4. Model-level MRI summary ────────────────────────────────────────
    summary_keys = ["model_judged"]
    if "max_tokens" in judge_avg.columns:
        summary_keys.append("max_tokens")

    model_mri = (
        judge_avg
        .groupby(summary_keys)["MRI"]
        .agg(
            MRI_mean = "mean",
            MRI_std  = "std",
            MRI_ci   = ci95,
            n_obs    = "count",
        )
        .reset_index()
        .sort_values("MRI_mean", ascending=False)
    )

    # ── 5. Dimension means ─────────────────────────────────────────────────
    dim_means = (
        judge_avg
        .groupby(summary_keys)[["S_P", "S_M", "S_E"]]
        .mean()
        .reset_index()
        .sort_values("S_P", ascending=False)
    )

    # ── 6. Save ───────────────────────────────────────────────────────────
    out_dir    = Path(output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mri_path = Path(output_csv)
    dim_path = out_dir / "dimension_summary.csv"

    model_mri.to_csv(mri_path, index=False)
    dim_means.to_csv(dim_path, index=False)

    log.info(f"\nMRI summary → {mri_path}")
    log.info(f"Dimension   → {dim_path}")
    log.info(f"\n--- Model MRI Summary ---\n{model_mri.to_string(index=False)}")
    log.info(f"\n--- Dimension Score Means ---\n{dim_means.to_string(index=False)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MRI composite scores from judge output"
    )
    parser.add_argument(
        "--judge_dir", required=True,
        help="Directory containing final_scores.csv file(s) from the judge pipeline",
    )
    parser.add_argument(
        "--output_csv", default="output/mri/mri_summary.csv",
        help="Path for the MRI model-level summary CSV",
    )
    args = parser.parse_args()

    run_mri_pipeline(
        judge_dir  = args.judge_dir,
        output_csv = args.output_csv,
    )