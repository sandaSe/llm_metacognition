"""
jury.py
--------
Jury pipeline
Judge panel: ibm/granite3.3:8b, cogito:8b, llama3.1:8b

Outputs:
  output/judge/judge_scores.csv  — one row per (trace x judge)
"""

from __future__ import annotations

import json
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import ollama

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import JudgeScores
from prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, format_rubric_for_judge

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_MODELS      = ["cogito:8b", "llama3.1:8b", "ibm/granite3.3:8b"]
JUDGE_TEMPERATURE = 0.1
JUDGE_MAX_TOKENS  = 4096
WORKERS           = 1      # Ollama serialises GPU inference — no benefit > 1
RETRIES           = 2

ALL_QUESTIONS = [f"Q{i}" for i in range(1, 17)]   # Q1–Q16, 16-item rubric


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_judge_response(content: str) -> dict | None:
    content = re.sub(r"```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = content.replace("```", "").strip()

    if len(content) > 50_000:
        return None

    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data.get("scores"), dict):
                for q in ALL_QUESTIONS:
                    if q not in data["scores"]:
                        data["scores"][q] = np.nan
                return data
        except json.JSONDecodeError:
            pass
    return None


def _nan_scores(
    problem_id:   str,
    model_judged: str,
    judge_model:  str,
    effort_level: str,
    sample_idx:   int,
    raw_response: str = "",
) -> JudgeScores:
    """NaN scores when a judge call fails or cannot be parsed."""
    return JudgeScores(
        problem_id   = problem_id,
        model_judged = model_judged,
        judge_model  = judge_model,
        effort_level = effort_level,
        sample_idx   = sample_idx,
        scores       = {q: np.nan for q in ALL_QUESTIONS},
        rationale    = {q: "" for q in ALL_QUESTIONS},
        raw_response = raw_response,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE JUDGE CALL
# ─────────────────────────────────────────────────────────────────────────────

def _call_judge(
    judge_model:    str,
    problem:        str,
    thinking_trace: str,
    code:           str,
    problem_id:     str,
    model_judged:   str,
    effort_level:   str,
    sample_idx:     int,
) -> JudgeScores:

    rubric_str = format_rubric_for_judge(
        thinking_trace=thinking_trace,
        problem=problem,
        code=code,
    )
    prompt = JUDGE_USER_TEMPLATE.format(
        problem        = problem,
        thinking_trace = thinking_trace or "",
        code           = code or "",
        rubric_items   = rubric_str,
    )

    raw_content = ""
    for attempt in range(RETRIES):
        try:
            response = ollama.chat(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={
                    "temperature": JUDGE_TEMPERATURE,
                    "num_predict": JUDGE_MAX_TOKENS,
                },
            )
            raw_content = response["message"]["content"]
            break
        except Exception as e:
            log.warning(
                f"[{problem_id}] judge={judge_model} "
                f"attempt {attempt+1}/{RETRIES} failed: {e}"
            )
            time.sleep(1.5 * (attempt + 1))

    if not raw_content:
        log.error(f"[{problem_id}] judge={judge_model} — no response after retries, scores set to NaN")
        return _nan_scores(problem_id, model_judged, judge_model, effort_level, sample_idx)

    parsed = _parse_judge_response(raw_content)
    if not parsed:
        log.warning(
            f"[{problem_id}] judge={judge_model} sample={sample_idx} — "
            f"parse failed, scores set to NaN. Raw response (first 800 chars):\n"
            f"{raw_content[:800]}..."
        )
        return _nan_scores(problem_id, model_judged, judge_model, effort_level, sample_idx, raw_content)

    scores = {q: parsed["scores"].get(q, np.nan) for q in ALL_QUESTIONS}
    scores = {q: (int(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan) for q, v in scores.items()}
    rationale = {
        q: str(parsed.get("rationale", {}).get(q, ""))
        for q in ALL_QUESTIONS
    }

    return JudgeScores(
        problem_id   = problem_id,
        model_judged = model_judged,
        judge_model  = judge_model,
        effort_level = effort_level,
        sample_idx   = sample_idx,
        scores       = scores,
        rationale    = rationale,
        raw_response = raw_content,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ONE TRACE WITH ALL JUDGES
# ─────────────────────────────────────────────────────────────────────────────

def score_trace_all_judges(
    problem_id:     str,
    model_judged:   str,
    effort_level:   str,
    sample_idx:     int,
    problem:        str,
    thinking_trace: str,
    code:           str,
    workers:        int = WORKERS,
) -> list[JudgeScores]:
    """Score one trace with all three judge models."""

    if not thinking_trace.strip():
        log.warning(
            f"[{problem_id}] sample={sample_idx}: "
            "empty thinking trace — all judges return NaN"
        )
        return [
            _nan_scores(problem_id, model_judged, jm, effort_level, sample_idx)
            for jm in JUDGE_MODELS
        ]

    results: list[JudgeScores] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _call_judge,
                jm, problem, thinking_trace, code,
                problem_id, model_judged, effort_level, sample_idx,
            ): jm
            for jm in JUDGE_MODELS
        }
        for future in as_completed(futures):
            jm = futures[future]
            try:
                result = future.result()
                results.append(result)
                log.info(
                    f"[{problem_id}] judge={jm} effort={effort_level} "
                    f"sample={sample_idx} — scored"
                )
            except Exception as e:
                log.error(f"[{problem_id}] judge={jm} future failed: {e}, scores set to NaN")
                results.append(
                    _nan_scores(problem_id, model_judged, jm, effort_level, sample_idx)
                )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_judge_pipeline(
    traces_csv:   str,
    problems_csv: str,
    output_dir:   str = "output/judge",
    workers:      int = WORKERS,
) -> None:
    """
    Score all thinking traces with all three judges and compute IRR.

    Input:
        traces_csv   — solver output (problem_id, model, effort_level,
                        sample_idx, thinking, code, is_error)
        problems_csv — original problems (problem_id → problem text)
        output_dir   — where to write outputs

    Output:
        output_dir/judge_scores.csv  — one row per (trace × judge)
        output_dir/irr_summary.csv   — Krippendorff alpha per trace

    Note: scores are NaN where a judge did not score a question.
    """
    traces_df   = pd.read_csv(traces_csv)
    problems_df = pd.read_csv(problems_csv)
    problems    = dict(zip(problems_df["problem_id"], problems_df["problem"]))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_score_rows: list[dict] = []
    irr_rows:       list[dict] = []

    total   = len(traces_df)
    skipped = 0

    for row_idx, row in traces_df.iterrows():

        if int(row.get("is_error", 0)) == 1:
            skipped += 1
            continue
        if not str(row.get("thinking", "")).strip():
            skipped += 1
            continue

        pid = str(row["problem_id"])
        if pid not in problems:
            log.warning(f"problem_id {pid} not in problems CSV — skipping")
            skipped += 1
            continue

        effort       = str(row.get("effort_level", "think"))
        sample_idx   = int(row.get("sample_idx", 0))
        model_judged = str(row.get("model", "unknown"))

        log.info(
            f"[{pid}] effort={effort} sample={sample_idx} "
            f"({row_idx+1}/{total}) — judging"
        )

        judge_scores_list = score_trace_all_judges(
            problem_id     = pid,
            model_judged   = model_judged,
            effort_level   = effort,
            sample_idx     = sample_idx,
            problem        = problems[pid],
            thinking_trace = str(row["thinking"]),
            code           = str(row.get("code", "")),
            workers        = workers,
        )

        for js in judge_scores_list:
            flat = {
                "problem_id":   js.problem_id,
                "model_judged": js.model_judged,
                "judge_model":  js.judge_model,
                "effort_level": js.effort_level,
                "sample_idx":   js.sample_idx,
            }
            for qid in ALL_QUESTIONS:
                flat[f"score_{qid}"]     = js.scores.get(qid, np.nan)
                flat[f"rationale_{qid}"] = js.rationale.get(qid, "")
            all_score_rows.append(flat)

        irr = compute_irr(judge_scores_list)
        irr_rows.append({
            "problem_id":              pid,
            "model_judged":            model_judged,
            "effort_level":            effort,
            "sample_idx":              sample_idx,
            "krippendorff_alpha":      irr["krippendorff_alpha"],
            "mean_pairwise_agreement": irr["mean_pairwise_agreement"],
            "reliable":                (
                irr["krippendorff_alpha"] >= 0.7
                if not np.isnan(irr["krippendorff_alpha"])
                else False
            ),
        })

        log.info(
            f"[{pid}] effort={effort} sample={sample_idx} "
            f"IRR alpha={irr['krippendorff_alpha']:.3f}"
        )

    scores_path = out_dir / "judge_scores.csv"
    irr_path    = out_dir / "irr_summary.csv"

    pd.DataFrame(all_score_rows).to_csv(scores_path, index=False)
    pd.DataFrame(irr_rows).to_csv(irr_path,          index=False)

    irr_df       = pd.DataFrame(irr_rows)
    mean_alpha   = irr_df["krippendorff_alpha"].mean() if not irr_df.empty else float("nan")
    pct_reliable = irr_df["reliable"].mean() * 100     if not irr_df.empty else 0.0

    log.info(f"\nJudge pipeline complete.")
    log.info(f"Scores  → {scores_path}  ({len(all_score_rows)} rows)")
    log.info(f"IRR     → {irr_path}     ({len(irr_rows)} rows)")
    log.info(f"Skipped : {skipped} rows (errors or empty thinking)")
    log.info(f"Mean Krippendorff alpha : {mean_alpha:.3f}")
    log.info(f"% traces above alpha=0.7: {pct_reliable:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_judge_pipeline(
        traces_csv   = "output/solver_new/traces.csv",
        problems_csv = "data/code_problems.csv",
        output_dir   = "output/judge_new",
    )