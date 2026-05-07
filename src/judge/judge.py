"""
judge_glm.py
------------
Judge pipeline using Azure OpenAI as the single judge model.

Azure OpenAI is called via the openai SDK with AzureOpenAI client.

Works for all solver outputs:
    GLM, Qwen3, DeepSeek, GPT-OSS, Gemini, Grok-3-mini, Mistral, Claude

Skips rows where is_error==1 or thinking trace is empty.

Produces:
    output_dir/judge_scores.csv     — one row per (trace × call)

Note: scores are NaN where a judge did not score a question.
"""

from __future__ import annotations

import json
import os
import re
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import JudgeScores
from prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, format_rubric_for_judge

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

N_CALLS     = 3      # number of times to call the same judge per trace
                     # used for self-consistency reliability estimate
TEMPERATURE = 0.1
MAX_TOKENS  = 4096
RETRIES     = 2

ALL_QUESTIONS = [f"Q{i}" for i in range(1, 17)]


# ─────────────────────────────────────────────────────────────────────────────
# AZURE OPENAI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

_client: AzureOpenAI | None = None

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        endpoint    = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key     = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

        if not endpoint or not api_key:
            raise EnvironmentError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env"
            )

        _client = AzureOpenAI(
            azure_endpoint = endpoint,
            api_key        = api_key,
            api_version    = api_version,
        )
    return _client


def _get_deployment() -> str:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise EnvironmentError("AZURE_OPENAI_DEPLOYMENT must be set in .env")
    return deployment


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_judge_response(content: str) -> dict | None:
    content = re.sub(r"```(?:json)?\s*", "", content, flags=re.IGNORECASE).replace("```", "").strip()
    if len(content) > 100_000:
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
        except Exception:
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
# SINGLE AZURE JUDGE CALL
# ─────────────────────────────────────────────────────────────────────────────

def _call_azure_judge(
    problem:        str,
    thinking_trace: str,
    code:           str,
    problem_id:     str,
    model_judged:   str,
    effort_level:   str,
    sample_idx:     int,
    call_idx:       int,
) -> JudgeScores:

    rubric_str = format_rubric_for_judge(
        thinking_trace = thinking_trace,
        problem        = problem,
        code           = code,
    )
    prompt = JUDGE_USER_TEMPLATE.format(
        problem        = problem,
        thinking_trace = thinking_trace or "",
        code           = code or "",
        rubric_items   = rubric_str,
    )

    deployment  = _get_deployment()
    judge_label = f"{deployment}-call{call_idx}"
    raw_content = ""

    for attempt in range(RETRIES):
        try:
            response = _get_client().chat.completions.create(
                model           = deployment,
                messages        = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature     = TEMPERATURE,
                max_tokens      = MAX_TOKENS,
                response_format = {"type": "json_object"},
            )
            raw_content = response.choices[0].message.content or ""
            break
        except Exception as e:
            log.warning(f"[{problem_id}] Azure attempt {attempt+1} call={call_idx}: {e}")
            time.sleep(2 ** attempt)

    if not raw_content:
        log.error(f"[{problem_id}] Azure judge call={call_idx} — no response after retries, scores set to NaN")
        return _nan_scores(problem_id, model_judged, judge_label, effort_level, sample_idx)

    parsed = _parse_judge_response(raw_content)
    if not parsed:
        log.warning(f"[{problem_id}] Azure judge call={call_idx} — parse failed, scores set to NaN")
        return _nan_scores(problem_id, model_judged, judge_label, effort_level, sample_idx, raw_content)

    scores = {q: parsed["scores"].get(q, np.nan) for q in ALL_QUESTIONS}
    scores = {q: (int(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan) for q, v in scores.items()}
    rationale = {q: str(parsed.get("rationale", {}).get(q, "")) for q in ALL_QUESTIONS}

    return JudgeScores(
        problem_id   = problem_id,
        model_judged = model_judged,
        judge_model  = judge_label,
        effort_level = effort_level,
        sample_idx   = sample_idx,
        scores       = scores,
        rationale    = rationale,
        raw_response = raw_content,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ONE TRACE — N_CALLS times
# ─────────────────────────────────────────────────────────────────────────────

def score_trace(
    problem_id:     str,
    model_judged:   str,
    effort_level:   str,
    sample_idx:     int,
    problem:        str,
    thinking_trace: str,
    code:           str,
) -> list[JudgeScores]:
    """
    Call the Azure judge N_CALLS times for one trace.
    Returns list of N_CALLS JudgeScores for self-consistency analysis.
    Scores are NaN where the judge did not score a question.
    """
    deployment = _get_deployment()

    if not thinking_trace.strip():
        return [
            _nan_scores(
                problem_id, model_judged,
                f"{deployment}-call{i}", effort_level, sample_idx,
            )
            for i in range(N_CALLS)
        ]

    results = []
    for call_idx in range(N_CALLS):
        js = _call_azure_judge(
            problem        = problem,
            thinking_trace = thinking_trace,
            code           = code,
            problem_id     = problem_id,
            model_judged   = model_judged,
            effort_level   = effort_level,
            sample_idx     = sample_idx,
            call_idx       = call_idx,
        )
        results.append(js)
        log.info(
            f"[{problem_id}] Azure judge call={call_idx}/{N_CALLS-1} "
            f"effort={effort_level} sample={sample_idx} — scored"
        )
        if call_idx < N_CALLS - 1:
            time.sleep(0.5)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_judge_pipeline(
    traces_csv:   str,
    problems_csv: str,
    output_dir:   str,
) -> None:
    """
    Judge all traces using Azure OpenAI, N_CALLS times per trace.

    Skips rows where is_error==1 or thinking trace is empty.

    Outputs:
        judge_scores.csv  — all N_CALLS rows per trace (for audit)
        irr_summary.csv   — self-consistency alpha per trace
        final_scores.csv  — mean scores across calls (use for MRI downstream)

    Note: scores are NaN where a judge did not score a question.
    """
    traces_df   = pd.read_csv(traces_csv)
    problems_df = pd.read_csv(problems_csv)
    problems    = dict(zip(problems_df["problem_id"], problems_df["problem"]))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deployment = _get_deployment()
    log.info(f"Azure OpenAI judge: {deployment}  (N_CALLS={N_CALLS} per trace)")
    log.info(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'NOT SET')}")

    all_score_rows:   list[dict] = []
    irr_rows:         list[dict] = []
    final_score_rows: list[dict] = []

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
            log.warning(f"problem_id {pid} not found — skipping")
            skipped += 1
            continue

        effort       = str(row.get("effort_level", "think"))
        sample_idx   = int(row.get("sample_idx", 0))
        model_judged = str(row.get("model", "unknown"))

        log.info(
            f"[{pid}] model={model_judged} effort={effort} "
            f"sample={sample_idx} ({row_idx+1}/{total})"
        )

        call_results = score_trace(
            problem_id     = pid,
            model_judged   = model_judged,
            effort_level   = effort,
            sample_idx     = sample_idx,
            problem        = problems[pid],
            thinking_trace = str(row["thinking"]),
            code           = str(row.get("code", "")),
        )

        # ── Flatten all N_CALLS to score rows ────────────────────────────
        for js in call_results:
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

        # ── Self-consistency reliability ──────────────────────────────────
        call_score_dicts = [js.scores for js in call_results]
        sc_metrics = compute_self_consistency_alpha(call_score_dicts)
        sc_alpha   = sc_metrics["alpha"]
        sc_var     = sc_metrics["score_variance"]
        sc_agree   = compute_mean_pairwise_agreement(call_score_dicts)

        irr_rows.append({
            "problem_id":              pid,
            "model_judged":            model_judged,
            "effort_level":            effort,
            "sample_idx":              sample_idx,
            "self_consistency_alpha":  sc_alpha,
            "mean_pairwise_agreement": sc_agree,
            "score_variance":          sc_var,
            "stable": (sc_alpha >= 0.6 if not np.isnan(sc_alpha) else False),
            "note": "Self-consistency (same judge called 3x) — NOT inter-rater reliability.",
        })
        log.info(
            f"[{pid}] self-consistency alpha={sc_alpha:.3f}  "
            f"agreement={sc_agree:.3f}"
        )

        # ── Mean scores across N_CALLS — used downstream by mri.py ───────
        mean_scores = {}
        for qid in ALL_QUESTIONS:
            vals = [js.scores.get(qid, np.nan) for js in call_results]
            valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            mean_scores[qid] = round(float(np.mean(valid)), 3) if valid else np.nan

        best_rationale = {}
        for js in call_results:
            if any(v for v in js.rationale.values() if v):
                best_rationale = js.rationale
                break

        final_row = {
            "problem_id":   pid,
            "model_judged": model_judged,
            "judge_model":  deployment,
            "effort_level": effort,
            "sample_idx":   sample_idx,
        }
        for qid in ALL_QUESTIONS:
            final_row[f"score_{qid}"]     = mean_scores[qid]
            final_row[f"rationale_{qid}"] = best_rationale.get(qid, "")
        final_score_rows.append(final_row)

    # ── Save outputs ──────────────────────────────────────────────────────
    scores_path = out_dir / "judge_scores.csv"
    irr_path    = out_dir / "irr_summary.csv"
    final_path  = out_dir / "final_scores.csv"

    pd.DataFrame(all_score_rows).to_csv(scores_path,  index=False)
    pd.DataFrame(irr_rows).to_csv(irr_path,           index=False)
    pd.DataFrame(final_score_rows).to_csv(final_path, index=False)

    irr_df     = pd.DataFrame(irr_rows)
    mean_alpha = irr_df["self_consistency_alpha"].mean() if not irr_df.empty else float("nan")
    pct_stable = irr_df["stable"].mean() * 100           if not irr_df.empty else 0.0

    log.info(f"\nJudge pipeline (Azure OpenAI) complete.")
    log.info(f"Judge model : {deployment}  (called {N_CALLS}× per trace)")
    log.info(f"Scores      → {scores_path}  ({len(all_score_rows)} rows)")
    log.info(f"IRR / SC    → {irr_path}     ({len(irr_rows)} rows)")
    log.info(f"Final       → {final_path}   ({len(final_score_rows)} rows)")
    log.info(f"Skipped     : {skipped} rows (errors / empty thinking)")
    log.info(f"Mean self-consistency alpha : {mean_alpha:.3f}")
    log.info(f"% traces stable (α≥0.6)     : {pct_stable:.1f}%")
    log.info(
        "\nNote: final_scores.csv contains mean scores across calls. "
        "Pass this to mri.py as --judge_csv."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Azure OpenAI judge pipeline — metacognitive rubric scoring"
    )
    parser.add_argument("--traces_csv",   required=True,
                        help="Path to solver traces.csv")
    parser.add_argument("--problems_csv", default="data/problems.csv")
    parser.add_argument("--output_dir",   required=True,
                        help="Output directory (e.g. output/judge_azure)")
    args = parser.parse_args()

    run_judge_pipeline(
        traces_csv   = args.traces_csv,
        problems_csv = args.problems_csv,
        output_dir   = args.output_dir,
    )