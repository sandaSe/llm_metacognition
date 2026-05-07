"""
cfc.py
------
CoT Faithfulness Check (CFC) — binary alignment scorer.

Determines whether the final code faithfully implements the reasoning
described in the thinking trace, for each (thinking, code) pair.

Uses qwen2.5-coder:7b exclusively — independent of all solver and judge models.

Outputs:
    output/cfs/cfs_scores.csv — one row per trace, code_aligned: True/False
"""

from __future__ import annotations

import json
import re
import sys
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow imports from src/ root regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import ollama

from schemas import CFSScore
from prompts import CFS_SYSTEM_PROMPT, CFS_USER_TEMPLATE

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

CFS_JUDGE_MODEL = "qwen2.5-coder:7b"
TEMPERATURE     = 0.1
MAX_TOKENS      = 512    # binary response — very few tokens needed
WORKERS         = 1
RETRIES         = 2


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING
# qwen2.5-coder:7b frequently wraps JSON in ```json...``` fences;
# fence-stripping is required before parsing.
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cfs_response(content: str) -> dict | None:
    """Strip markdown fences, find outermost JSON object, parse and return."""
    content = re.sub(r"```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = content.replace("```", "").strip()

    # Try largest JSON object first — most complete
    matches = re.findall(r"\{[\s\S]*?\}", content)
    if not matches:
        return None

    for block in sorted(matches, key=len, reverse=True):
        try:
            data = json.loads(block)
            if "code_aligned" in data:
                return data
        except json.JSONDecodeError:
            continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE CFC CALL
# ─────────────────────────────────────────────────────────────────────────────

def score_one_trace(
    problem_id:   str,
    model_name:   str,
    effort_level: str,
    sample_idx:   int,
    problem:      str,
    thinking:     str,
    code:         str,
    retries:      int = RETRIES,
) -> CFSScore:
    """Score faithfulness for one (thinking, code) pair."""

    # No thinking — skip
    if not thinking.strip():
        log.info(
            f"[{problem_id}] sample={sample_idx}: "
            "empty thinking trace — CFC skipped"
        )
        return CFSScore(
            problem_id=problem_id, model=model_name,
            effort_level=effort_level, sample_idx=sample_idx,
            code_aligned=False, raw_response="", skipped=True,
        )

    # No code — cannot evaluate alignment
    if not code.strip():
        log.warning(
            f"[{problem_id}] sample={sample_idx}: "
            "empty code — CFC cannot be computed"
        )
        return CFSScore(
            problem_id=problem_id, model=model_name,
            effort_level=effort_level, sample_idx=sample_idx,
            code_aligned=False, raw_response="", skipped=False,
        )

    prompt = CFS_USER_TEMPLATE.format(
        problem=problem,
        thinking_trace=thinking,
        code=code,
    )

    raw_content = ""
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=CFS_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": CFS_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                },
            )
            raw_content = response["message"]["content"]
            break
        except Exception as e:
            log.error(
                f"[{problem_id}] CFC attempt {attempt+1}/{retries} error: {e}"
            )
            time.sleep(1.0 * (attempt + 1))

    if not raw_content:
        return CFSScore(
            problem_id=problem_id, model=model_name,
            effort_level=effort_level, sample_idx=sample_idx,
            code_aligned=False, raw_response="", skipped=False,
        )

    parsed = _parse_cfs_response(raw_content)

    if not parsed:
        log.warning(
            f"[{problem_id}] sample={sample_idx}: "
            f"CFC JSON parse failed — raw: {raw_content[:120]!r}"
        )
        return CFSScore(
            problem_id=problem_id, model=model_name,
            effort_level=effort_level, sample_idx=sample_idx,
            code_aligned=False, raw_response=raw_content, skipped=False,
        )

    aligned = bool(parsed.get("code_aligned", False))
    log.info(
        f"[{problem_id}] sample={sample_idx} effort={effort_level} "
        f"| code_aligned={aligned}"
    )

    return CFSScore(
        problem_id=problem_id, model=model_name,
        effort_level=effort_level, sample_idx=sample_idx,
        code_aligned=aligned, raw_response=raw_content, skipped=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_cfc_pipeline(
    traces_csv:   str,
    problems_csv: str,
    output_dir:   str = "output/cfc",
    workers:      int = WORKERS,
) -> None:
    """
    Score faithfulness for all traces in traces.csv.

    Input:
        traces_csv   — solver output (thinking + code columns required)
        problems_csv — original problems CSV (problem_id → problem text)
        output_dir   — where to write cfs_scores.csv
    """
    traces_df   = pd.read_csv(traces_csv)
    problems_df = pd.read_csv(problems_csv)

    required = {"problem_id", "thinking", "code"}
    missing  = required - set(traces_df.columns)
    if missing:
        raise ValueError(f"traces CSV missing required columns: {missing}")

    problems = dict(zip(problems_df["problem_id"], problems_df["problem"]))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks          = []
    skipped_errors = 0

    for _, row in traces_df.iterrows():
        if int(row.get("is_error", 0)) == 1:
            skipped_errors += 1
            continue
        pid = str(row["problem_id"])
        if pid not in problems:
            log.warning(f"problem_id {pid} not found in problems CSV — skipping")
            continue
        tasks.append({
            "problem_id":   pid,
            "model_name":   str(row.get("model", "unknown")),
            "effort_level": str(row.get("effort_level", "think")),
            "sample_idx":   int(row.get("sample_idx", 0)),
            "problem":      problems[pid],
            "thinking":     str(row.get("thinking", "")),
            "code":         str(row.get("code", "")),
        })

    log.info(
        f"CFC pipeline: {len(tasks)} traces → {CFS_JUDGE_MODEL} "
        f"({skipped_errors} error rows skipped)"
    )

    results: list[CFSScore] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(score_one_trace, **task): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                log.error(
                    f"[{task['problem_id']}] sample={task['sample_idx']} "
                    f"CFC future failed: {e}"
                )

    out_path = out_dir / "cfs_scores.csv"
    pd.DataFrame([r.model_dump() for r in results]).to_csv(out_path, index=False)

    n_aligned = sum(1 for r in results if r.code_aligned and not r.skipped)
    n_scored  = sum(1 for r in results if not r.skipped)
    n_skipped = sum(1 for r in results if r.skipped)
    rate      = n_aligned / n_scored * 100 if n_scored > 0 else 0.0

    log.info(f"\nCFC complete → {out_path} ({len(results)} rows)")
    log.info(f"  Scored:   {n_scored}")
    log.info(f"  Skipped:  {n_skipped}  (empty thinking traces)")
    log.info(f"  Aligned:  {n_aligned}  ({rate:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="CoT Faithfulness Check — binary alignment scorer"
    )
    parser.add_argument("--traces_csv",   required=True,
                        help="Path to solver traces.csv")
    parser.add_argument("--problems_csv", default="data/code_problems.csv",
                        help="Path to problems CSV")
    parser.add_argument("--output_dir",   default="output/cfc",
                        help="Output directory for cfs_scores.csv")
    args = parser.parse_args()

    run_cfc_pipeline(
        traces_csv   = args.traces_csv,
        problems_csv = args.problems_csv,
        output_dir   = args.output_dir,
    )