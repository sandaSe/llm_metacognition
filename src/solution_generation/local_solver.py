"""
local_solver.py
---------------
Solver pipeline for locally-hosted Ollama models.

Generates K_SAMPLES thinking traces + code solutions per problem.
Writes traces.csv and logprobs.csv to output_dir.

Supported: any Ollama model with <thinking> / <think> tag support.
Default: deepseek-r1:8b
"""

from __future__ import annotations

import re
import sys
import time
import logging
import argparse
from pathlib import Path

# Allow imports from src/ root regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import ollama

from schemas import SolverOutput, SolverBatch
from prompts import SOLVER_SYSTEM_PROMPT_STANDARD, SOLVER_USER_TEMPLATE
from logprob_utils import parse_ollama_logprobs, summarise_logprob_sequence

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL       = "deepseek-r1:8b"   # change to target Ollama model
K_SAMPLES   = 3
TEMPERATURE = 0.4

EFFORT_OPTIONS = {
    "think": {
        "temperature": TEMPERATURE,
        "num_predict": 4096,
        "logprobs":    True,
        "top_logprobs": 5,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN COUNTING
# ─────────────────────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    return max(1, int(len(text) / 3.8))


# ─────────────────────────────────────────────────────────────────────────────
# CODE BLOCK HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:python)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_last_code_block(text: str) -> str:
    """Return the last python/py fenced block, falling back to fence-stripped text."""
    blocks = re.findall(r"```python\s*\n?(.*?)\n?```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```(?:py)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return _strip_fences(text)


# ─────────────────────────────────────────────────────────────────────────────
# THINKING TRACE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_thinking(raw_content: str) -> tuple[str, str, str]:
    """
    Extract (thinking, code, extraction_type) from raw model output.

    Extraction types:
        primary  — explicit <thinking> or <think> tags, fully closed
        partial  — tag opened but not closed
        implicit — prose before first code fence (no tags)
        absent   — no thinking found
    """
    # Tier 1: <thinking> tags
    think_blocks = re.findall(
        r"<thinking>\s*(.*?)\s*</thinking>", raw_content, re.DOTALL | re.IGNORECASE
    )
    if think_blocks:
        thinking  = "\n\n".join(b.strip() for b in think_blocks)
        remaining = re.sub(r"<thinking>.*?</thinking>", "", raw_content,
                           flags=re.DOTALL | re.IGNORECASE)
        return thinking, _extract_last_code_block(remaining), "primary"

    # Tier 1 alt: native DeepSeek <think> tags
    think_blocks = re.findall(
        r"<think>\s*(.*?)\s*</think>", raw_content, re.DOTALL | re.IGNORECASE
    )
    if think_blocks:
        thinking  = "\n\n".join(b.strip() for b in think_blocks)
        remaining = re.sub(r"<think>.*?</think>", "", raw_content,
                           flags=re.DOTALL | re.IGNORECASE)
        return thinking, _extract_last_code_block(remaining), "primary"

    # Tier 2: partial <thinking>
    partial = re.search(r"<thinking>\s*(.*)", raw_content, re.DOTALL | re.IGNORECASE)
    if partial:
        after = partial.group(1).strip()
        fence = re.search(r"```(?:python|py)?", after, re.IGNORECASE)
        if fence:
            return after[:fence.start()].strip(), _extract_last_code_block(after[fence.start():]), "partial"
        return after, "", "partial"

    # Tier 2 alt: partial <think>
    partial = re.search(r"<think>\s*(.*)", raw_content, re.DOTALL | re.IGNORECASE)
    if partial:
        after = partial.group(1).strip()
        fence = re.search(r"```(?:python|py)?", after, re.IGNORECASE)
        if fence:
            return after[:fence.start()].strip(), _extract_last_code_block(after[fence.start():]), "partial"
        return after, "", "partial"

    # Tier 3: implicit prose before first code fence
    fence_match = re.search(r"```(?:python|py)?", raw_content, re.IGNORECASE)
    if fence_match:
        pre_code = raw_content[:fence_match.start()].strip()
        code     = _extract_last_code_block(raw_content[fence_match.start():])
        if any(c.isalpha() for c in pre_code):
            return pre_code, code, "implicit"
        return "", code, "absent"

    return "", _strip_fences(raw_content), "absent"


# ─────────────────────────────────────────────────────────────────────────────
# CODE SAVER
# ─────────────────────────────────────────────────────────────────────────────

def save_code_to_file(
    problem_id: str, code: str, sample_idx: int, output_dir: Path
) -> None:
    if not code.strip():
        return
    safe_id  = str(problem_id).replace(" ", "_").replace("/", "_").replace("\\", "_")
    filename = f"{safe_id}_effort-think_sample-{sample_idx:02d}.py"
    path     = output_dir / "solutions" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(code.strip() + "\n", encoding="utf-8")
        log.info(f"[{problem_id}] sample={sample_idx}: code saved → {filename}")
    except Exception as e:
        log.error(f"[{problem_id}] Failed to save code: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CALL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def call_model(problem: str, sample_idx: int) -> dict:
    options        = dict(EFFORT_OPTIONS["think"])
    options["seed"] = 42 + sample_idx
    return ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT_STANDARD},
            {"role": "user",   "content": SOLVER_USER_TEMPLATE.format(problem=problem)},
        ],
        options=options,
        think=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE SAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def process_sample(
    problem_id: str,
    problem:    str,
    sample_idx: int,
    output_dir: Path,
    retries:    int = 2,
) -> tuple[SolverOutput, dict]:

    for attempt in range(retries):
        try:
            start    = time.time()
            response = call_model(problem, sample_idx)
            latency  = round(time.time() - start, 3)

            message    = response.get("message") or {}
            raw_content = message.get("content") or ""

            dedicated_thinking = message.get("thinking") or ""
            if dedicated_thinking:
                thinking        = dedicated_thinking.strip()
                code            = _extract_last_code_block(raw_content)
                extraction_type = "primary"
                log.info(f"[{problem_id}] sample={sample_idx}: dedicated thinking key found")
            else:
                thinking, code, extraction_type = extract_thinking(raw_content)

            if extraction_type == "absent":
                log.warning(f"[{problem_id}] sample={sample_idx}: no thinking trace detected")
            elif extraction_type in ("partial", "implicit"):
                log.info(f"[{problem_id}] sample={sample_idx}: extraction_type={extraction_type}")

            thinking_len    = len(thinking)
            thinking_tokens = _count_tokens(thinking) if thinking.strip() else 0
            thinking_active = int(thinking.strip() and len(thinking.split()) > 3)

            save_code_to_file(problem_id, code, sample_idx, output_dir)

            solver_out = SolverOutput(
                problem_id=problem_id,
                model=MODEL,
                effort_level="think",
                sample_idx=sample_idx,
                thinking=thinking,
                code=code,
                raw_content=raw_content,
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
                thinking_active=thinking_active,
                thinking_len=thinking_len,
                thinking_tokens=thinking_tokens,
                extraction_type=extraction_type,
                is_error=0,
            )

            logprob_seq     = parse_ollama_logprobs(
                response=response,
                problem_id=problem_id,
                model=MODEL,
                effort_level="think",
                sample_idx=sample_idx,
                thinking_text=thinking,
            )
            logprob_summary = summarise_logprob_sequence(logprob_seq) or {}
            logprob_summary["inference_seconds"] = latency
            logprob_summary["is_error"]          = 0

            log.info(
                f"[{problem_id}] sample={sample_idx} done | "
                f"thinking_active={thinking_active} | "
                f"thinking_tokens={thinking_tokens} | "
                f"extraction={extraction_type} | latency={latency}s"
            )
            return solver_out, logprob_summary

        except Exception as e:
            log.warning(
                f"[{problem_id}] sample={sample_idx} "
                f"attempt {attempt+1}/{retries} failed: {e}"
            )
            time.sleep(1.0 * (attempt + 1))

    log.error(f"[{problem_id}] sample={sample_idx} failed after {retries} attempts")
    return SolverOutput(
        problem_id=problem_id,
        model=MODEL,
        effort_level="think",
        sample_idx=sample_idx,
        thinking="", code="", raw_content="",
        prompt_tokens=0, completion_tokens=0,
        thinking_active=0, thinking_len=0, thinking_tokens=0,
        extraction_type="absent", is_error=1,
    ), {"inference_seconds": None, "is_error": 1}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVE ONE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────

def solve_one_problem(
    problem_id: str, problem: str, output_dir: Path
) -> tuple[SolverBatch, list[dict]]:
    outputs:      list[SolverOutput] = []
    logprob_rows: list[dict]         = []

    log.info(f"[{problem_id}] effort=think — generating {K_SAMPLES} samples")

    for i in range(K_SAMPLES):
        out, lp = process_sample(problem_id, problem, i, output_dir)
        outputs.append(out)
        if lp is None:
            lp = {}
        lp["problem_id"]   = problem_id
        lp["effort_level"] = "think"
        lp["sample_idx"]   = i
        logprob_rows.append(lp)

    return SolverBatch(problem_id=problem_id, outputs=outputs), logprob_rows


# ─────────────────────────────────────────────────────────────────────────────
# RUN SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_run_summary(traces: list[dict]) -> None:
    df    = pd.DataFrame(traces)
    valid = df[df["is_error"] == 0]
    n_errors = int((df["is_error"] == 1).sum())
    if n_errors:
        log.warning(f"{n_errors} samples failed")

    if "thinking_tokens" in valid.columns:
        stats = valid["thinking_tokens"].describe().round(1)
        log.info(f"\n--- Thinking token stats ---\n{stats.to_string()}")

    if "thinking_active" in valid.columns:
        rate = valid["thinking_active"].mean()
        log.info(f"Thinking active rate: {rate:.1%} ({int(valid['thinking_active'].sum())}/{len(valid)})")

    if "extraction_type" in valid.columns:
        ext_counts = valid["extraction_type"].value_counts()
        log.info(f"\nExtraction type breakdown:\n{ext_counts.to_string()}")
        primary_rate = ext_counts.get("primary", 0) / len(valid) * 100 if len(valid) else 0
        log.info(f"Primary extraction rate: {primary_rate:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_solver_pipeline(input_csv: str, output_dir: str) -> None:
    df = pd.read_csv(input_csv)
    required = {"problem_id", "problem", "domain"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    out_dir      = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path  = out_dir / "traces.csv"
    logprobs_path = out_dir / "logprobs.csv"

    completed:        set[tuple] = set()
    all_trace_rows:   list[dict] = []
    all_logprob_rows: list[dict] = []

    # Checkpoint resume
    if traces_path.exists() and traces_path.stat().st_size > 0:
        try:
            existing  = pd.read_csv(traces_path)
            completed = set(zip(
                existing["problem_id"].astype(str),
                existing["effort_level"].astype(str),
                existing["sample_idx"].astype(int),
            ))
            all_trace_rows = existing.to_dict("records")
            log.info(f"Resuming — {len(completed)} samples complete")
        except Exception as e:
            log.warning(f"Resume failed: {e}")

    total = len(df)
    log.info(f"Starting pipeline — {total} problems × k={K_SAMPLES}")

    for idx, row in df.iterrows():
        pid     = str(row["problem_id"])
        problem = str(row["problem"])
        domain  = str(row["domain"])

        if sum(1 for i in range(K_SAMPLES) if (pid, "think", i) in completed) == K_SAMPLES:
            log.info(f"[{pid}] already complete — skipping")
            continue

        log.info(f"=== [{idx+1}/{total}] Problem {pid} ({domain}) ===")

        batch, logprob_summaries = solve_one_problem(pid, problem, out_dir)

        for out in batch.outputs:
            key = (out.problem_id, out.effort_level, out.sample_idx)
            if key in completed:
                continue
            all_trace_rows.append({
                "problem_id":        pid,
                "domain":            domain,
                "model":             out.model,
                "effort_level":      out.effort_level,
                "sample_idx":        out.sample_idx,
                "thinking":          out.thinking,
                "code":              out.code,
                "prompt_tokens":     out.prompt_tokens,
                "completion_tokens": out.completion_tokens,
                "thinking_active":   out.thinking_active,
                "thinking_len":      out.thinking_len,
                "thinking_tokens":   out.thinking_tokens,
                "extraction_type":   out.extraction_type,
                "is_error":          out.is_error,
            })
            completed.add(key)

        for lp in logprob_summaries:
            lp["domain"] = domain
            all_logprob_rows.append(lp)

        pd.DataFrame(all_trace_rows).to_csv(traces_path,    index=False)
        pd.DataFrame(all_logprob_rows).to_csv(logprobs_path, index=False)
        log.info(f"[{pid}] saved — {idx+1}/{total} done")

    log.info(f"\nPipeline complete. Traces: {len(all_trace_rows)} rows")
    print_run_summary(all_trace_rows)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local Ollama solver — metacognitive evaluation pipeline"
    )
    parser.add_argument("--input_csv",  default="data/code_problems.csv",
                        help="Path to problems CSV (problem_id, problem, domain)")
    parser.add_argument("--output_dir", default="output/solver_local",
                        help="Directory for traces.csv and logprobs.csv")
    args = parser.parse_args()

    run_solver_pipeline(
        input_csv  = args.input_csv,
        output_dir = args.output_dir,
    )