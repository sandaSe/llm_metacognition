"""
commercial_solver.py
--------------------
Unified commercial API solver — metacognitive evaluation pipeline.

Supported models:
  THINKING:
    gemini-2.5-flash   — Google GenAI, thinking via thinkingBudget + include_thoughts

  NON-THINKING BASELINES:
    mistral-large      — Mistral Chat API

API keys (set in .env or environment):
    GEMINI_API_KEY, MISTRAL_API_KEY
"""

from __future__ import annotations

import os
import re
import sys
import time
import logging
import argparse
import warnings
from pathlib import Path

# Allow imports from src/ root regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from schemas import SolverOutput
from prompts import SOLVER_SYSTEM_PROMPT_STANDARD, SOLVER_USER_TEMPLATE

warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=UserWarning, message=".*LibreSSL.*")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "gemini-2.5-flash": {
        "provider":      "gemini",
        "api_model":     "gemini-2.5-flash",
        "effort_levels": ["think"],
        "temperature":   0.4,
        "max_tokens":    8192,
        "thinking":      True,
    },
    "mistral-large": {
        "provider":      "mistral",
        "api_model":     "mistral-large-2512",
        "effort_levels": ["think"],
        "temperature":   0.4,
        "max_tokens":    4096,
        "thinking":      False,
    },
}

K_SAMPLES = 3


# ─────────────────────────────────────────────────────────────────────────────
# LAZY CLIENTS
# ─────────────────────────────────────────────────────────────────────────────

_clients: dict = {}

def _get_gemini():
    if "gemini" not in _clients:
        from google import genai
        _clients["gemini"] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _clients["gemini"]

def _get_mistral():
    if "mistral" not in _clients:
        from mistralai import Mistral
        _clients["mistral"] = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    return _clients["mistral"]


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:python)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_last_code_block(text: str) -> str:
    for lang in ["python", "py", ""]:
        pattern = rf"```{lang}\s*\n?(.*?)\n?```" if lang else r"```\s*\n?(.*?)\n?```"
        blocks  = re.findall(pattern, text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
    return _strip_fences(text)


def extract_thinking(raw_content: str) -> tuple[str, str, str]:
    """Four-tier extraction — returns (thinking, code, extraction_type)."""
    # Tier 1: <thinking> tags
    think_blocks = re.findall(
        r"<thinking>\s*(.*?)\s*</thinking>",
        raw_content, re.DOTALL | re.IGNORECASE
    )
    if think_blocks:
        thinking  = "\n\n".join(b.strip() for b in think_blocks)
        remaining = re.sub(r"<thinking>.*?</thinking>", "", raw_content,
                           flags=re.DOTALL | re.IGNORECASE)
        return thinking, _extract_last_code_block(remaining), "primary"

    # Tier 2: implicit prose before first fence
    fence = re.search(r"```(?:python|py)?", raw_content, re.IGNORECASE)
    if fence:
        pre = raw_content[:fence.start()].strip()
        if any(c.isalpha() for c in pre):
            return pre, _extract_last_code_block(raw_content[fence.start():]), "implicit"
        return "", _extract_last_code_block(raw_content), "absent"

    return "", _strip_fences(raw_content), "absent"


def _count_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    return max(1, int(len(text) / 4.0))


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER CALL FUNCTIONS
# Each returns: (thinking, code, raw_content, extraction_type, prompt_tokens, completion_tokens)
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(
    config: dict, problem: str, effort: str, sample_idx: int, problem_id: str
) -> tuple[str, str, str, str, int, int]:
    from google import genai
    from google.genai import types

    think_enabled = (effort == "think")
    budget        = 8192 if think_enabled else 0

    resp = _get_gemini().models.generate_content(
        model=config["api_model"],
        contents=SOLVER_USER_TEMPLATE.format(problem=problem),
        config=types.GenerateContentConfig(
            system_instruction=SOLVER_SYSTEM_PROMPT_STANDARD,
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"],
            thinking_config=types.ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=think_enabled,
            ),
        ),
    )

    thinking_parts: list[str] = []
    text_parts:     list[str] = []

    if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
        for part in resp.candidates[0].content.parts:
            part_text = getattr(part, "text", "") or ""
            if getattr(part, "thought", False):
                thinking_parts.append(part_text)
            else:
                text_parts.append(part_text)

    thinking = "\n\n".join(thinking_parts).strip()
    raw      = "\n".join(text_parts).strip()

    # Fallback: detect prose before first fence if Gemini didn't separate thoughts
    if not thinking and raw and think_enabled:
        fence = re.search(r"```(?:python|py)?", raw, re.IGNORECASE)
        if fence:
            pre = raw[:fence.start()].strip()
            if any(c.isalpha() for c in pre):
                thinking = pre
                raw      = raw[fence.start():]

    code = _extract_last_code_block(raw)
    ext  = "primary" if thinking and len(thinking.split()) > 5 else (
           "implicit" if thinking else "absent")

    # Coerce None → 0 (Gemini returns None for unused token types)
    usage        = getattr(resp, "usage_metadata", None)
    pt           = (getattr(usage, "prompt_token_count",     None) or 0)
    ct           = (getattr(usage, "candidates_token_count", None) or 0)
    thoughts_tok = (getattr(usage, "thoughts_token_count",   None) or 0)

    if thoughts_tok > 0:
        log.info(f"[{problem_id}] Gemini thinking tokens: {thoughts_tok}")

    return thinking, code, raw, ext, pt, ct


def _call_mistral(
    config: dict, problem: str, effort: str, sample_idx: int, problem_id: str
) -> tuple[str, str, str, str, int, int]:
    resp = _get_mistral().chat.complete(
        model=config["api_model"],
        messages=[
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT_STANDARD},
            {"role": "user",   "content": SOLVER_USER_TEMPLATE.format(problem=problem)},
        ],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    raw      = resp.choices[0].message.content or ""
    thinking, code, ext = extract_thinking(raw)
    pt = getattr(resp.usage, "prompt_tokens",     0) or 0
    ct = getattr(resp.usage, "completion_tokens", 0) or 0
    return thinking, code, raw, ext, pt, ct


_PROVIDER_FN = {
    "gemini":  _call_gemini,
    "mistral": _call_mistral,
}


# ─────────────────────────────────────────────────────────────────────────────
# CODE SAVER
# ─────────────────────────────────────────────────────────────────────────────

def save_code_to_file(
    problem_id: str, code: str, model_key: str,
    effort: str, sample_idx: int, output_dir: Path,
) -> None:
    if not code or len(code.strip()) < 10:
        return
    safe_id  = str(problem_id).replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_mdl = model_key.replace(".", "-").replace(":", "-")
    path     = (
        output_dir / "solutions"
        / f"{safe_id}_{safe_mdl}_effort-{effort}_sample-{sample_idx:02d}.py"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(code.strip() + "\n", encoding="utf-8")
        log.info(f"[{problem_id}] code saved → {path.name}")
    except Exception as e:
        log.error(f"[{problem_id}] Failed to save code: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE SAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def process_sample(
    problem_id: str, problem: str, model_key: str,
    effort: str, sample_idx: int, output_dir: Path,
    retries: int = 2,
) -> tuple[SolverOutput, dict]:

    config   = MODEL_REGISTRY[model_key]
    provider = config["provider"]
    fn       = _PROVIDER_FN[provider]

    for attempt in range(retries):
        try:
            start = time.time()
            thinking, code, raw_content, extraction_type, pt, ct = fn(
                config, problem, effort, sample_idx, problem_id
            )
            latency = round(time.time() - start, 3)

            thinking_tokens = _count_tokens(thinking)
            thinking_active = int(thinking.strip() and len(thinking.split()) > 3)

            if config.get("thinking") and effort == "think" and extraction_type == "absent":
                log.warning(
                    f"[{problem_id}] {model_key} effort=think sample={sample_idx}: "
                    "no thinking trace detected"
                )

            save_code_to_file(problem_id, code, model_key, effort, sample_idx, output_dir)

            solver_out = SolverOutput(
                problem_id=problem_id, model=model_key,
                effort_level=effort, sample_idx=sample_idx,
                thinking=thinking, code=code, raw_content=raw_content,
                prompt_tokens=pt, completion_tokens=ct,
                thinking_active=thinking_active,
                thinking_len=len(thinking),
                thinking_tokens=thinking_tokens,
                extraction_type=extraction_type,
                is_error=0,
            )

            log.info(
                f"[{problem_id}] {model_key} effort={effort} sample={sample_idx} | "
                f"thinking_active={thinking_active} thinking_tokens={thinking_tokens} "
                f"extraction={extraction_type} latency={latency}s"
            )
            return solver_out, {"inference_seconds": latency, "is_error": 0}

        except Exception as e:
            log.warning(
                f"[{problem_id}] {model_key} effort={effort} sample={sample_idx} "
                f"attempt {attempt+1}/{retries} failed: {e}"
            )
            time.sleep(2.0 * (attempt + 1))

    log.error(f"[{problem_id}] {model_key} effort={effort} sample={sample_idx} — all retries failed")
    return SolverOutput(
        problem_id=problem_id, model=model_key,
        effort_level=effort, sample_idx=sample_idx,
        thinking="", code="", raw_content="",
        prompt_tokens=0, completion_tokens=0,
        thinking_active=0, thinking_len=0, thinking_tokens=0,
        extraction_type="absent", is_error=1,
    ), {"inference_seconds": None, "is_error": 1}


# ─────────────────────────────────────────────────────────────────────────────
# SOLVE ONE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────

def solve_one_problem(
    problem_id: str, problem: str, model_key: str, output_dir: Path,
) -> tuple[list[SolverOutput], list[dict]]:

    config  = MODEL_REGISTRY[model_key]
    outputs: list[SolverOutput] = []
    lp_rows: list[dict]         = []

    for effort in config["effort_levels"]:
        log.info(f"[{problem_id}] {model_key} effort={effort} — {K_SAMPLES} samples")
        for i in range(K_SAMPLES):
            out, lp = process_sample(problem_id, problem, model_key, effort, i, output_dir)
            outputs.append(out)
            lp.update({"problem_id": problem_id, "effort_level": effort, "sample_idx": i})
            lp_rows.append(lp)

    return outputs, lp_rows


# ─────────────────────────────────────────────────────────────────────────────
# RUN SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_run_summary(traces: list[dict], model_key: str) -> None:
    df    = pd.DataFrame(traces)
    valid = df[df["is_error"] == 0]
    if valid.empty:
        log.warning("No valid samples to summarise")
        return

    n_errors = int((df["is_error"] == 1).sum())
    if n_errors:
        log.warning(f"{n_errors} samples failed — excluded from summary")

    if "effort_level" in valid.columns and "thinking_active" in valid.columns:
        for effort, sub in valid.groupby("effort_level"):
            rate = sub["thinking_active"].mean()
            log.info(
                f"[{model_key}] effort={effort} thinking active: "
                f"{rate:.1%} ({int(sub['thinking_active'].sum())}/{len(sub)})"
            )

    if "extraction_type" in valid.columns:
        log.info(f"\nExtraction breakdown:\n{valid['extraction_type'].value_counts().to_string()}")

    if "thinking_tokens" in valid.columns and valid["thinking_tokens"].sum() > 0:
        log.info(f"\nThinking tokens:\n{valid['thinking_tokens'].describe().round(1).to_string()}")

    config = MODEL_REGISTRY.get(model_key, {})
    if not config.get("thinking"):
        log.info(
            f"\nNote: {model_key} is a non-thinking model — "
            "MRI/CFS scoring will be skipped. Contributes correctness axis only."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE — with checkpoint/resume
# ─────────────────────────────────────────────────────────────────────────────

def run_solver_pipeline(input_csv: str, output_dir: str, model_key: str) -> None:

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY)}")

    config   = MODEL_REGISTRY[model_key]
    safe_mdl = model_key.replace(".", "-").replace(":", "-")
    out_dir  = Path(output_dir) / f"solver_{safe_mdl}"
    out_dir.mkdir(parents=True, exist_ok=True)

    traces_path   = out_dir / "traces.csv"
    logprobs_path = out_dir / "logprobs.csv"

    log.info(f"Model     : {model_key} (provider={config['provider']})")
    log.info(f"Thinking  : {config['thinking']}")
    log.info(f"Efforts   : {config['effort_levels']}")
    log.info(f"k samples : {K_SAMPLES}")

    # Checkpoint resume
    completed:        set[tuple] = set()
    all_trace_rows:   list[dict] = []
    all_logprob_rows: list[dict] = []

    if traces_path.exists() and traces_path.stat().st_size > 0:
        try:
            existing   = pd.read_csv(traces_path)
            successful = (
                existing[existing["is_error"] == 0]
                if "is_error" in existing.columns else existing
            )
            completed = set(zip(
                successful["problem_id"],
                successful["model"],
                successful["effort_level"],
                successful["sample_idx"],
            ))
            all_trace_rows = existing.to_dict("records")
            n_errors = len(existing) - len(successful)
            log.info(
                f"Resuming — {len(completed)} successful samples "
                f"({n_errors} error rows queued for retry)"
            )
        except Exception as e:
            log.warning(f"Could not read existing traces ({e}) — starting fresh")

    if logprobs_path.exists() and logprobs_path.stat().st_size > 0:
        try:
            all_logprob_rows = pd.read_csv(logprobs_path).to_dict("records")
        except Exception:
            pass

    df    = pd.read_csv(input_csv)
    total = len(df)
    log.info(f"Problems  : {total} × {len(config['effort_levels'])} efforts × k={K_SAMPLES}")

    for idx, row in df.iterrows():
        pid     = str(row["problem_id"])
        problem = str(row["problem"])
        domain  = str(row["domain"])

        already = sum(
            1 for effort in config["effort_levels"] for i in range(K_SAMPLES)
            if (pid, model_key, effort, i) in completed
        )
        if already == len(config["effort_levels"]) * K_SAMPLES:
            log.info(f"[{pid}] all samples complete — skipping")
            continue

        log.info(f"=== [{idx+1}/{total}] {pid} ({domain}) ===")
        outputs, lp_summaries = solve_one_problem(pid, problem, model_key, out_dir)

        for out in outputs:
            if (out.problem_id, out.model, out.effort_level, out.sample_idx) in completed:
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
            completed.add((out.problem_id, out.model, out.effort_level, out.sample_idx))

        for lp in lp_summaries:
            lp["domain"] = domain
            all_logprob_rows.append(lp)

        pd.DataFrame(all_trace_rows).to_csv(traces_path,    index=False)
        pd.DataFrame(all_logprob_rows).to_csv(logprobs_path, index=False)
        log.info(f"[{pid}] saved — {idx+1}/{total} done")

    log.info(f"\nPipeline complete.")
    log.info(f"Traces   → {traces_path}  ({len(all_trace_rows)} rows)")
    log.info(f"Logprobs → {logprobs_path} ({len(all_logprob_rows)} rows)")
    print_run_summary(all_trace_rows, model_key)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Commercial model solver — metacognitive evaluation pipeline"
    )
    parser.add_argument("--model",      required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to run")
    parser.add_argument("--input_csv",  default="data/code_problems.csv",
                        help="Path to problems CSV (problem_id, problem, domain)")
    parser.add_argument("--output_dir", default="output",
                        help="Root output directory")
    args = parser.parse_args()

    run_solver_pipeline(
        input_csv  = args.input_csv,
        output_dir = args.output_dir,
        model_key  = args.model,
    )