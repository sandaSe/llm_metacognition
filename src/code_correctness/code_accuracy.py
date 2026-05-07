"""
code_accuracy.py
----------------
Evaluates generated Python solutions against held-out test cases.

For each solution file in the solutions directory, runs all matching
input/output test cases and records pass/fail per test case.

Expected directory layout:
    data/tests/<problem_id>/input1.txt
    data/tests/<problem_id>/output1.txt
    ...

Input:
    --sol_dir    Directory of generated .py solution files
                 (default: output/solver/solutions)
    --test_dir   Root directory of test cases
                 (default: data/tests)
    --output_csv Path for the results CSV
                 (default: output/code_correctness/results.csv)

Output:
    results.csv — one row per (solution × test case)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_solutions(
    sol_dir:    Path,
    test_dir:   Path,
    output_csv: Path,
) -> None:
    rows: list[dict] = []

    for script in sorted(sol_dir.glob("*.py")):
        filename = script.stem           # e.g. P01_effort-think_sample-00

        # Extract problem_id and sample_id from filename
        parts      = filename.split("_")
        problem_id = parts[0]            # e.g. P01
        sample_id  = parts[-1]           # e.g. sample-00

        test_folder = test_dir / problem_id

        if not test_folder.exists():
            print(f"No tests found for {problem_id} (script: {script.name})")
            continue

        input_files = sorted(test_folder.glob("input*.txt"))
        if not input_files:
            print(f"No input files found in {test_folder}")
            continue

        pass_count      = 0
        total_tests     = 0
        testcase_results: list[dict] = []

        for input_file in input_files:
            test_num    = input_file.stem.replace("input", "")
            output_file = test_folder / f"output{test_num}.txt"

            if not output_file.exists():
                continue

            total_tests += 1
            test_input  = input_file.read_text(encoding="utf-8")
            expected    = output_file.read_text(encoding="utf-8").strip()

            try:
                result = subprocess.run(
                    [sys.executable, str(script)],
                    input=test_input,
                    text=True,
                    capture_output=True,
                    timeout=5,
                )
                actual = result.stdout.strip()
                stderr = result.stderr.strip()
                status = "PASS" if actual == expected else "FAIL"
                if status == "PASS":
                    pass_count += 1

            except subprocess.TimeoutExpired:
                actual = ""
                stderr = "Timeout (exceeded 5 s)"
                status = "ERROR"
            except Exception as e:
                actual = ""
                stderr = str(e)
                status = "ERROR"

            testcase_results.append({
                "problem_id":      problem_id,
                "sample_id":       sample_id,
                "solution_file":   script.name,
                "testcase":        test_num,
                "status":          status,
                "expected_output": expected,
                "actual_output":   actual,
                "stderr":          stderr,
            })

        pass_ratio     = pass_count / total_tests if total_tests > 0 else 0.0
        overall_result = "CORRECT" if pass_ratio >= 1.0 else "WRONG"

        for row in testcase_results:
            row["passed_count"]   = pass_count
            row["total_tests"]    = total_tests
            row["overall_result"] = overall_result
            rows.append(row)

        print(f"{script.name}: {pass_count}/{total_tests} → {overall_result}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "problem_id", "sample_id", "solution_file", "testcase",
                    "status", "expected_output", "actual_output", "stderr",
                    "passed_count", "total_tests", "overall_result",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved results → {output_csv}  ({len(rows)} rows)")
    else:
        print("\nNo results generated. Check your folder structures.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated solutions against held-out test cases"
    )
    parser.add_argument("--sol_dir",    default="output/solver/solutions",
                        help="Directory of generated .py solution files")
    parser.add_argument("--test_dir",   default="data/tests",
                        help="Root directory of test cases")
    parser.add_argument("--output_csv", default="output/code_correctness/results.csv",
                        help="Path for results CSV")
    args = parser.parse_args()

    evaluate_solutions(
        sol_dir    = Path(args.sol_dir),
        test_dir   = Path(args.test_dir),
        output_csv = Path(args.output_csv),
    )