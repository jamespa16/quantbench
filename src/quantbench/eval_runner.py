"""Generate HumanEval+ completions against a running llama-server, then score with evalplus."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

from evalplus.data import get_human_eval_plus, write_jsonl
from evalplus.eval import PASS
from evalplus.evaluate import evaluate as evalplus_evaluate
from evalplus.sanitize import script as evalplus_sanitize

from quantbench.llama_server import LlamaServer

_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given the start of a Python "
    "function (signature and docstring). Reply with the complete function, including "
    "the given signature and docstring, correctly implemented. Put the code in a "
    "single ```python code block and say nothing else."
)
# Syntactically valid but always-failing: pads out --limit-skipped tasks so
# evalplus.evaluate()'s "every dataset problem must be present" assertion is
# satisfied without actually spending generation time on them.
_STUB_SUFFIX = "    raise NotImplementedError\n"

if platform.system() == "Darwin":
    # macOS's setrlimit(RLIMIT_AS, ...) can't actually lower the address-space
    # limit (fails with "current limit exceeds maximum limit" regardless of
    # the requested value), which crashes every sandboxed test execution
    # inside evalplus.eval.utils.reliability_guard. -1 disables that guard's
    # memory cap; unaffected on Linux, where RLIMIT_AS works normally.
    os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")


@dataclass(frozen=True)
class EvalResult:
    base_pass1: float
    extra_pass1: float
    n_problems: int


def run_humaneval_plus(server: LlamaServer, output_dir: Path, *, limit: int | None = None) -> EvalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = get_human_eval_plus()
    all_task_ids = list(problems)
    tested_task_ids = all_task_ids[:limit] if limit else all_task_ids
    skipped_task_ids = all_task_ids[limit:] if limit else []

    samples = [
        {"task_id": task_id, "solution": server.generate(problems[task_id]["prompt"], system=_SYSTEM_PROMPT)}
        for task_id in tested_task_ids
    ]
    samples += [
        {"task_id": task_id, "solution": problems[task_id]["prompt"] + _STUB_SUFFIX}
        for task_id in skipped_task_ids
    ]

    samples_path = output_dir / "samples.jsonl"
    write_jsonl(str(samples_path), samples)

    evalplus_sanitize(str(samples_path))
    sanitized_path = output_dir / "samples-sanitized.jsonl"
    results_path = output_dir / "samples-sanitized_eval_results.json"
    results_path.unlink(missing_ok=True)  # evalplus prompts interactively if this already exists

    evalplus_evaluate(dataset="humaneval", samples=str(sanitized_path), i_just_wanna_run=True)
    with open(results_path) as f:
        results = json.load(f)["eval"]

    total = 0
    base_correct = 0
    extra_correct = 0
    for task_id in tested_task_ids:
        if task_id not in results:
            continue  # sanitize found no syntactically valid code for this task
        run = results[task_id][0]
        total += 1
        base_ok = run["base_status"] == PASS
        base_correct += base_ok
        extra_correct += base_ok and run["plus_status"] == PASS

    return EvalResult(
        base_pass1=base_correct / total if total else 0.0,
        extra_pass1=extra_correct / total if total else 0.0,
        n_problems=total,
    )
