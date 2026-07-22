"""Generate HumanEval+ completions against a running llama-server, then score with evalplus."""

from __future__ import annotations

import json
import math
import os
import platform
from collections.abc import Callable
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
class PassAtKStats:
    mean: float
    std_dev: float
    std_error: float


@dataclass(frozen=True)
class EvalResult:
    base_pass1: float
    extra_pass1: float
    n_problems: int
    pass_at_k: dict[int, float] | None = None
    pass_at_k_stats: dict[int, PassAtKStats] | None = None
    pass_at_k_per_task: dict[int, list[float]] | None = None


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k: probability at least 1 of k random samples is correct.

    Uses the formula from Chen et al. (2021): 1 - (n-c choose k) / (n choose k).
    """
    if n - c >= k:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)
    else:
        return 1.0


def run_humaneval_plus(
    server: LlamaServer,
    output_dir: Path,
    *,
    limit: int | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    pass_at_k: list[int] | None = None,
    on_problem_done: Callable[[int, int, int, float], None] | None = None,
) -> EvalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = get_human_eval_plus()
    all_task_ids = list(problems)
    tested_task_ids = all_task_ids[:limit] if limit else all_task_ids
    skipped_task_ids = all_task_ids[limit:] if limit else []

    samples = []
    for i, task_id in enumerate(tested_task_ids, start=1):
        task_total_tokens = 0
        task_total_time = 0.0
        for s in range(n_samples):
            result = server.generate(problems[task_id]["prompt"], system=_SYSTEM_PROMPT, temperature=temperature)
            samples.append({"task_id": task_id, "solution": result.text})
            task_total_tokens += result.completion_tokens
            task_total_time += result.elapsed_s
        if on_problem_done:
            on_problem_done(i, len(tested_task_ids), task_total_tokens, task_total_time)
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

    # Count correct samples per task (extra/tests+ suite for pass@k).
    task_extra_correct: dict[str, int] = {}
    task_base_correct: dict[str, int] = {}
    task_n_samples: dict[str, int] = {}
    total_tasks = 0
    for task_id in tested_task_ids:
        if task_id not in results:
            continue  # sanitize found no syntactically valid code for this task
        total_tasks += 1
        entries = results[task_id]
        task_n_samples[task_id] = len(entries)
        task_extra_correct[task_id] = sum(
            1 for e in entries if e["base_status"] == PASS and e["plus_status"] == PASS
        )
        task_base_correct[task_id] = sum(
            1 for e in entries if e["base_status"] == PASS
        )

    base_pass1 = sum(task_base_correct.values()) / (total_tasks * n_samples) if total_tasks else 0.0
    extra_pass1 = sum(task_extra_correct.values()) / (total_tasks * n_samples) if total_tasks else 0.0

    # Compute pass@k (extra/tests+ suite) with per-task scores + stats.
    pass_at_k_map: dict[int, float] | None = None
    pass_at_k_stats_map: dict[int, PassAtKStats] | None = None
    pass_at_k_per_task_map: dict[int, list[float]] | None = None
    if pass_at_k:
        pass_at_k_map = {}
        pass_at_k_stats_map = {}
        pass_at_k_per_task_map = {}
        for k in pass_at_k:
            per_task: list[float] = []
            for task_id in task_extra_correct:
                n = task_n_samples[task_id]
                c = task_extra_correct[task_id]
                per_task.append(_pass_at_k(n, c, k))
            pass_at_k_per_task_map[k] = per_task
            mean_val = sum(per_task) / len(per_task) if per_task else 0.0
            n_tasks = len(per_task)
            if n_tasks >= 2:
                variance = sum((s - mean_val) ** 2 for s in per_task) / (n_tasks - 1)
                std_dev = math.sqrt(variance)
            else:
                std_dev = 0.0
            std_error = std_dev / math.sqrt(n_tasks) if n_tasks else 0.0
            pass_at_k_map[k] = mean_val
            pass_at_k_stats_map[k] = PassAtKStats(mean=mean_val, std_dev=std_dev, std_error=std_error)

    return EvalResult(
        base_pass1=base_pass1,
        extra_pass1=extra_pass1,
        n_problems=total_tasks,
        pass_at_k=pass_at_k_map,
        pass_at_k_stats=pass_at_k_stats_map,
        pass_at_k_per_task=pass_at_k_per_task_map,
    )
