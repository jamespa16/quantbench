"""Generate HumanEval+ completions against a running llama-server, then score with evalplus."""

from __future__ import annotations

import json
import math
import os
import platform
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evalplus.data import get_human_eval_plus, write_jsonl
from evalplus.eval import PASS
from evalplus.evaluate import evaluate as evalplus_evaluate
from evalplus.sanitize import script as evalplus_sanitize

from quantbench.llama_server import GenerateResult, LlamaServer

_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given the start of a Python "
    "function (signature and docstring). Reply with the complete function, including "
    "the given signature and docstring, correctly implemented. Put the code in a "
    "single ```python code block and say nothing else."
)
# Syntactically valid but always-failing: pads out --limit-skipped tasks so
# evalplus.evaluate()'s "every dataset problem must be present" assertion is
# satisfied without actually spending generation time on them.
# STUB: skipped by --limit
_STUB_SUFFIX = "    # STUB: skipped by --limit\n    raise NotImplementedError\n"


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
    # k -> {task_id -> pass@k score for that task}. Keyed by task id (not a
    # bare list) so different quants can be compared on the *common* tasks
    # even when sanitize drops a task for one of them.
    pass_at_k_per_task: dict[int, dict[str, float]] | None = None
    # Aggregate generation throughput (total tokens / wall time) and total
    # wall time of the generation phase. With --parallel > 1 this is
    # aggregate throughput across concurrent requests, not per-stream speed.
    avg_tok_s: float | None = None
    wall_time_s: float | None = None


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k: probability at least 1 of k random samples is correct.

    Uses the formula from Chen et al. (2021): 1 - (n-c choose k) / (n choose k).
    """
    if n - c >= k:
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)
    else:
        return 1.0


class _GeneratesCompletions(Protocol):
    """Structural server type for generation, so tests can pass fakes
    without constructing a LlamaServer (which spawns a real process)."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> GenerateResult: ...


def _generate_samples(
    server: _GeneratesCompletions,
    problems: dict,
    tested_task_ids: list[str],
    *,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    parallel: int,
    seed: int | None,
    on_problem_done: Callable[[int, int, int, float], None] | None = None,
) -> tuple[list[dict], float, int]:
    """Generate every (task, sample) pair, optionally in parallel.

    Returns (samples in task-major order, wall_time_s, total_completion_tokens).
    `on_problem_done(completed, total, task_tokens, task_time)` fires when all
    of a task's samples are done; `completed` is the count of problems finished
    so far (not the problem's position), so it is monotonic even when problems
    finish out of order under parallelism.
    """
    task_tokens: dict[str, int] = dict.fromkeys(tested_task_ids, 0)
    task_time: dict[str, float] = dict.fromkeys(tested_task_ids, 0.0)
    pending: dict[str, int] = dict.fromkeys(tested_task_ids, n_samples)
    results: dict[tuple[str, int], dict] = {}
    total_tokens = 0
    completed = 0
    lock = threading.Lock()  # only contended when parallel > 1

    def work(task_id: str, sample_idx: int) -> None:
        nonlocal total_tokens, completed
        result = server.generate(
            problems[task_id]["prompt"],
            system=_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        results[(task_id, sample_idx)] = {"task_id": task_id, "solution": result.text}
        with lock:
            task_tokens[task_id] += result.completion_tokens
            task_time[task_id] += result.elapsed_s
            total_tokens += result.completion_tokens
            pending[task_id] -= 1
            if pending[task_id] == 0:
                completed += 1
                if on_problem_done is not None:
                    on_problem_done(completed, len(tested_task_ids), task_tokens[task_id], task_time[task_id])

    start = time.monotonic()
    if parallel <= 1:
        for task_id in tested_task_ids:
            for s in range(n_samples):
                work(task_id, s)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [
                pool.submit(work, task_id, s)
                for task_id in tested_task_ids
                for s in range(n_samples)
            ]
            for future in as_completed(futures):
                future.result()  # propagate generation errors to the caller
    wall_time = time.monotonic() - start

    samples = [results[(tid, s)] for tid in tested_task_ids for s in range(n_samples)]
    return samples, wall_time, total_tokens


def run_humaneval_plus(
    server: LlamaServer,
    output_dir: Path,
    *,
    limit: int | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    pass_at_k: list[int] | None = None,
    on_problem_done: Callable[[int, int, int, float], None] | None = None,
    max_tokens: int = 32000,
    parallel: int = 1,
    seed: int | None = None,
) -> EvalResult:
    if platform.system() == "Darwin":
        # macOS's setrlimit(RLIMIT_AS, ...) can't actually lower the address-space
        # limit (fails with "current limit exceeds maximum limit" regardless of
        # the requested value), which crashes every sandboxed test execution
        # inside evalplus.eval.utils.reliability_guard. -1 disables that guard's
        # memory cap; unaffected on Linux, where RLIMIT_AS works normally.
        os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = get_human_eval_plus()
    all_task_ids = list(problems)
    tested_task_ids = all_task_ids[:limit] if limit else all_task_ids
    skipped_task_ids = all_task_ids[limit:] if limit else []

    samples, wall_time, total_tokens = _generate_samples(
        server,
        problems,
        tested_task_ids,
        n_samples=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        parallel=parallel,
        seed=seed,
        on_problem_done=on_problem_done,
    )
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
    pass_at_k_per_task_map: dict[int, dict[str, float]] | None = None
    if pass_at_k:
        pass_at_k_map = {}
        pass_at_k_stats_map = {}
        pass_at_k_per_task_map = {}
        for k in pass_at_k:
            per_task: dict[str, float] = {}
            for task_id in task_extra_correct:
                per_task[task_id] = _pass_at_k(
                    task_n_samples[task_id], task_extra_correct[task_id], k
                )
            pass_at_k_per_task_map[k] = per_task
            scores = list(per_task.values())
            mean_val = sum(scores) / len(scores) if scores else 0.0
            n_tasks = len(scores)
            if n_tasks >= 2:
                variance = sum((s - mean_val) ** 2 for s in scores) / (n_tasks - 1)
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
        avg_tok_s=total_tokens / wall_time if wall_time > 0 else 0.0,
        wall_time_s=wall_time,
    )
