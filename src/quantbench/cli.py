"""quantbench CLI: benchmark every quantization in a Hugging Face GGUF repo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from quantbench import __version__
from quantbench.discovery import discover_quants
from quantbench.eval_runner import EvalResult, PassAtKStats
from quantbench.orchestrator import QuantOutcome, run_pipeline
from quantbench.report import write_chart, write_csv
from quantbench.ui import create_display, print_banner


def _load_outcome_from_summary(path: Path, *, run_key: str | None = None) -> QuantOutcome | None:
    try:
        data = json.loads((path / "summary.json").read_text())
    except Exception:
        return None
    # Skip stale summaries: if the caller provides a run key and the cached
    # one doesn't match, the benchmark parameters changed and the result is invalid.
    if run_key is not None and data.get("run_key") != run_key:
        return None
    quant_name = data.get("quant_name")
    size_bytes = data.get("size_bytes", 0)
    error = data.get("error")
    res_data = data.get("result")
    result = None
    if res_data:
        pass_at_k_stats = None
        if res_data.get("pass_at_k_stats"):
            pass_at_k_stats = {
                int(k): PassAtKStats(
                    mean=v["mean"],
                    std_dev=v["std_dev"],
                    std_error=v["std_error"],
                )
                for k, v in res_data["pass_at_k_stats"].items()
            }
        # pass_at_k may be dict with string keys from json
        pass_at_k = None
        if res_data.get("pass_at_k"):
            pass_at_k = {int(k): v for k, v in res_data["pass_at_k"].items()}
        # pass_at_k_per_task: {k: [per-task scores]}, keys are strings from json
        pass_at_k_per_task = None
        raw_per_task = res_data.get("pass_at_k_per_task")
        if raw_per_task:
            pass_at_k_per_task = {int(k): v for k, v in raw_per_task.items()}
        result = EvalResult(
            base_pass1=res_data["base_pass1"],
            extra_pass1=res_data["extra_pass1"],
            n_problems=res_data["n_problems"],
            pass_at_k=pass_at_k,
            pass_at_k_stats=pass_at_k_stats,
            pass_at_k_per_task=pass_at_k_per_task,
        )
    return QuantOutcome(quant_name, size_bytes, result, error)


def _load_existing_outcomes(output_dir: Path, *, run_key: str | None = None) -> dict[str, QuantOutcome]:
    outcomes: dict[str, QuantOutcome] = {}
    if not output_dir.exists():
        return outcomes
    for quant_dir in output_dir.iterdir():
        if not quant_dir.is_dir():
            continue
        summary_path = quant_dir / "summary.json"
        if summary_path.exists():
            outcome = _load_outcome_from_summary(quant_dir, run_key=run_key)
            if outcome:
                outcomes[outcome.quant_name] = outcome
    return outcomes


def _run_key(args: argparse.Namespace) -> str:
    """Deterministic key from parameters that affect benchmark results."""
    return f"{args.limit}:{args.n_samples}:{args.temperature}:{args.pass_at_k}:{args.ctx_size}:{args.max_tokens}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quantbench",
        description="Benchmark every GGUF quantization in a Hugging Face repo on HumanEval+.",
    )
    parser.add_argument("repo_id", help="Hugging Face repo, e.g. unsloth/Qwen3.5-0.8B-GGUF")
    parser.add_argument(
        "--quants", help="Comma-separated subset of quant names to run (default: all discovered)"
    )
    parser.add_argument(
        "--quants-except", help="Comma-separated quant names to exclude (default: none)"
    )
    parser.add_argument(
        "--list-quants",
        action="store_true",
        help="Print discovered quants and exit without downloading anything",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N HumanEval+ problems (for a quick smoke test)"
    )
    parser.add_argument(
        "--ctx-size", type=int, default=None, help="llama-server --ctx-size override (default: model's own metadata)"
    )
    parser.add_argument("--gpu-layers", default="all", help="llama-server -ngl value (default: all)")
    parser.add_argument("--llama-server-bin", default="llama-server", help="Path to the llama-server binary")
    parser.add_argument("--cache-dir", default=None, help="Override the Hugging Face cache directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write per-quant logs/samples and the final report (default: ./results/<repo_slug>)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1, help="Number of completions to generate per problem (default: 1)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32000, help="Maximum tokens per generation (default: 32000)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 for greedy)"
    )
    parser.add_argument(
        "--pass-at-k",
        default=None,
        help="Comma-separated k values for pass@k reporting, e.g. 1,10,100 (default: 1)",
    )
    parser.add_argument(
        "--keep-downloads", action="store_true", help="Don't delete quants that this run downloaded"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    console = Console()
    print_banner(console, subtitle="benchmark GGUF quantizations on HumanEval+")

    if args.quants and args.quants_except:
        print("error: --quants and --quants-except are mutually exclusive", file=sys.stderr)
        return 1

    if args.n_samples < 1:
        print("error: --n-samples must be >= 1", file=sys.stderr)
        return 1

    pass_at_k: list[int] | None = None
    if args.pass_at_k:
        try:
            pass_at_k = [int(x.strip()) for x in args.pass_at_k.split(",")]
        except ValueError:
            print("error: --pass-at-k must be comma-separated integers", file=sys.stderr)
            return 1
        if any(k < 1 for k in pass_at_k):
            print("error: --pass-at-k values must be >= 1", file=sys.stderr)
            return 1
        for k in pass_at_k:
            if k > args.n_samples:
                print(f"error: pass@k={k} requires --n-samples >= {k} (got {args.n_samples})", file=sys.stderr)
                return 1

    print(f"discovering quants in {args.repo_id}...")
    quants = discover_quants(args.repo_id)
    if not quants:
        print("error: no GGUF quants found in this repo", file=sys.stderr)
        return 1

    if args.quants:
        wanted = {name.strip() for name in args.quants.split(",")}
        selected = [q for q in quants if q.name in wanted]
        missing = wanted - {q.name for q in selected}
        if missing:
            print(f"error: unknown quant(s): {', '.join(sorted(missing))}", file=sys.stderr)
            print(f"available: {', '.join(q.name for q in quants)}", file=sys.stderr)
            return 1
        quants = selected
    elif args.quants_except:
        excluded = {name.strip() for name in args.quants_except.split(",")}
        known = {q.name for q in quants}
        missing = excluded - known
        if missing:
            print(f"error: unknown quant(s): {', '.join(sorted(missing))}", file=sys.stderr)
            print(f"available: {', '.join(q.name for q in quants)}", file=sys.stderr)
            return 1
        quants = [q for q in quants if q.name not in excluded]

    if args.list_quants:
        for q in quants:
            print(f"{q.name}\t{', '.join(q.files)}")
        return 0

    repo_slug = args.repo_id.replace("/", "__")
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / repo_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load any previously completed quants so we can resume
    current_run_key = _run_key(args)
    existing_outcomes = _load_existing_outcomes(output_dir, run_key=current_run_key)
    existing_names = set(existing_outcomes.keys())
    # Filter out quants already completed
    remaining_quants = [q for q in quants if q.name not in existing_names]
    if existing_names:
        print(f"resuming: {len(existing_names)} quant(s) already completed, {len(remaining_quants)} remaining")

    with create_display(console) as display:
        new_outcomes = run_pipeline(
            args.repo_id,
            remaining_quants,
            output_dir,
            display,
            run_key=current_run_key,
            cache_dir=args.cache_dir,
            llama_server_bin=args.llama_server_bin,
            gpu_layers=args.gpu_layers,
            ctx_size=args.ctx_size,
            limit=args.limit,
            n_samples=args.n_samples,
            temperature=args.temperature,
            pass_at_k=pass_at_k,
            keep_downloads=args.keep_downloads,
            max_tokens=args.max_tokens,
        )

    # Merge existing and new outcomes, preserving order
    all_outcomes = list(existing_outcomes.values()) + new_outcomes
    # Ensure we have all quants in output (even those not yet run) for completeness
    # Build a dict for quick lookup
    outcome_map = {o.quant_name: o for o in all_outcomes}
    # Preserve original quant order
    ordered_outcomes = [outcome_map[q.name] for q in quants if q.name in outcome_map]

    write_csv(ordered_outcomes, output_dir / "results.csv")
    write_chart(ordered_outcomes, output_dir / "results.png", repo_id=args.repo_id)

    n_ok = sum(1 for o in ordered_outcomes if o.result is not None)
    print(f"\ndone: {n_ok}/{len(ordered_outcomes)} quants benchmarked successfully")
    print(f"results: {output_dir / 'results.csv'}")
    print(f"chart:   {output_dir / 'results.png'}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
