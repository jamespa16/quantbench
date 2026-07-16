"""quantbench CLI: benchmark every quantization in a Hugging Face GGUF repo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from quantbench.discovery import discover_quants
from quantbench.orchestrator import run_pipeline
from quantbench.report import write_chart, write_csv
from quantbench.ui import create_display, print_banner


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
        "--keep-downloads", action="store_true", help="Don't delete quants that this run downloaded"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    console = Console()
    print_banner(console, subtitle="benchmark GGUF quantizations on HumanEval+")

    if args.quants and args.quants_except:
        print("error: --quants and --quants-except are mutually exclusive", file=sys.stderr)
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

    with create_display(console) as display:
        outcomes = run_pipeline(
            args.repo_id,
            quants,
            output_dir,
            display,
            cache_dir=args.cache_dir,
            llama_server_bin=args.llama_server_bin,
            gpu_layers=args.gpu_layers,
            ctx_size=args.ctx_size,
            limit=args.limit,
            keep_downloads=args.keep_downloads,
        )

    write_csv(outcomes, output_dir / "results.csv")
    write_chart(outcomes, output_dir / "results.png", repo_id=args.repo_id)

    n_ok = sum(1 for o in outcomes if o.result is not None)
    print(f"\ndone: {n_ok}/{len(outcomes)} quants benchmarked successfully")
    print(f"results: {output_dir / 'results.csv'}")
    print(f"chart:   {output_dir / 'results.png'}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
