"""Overlap downloading the next quant with strictly-serialized benchmarking."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from quantbench.discovery import Quant, fetch_file_sizes
from quantbench.eval_runner import EvalResult, run_humaneval_plus
from quantbench.hf_cache import DownloadedQuant, cleanup_quant, download_quant, was_cached
from quantbench.llama_server import LlamaServer
from quantbench.ui import BenchDisplay

_DONE = object()


@dataclass(frozen=True)
class QuantOutcome:
    quant_name: str
    size_bytes: int
    result: EvalResult | None
    error: str | None = None


def _size_on_disk(downloaded: DownloadedQuant) -> int:
    return sum((Path(downloaded.snapshot_dir) / f).stat().st_size for f in downloaded.quant.files)


def _write_summary(quant_dir: Path, outcome: QuantOutcome, *, run_key: str) -> None:
    data = {
        "quant_name": outcome.quant_name,
        "size_bytes": outcome.size_bytes,
        "error": outcome.error,
        "run_key": run_key,
    }
    if outcome.result:
        res = outcome.result
        data["result"] = {
            "base_pass1": res.base_pass1,
            "extra_pass1": res.extra_pass1,
            "n_problems": res.n_problems,
            "pass_at_k": res.pass_at_k,
            "pass_at_k_stats": (
                {k: {"mean": v.mean, "std_dev": v.std_dev, "std_error": v.std_error}
                 for k, v in res.pass_at_k_stats.items()}
                if res.pass_at_k_stats else None
            ),
            "pass_at_k_per_task": res.pass_at_k_per_task,
        }
    else:
        data["result"] = None
    quant_dir.mkdir(parents=True, exist_ok=True)
    (quant_dir / "summary.json").write_text(json.dumps(data))


def run_pipeline(
    repo_id: str,
    quants: list[Quant],
    output_dir: Path,
    display: BenchDisplay,
    *,
    run_key: str = "",
    cache_dir: str | None = None,
    llama_server_bin: str = "llama-server",
    gpu_layers: str = "all",
    ctx_size: int | None = None,
    limit: int | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    pass_at_k: list[int] | None = None,
    keep_downloads: bool = False,
    max_tokens: int = 32000,
) -> list[QuantOutcome]:
    """Benchmark each quant in order, downloading quant N+1 while quant N is tested.

    Two single-slot queues bound prefetch to exactly one quant ahead. A plain
    `Queue(maxsize=1)` on its own doesn't do this: `put()` only blocks *after*
    a download's bytes have already landed on disk, so the downloader could
    race ahead and have two full quants sitting on disk before the main
    thread finishes with the first. Here the downloader instead waits for a
    `permission` token before it *starts* each download, and the main thread
    only hands out a new token once it has taken the previous quant off
    `ready` (i.e. started testing it) -- so at most one quant is ever
    downloading/queued ahead of the one currently under test.
    """
    # Must be captured before any download in this run touches the cache.
    pre_existing = {q.name: was_cached(repo_id, q, cache_dir=cache_dir) for q in quants}
    file_sizes = fetch_file_sizes(repo_id)

    ready: queue.Queue = queue.Queue(maxsize=1)
    permission: queue.Queue = queue.Queue(maxsize=1)
    permission.put(None)  # let the first download start immediately

    def downloader() -> None:
        for quant in quants:
            permission.get()
            total_bytes = sum(file_sizes.get(f, 0) for f in quant.files)
            display.start_download(quant.name, total_bytes or None)
            try:
                downloaded = download_quant(
                    repo_id,
                    quant,
                    cache_dir=cache_dir,
                    pre_existing=pre_existing[quant.name],
                    tqdm_class=display.download_tqdm_class(),
                )
                ready.put(("ok", downloaded))
            except Exception as e:
                ready.put(("error", (quant, e)))
        ready.put(_DONE)

    thread = threading.Thread(target=downloader, daemon=True)
    thread.start()

    outcomes: list[QuantOutcome] = []
    quant_index = 0
    while True:
        item = ready.get()
        if item is _DONE:
            break
        permission.put(None)  # release the downloader to prefetch the next quant

        status, payload = item
        if status == "error":
            quant, exc = payload
            quant_index += 1
            display.start_quant(quant.name, quant_index, len(quants))
            display.log(f"error: failed to download {quant.name}: {exc}")
            display.finish_quant()
            error_outcome = QuantOutcome(quant.name, 0, None, error=str(exc))
            outcomes.append(error_outcome)
            # Cache the download error so it's not retried on resume.
            quant_dir = output_dir / quant.name
            _write_summary(quant_dir, error_outcome, run_key=run_key)
            continue

        downloaded: DownloadedQuant = payload
        quant_index += 1
        quant_dir = output_dir / downloaded.quant.name
        size_bytes = _size_on_disk(downloaded)
        display.start_quant(downloaded.quant.name, quant_index, len(quants))
        display.log(f"benchmarking {downloaded.quant.name} ({size_bytes / 1e9:.2f} GB)...")
        try:
            server = LlamaServer(
                model_path=downloaded.model_path,
                binary=llama_server_bin,
                gpu_layers=gpu_layers,
                ctx_size=ctx_size,
                log_path=str(quant_dir / "llama-server.log"),
            )
            with server:
                display.start_eval(downloaded.quant.name)
                result = run_humaneval_plus(
                    server, quant_dir,
                    limit=limit,
                    n_samples=n_samples,
                    temperature=temperature,
                    pass_at_k=pass_at_k,
                    on_problem_done=display.on_problem_done,
                    max_tokens=max_tokens,
                )
            pass1_str = (
                f"pass@1={result.extra_pass1:.3f} "
                f"(base {result.base_pass1:.3f}, n={result.n_problems})"
            )
            passk_str = ""
            if result.pass_at_k and result.pass_at_k_stats:
                parts = []
                for k in sorted(result.pass_at_k):
                    st = result.pass_at_k_stats[k]
                    parts.append(f"pass@{k}={st.mean:.3f} ±{st.std_error:.3f}")
                passk_str = f" [{', '.join(parts)}]"
            display.log(f"  {downloaded.quant.name}: {pass1_str}{passk_str}")
            outcome = QuantOutcome(downloaded.quant.name, size_bytes, result)
            outcomes.append(outcome)
            _write_summary(quant_dir, outcome, run_key=run_key)
        except Exception as e:
            display.log(f"error: failed to benchmark {downloaded.quant.name}: {e}")
            outcome = QuantOutcome(downloaded.quant.name, size_bytes, None, error=str(e))
            outcomes.append(outcome)
            _write_summary(quant_dir, outcome, run_key=run_key)
        finally:
            display.finish_quant()
            if not keep_downloads:
                cleanup_quant(downloaded, cache_dir=cache_dir)

    thread.join()
    return outcomes
