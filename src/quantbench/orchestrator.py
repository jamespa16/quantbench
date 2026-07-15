"""Overlap downloading the next quant with strictly-serialized benchmarking."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from quantbench.discovery import Quant
from quantbench.eval_runner import EvalResult, run_humaneval_plus
from quantbench.hf_cache import DownloadedQuant, cleanup_quant, download_quant, was_cached
from quantbench.llama_server import LlamaServer

_DONE = object()


@dataclass(frozen=True)
class QuantOutcome:
    quant_name: str
    size_bytes: int
    result: EvalResult | None
    error: str | None = None


def _size_on_disk(downloaded: DownloadedQuant) -> int:
    return sum((Path(downloaded.snapshot_dir) / f).stat().st_size for f in downloaded.quant.files)


def run_pipeline(
    repo_id: str,
    quants: list[Quant],
    output_dir: Path,
    *,
    cache_dir: str | None = None,
    llama_server_bin: str = "llama-server",
    gpu_layers: str = "all",
    ctx_size: int | None = None,
    limit: int | None = None,
    keep_downloads: bool = False,
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

    ready: queue.Queue = queue.Queue(maxsize=1)
    permission: queue.Queue = queue.Queue(maxsize=1)
    permission.put(None)  # let the first download start immediately

    def downloader() -> None:
        for quant in quants:
            permission.get()
            try:
                downloaded = download_quant(
                    repo_id, quant, cache_dir=cache_dir, pre_existing=pre_existing[quant.name]
                )
                ready.put(("ok", downloaded))
            except Exception as e:
                ready.put(("error", (quant, e)))
        ready.put(_DONE)

    thread = threading.Thread(target=downloader, daemon=True)
    thread.start()

    outcomes: list[QuantOutcome] = []
    while True:
        item = ready.get()
        if item is _DONE:
            break
        permission.put(None)  # release the downloader to prefetch the next quant

        status, payload = item
        if status == "error":
            quant, exc = payload
            print(f"error: failed to download {quant.name}: {exc}")
            outcomes.append(QuantOutcome(quant.name, 0, None, error=str(exc)))
            continue

        downloaded: DownloadedQuant = payload
        quant_dir = output_dir / downloaded.quant.name
        size_bytes = _size_on_disk(downloaded)
        print(f"benchmarking {downloaded.quant.name} ({size_bytes / 1e9:.2f} GB)...")
        try:
            server = LlamaServer(
                model_path=downloaded.model_path,
                binary=llama_server_bin,
                gpu_layers=gpu_layers,
                ctx_size=ctx_size,
                log_path=str(quant_dir / "llama-server.log"),
            )
            with server:
                result = run_humaneval_plus(server, quant_dir, limit=limit)
            print(
                f"  {downloaded.quant.name}: pass@1={result.extra_pass1:.3f} "
                f"(base {result.base_pass1:.3f}, n={result.n_problems})"
            )
            outcomes.append(QuantOutcome(downloaded.quant.name, size_bytes, result))
        except Exception as e:
            print(f"error: failed to benchmark {downloaded.quant.name}: {e}")
            outcomes.append(QuantOutcome(downloaded.quant.name, size_bytes, None, error=str(e)))
        finally:
            if not keep_downloads:
                cleanup_quant(downloaded, cache_dir=cache_dir)

    thread.join()
    return outcomes
