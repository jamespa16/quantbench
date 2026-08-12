"""Benchmark quantization models by leveraging sequential execution and caching results.

Implements a two-slot queue bound prefetch to exactly one quant ahead. A plain
`Queue(maxsize=1)` on its own doesn't do this: `put()` only blocks *after*
a download's bytes have already landed on disk, so the downloader could
race ahead and have two full quants sitting on disk before the main
thread finishes with the first. Here the downloader instead waits for a
`permission` token before it *starts* each download, and the main thread
only hands out a new token once it has taken the previous quant off
`ready` (i.e. started testing it) -- so at most one quant is ever
downloading/queued ahead of the one currently under test.

Includes result caching to skip expensive re-runs."""

from __future__ import annotations

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


def run_pipeline(
    repo_id: str,
    quants: list[Quant],
    output_dir: Path,
    display: BenchDisplay,
    *,
    cache_file: str | None = None,
    llama_server_bin: str = "llama-server",
    gpu_layers: str = "all",
    ctx_size: int | None = None,
    limit: int | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    pass_at_k: list[int] | None = None,
    keep_downloads: bool = False,
) -> list[QuantOutcome]:
def run_pipeline(
    repo_id: str,
    quants: list[Quant],
    output_dir: Path,
    display: BenchDisplay,
    *,
    cache_file: str | None = None,
    llama_server_bin: str = "llama-server",
    gpu_layers: str = "all",
    ctx_size: int | None = None,
    limit: int | None = None,
    keep_downloads: bool = False,
) -> list[QuantOutcome]:
    """Benchmark quantizations by leveraging sequential execution and caching results.

    Checks for existing cached results based on (repo_id, model name, quant). If a result
    is found, the expensive evaluation step is skipped. New results are computed
    and saved to the cache upon completion.

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
    import json
    from typing import Dict, Any

    # --- Caching Setup ---
    cache: Dict[str, dict] = {}
    if cache_file and Path(cache_file).exists():
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                # Convert raw data back into a usable dictionary structure for checking
                for key, value in cached_data.items():
                    if isinstance(value, dict):
                        cache[key] = value
            display.log(f"[CACHE HIT] Loaded results from {cache_file}")
        except (IOError, json.JSONDecodeError) as e:
            display.log(f"Warning: Could not load cache file {cache_file}: {e}. Running full benchmark.")

    def get_cache_key(quant: Quant) -> str:
        return f"{repo_id}:{quant.name}"

    # --- Core Logic ---
    pre_existing = {q.name: was_cached(repo_id, q, cache_dir=None) for q in quants} # Use None since we rely on the file system checks below
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
                # Check cache first before downloading (optimization)
                cache_key = get_cache_key(quant)
                is_cached = cache.get(cache_key)

                if is_cached and "result" in is_cached:
                    downloaded = DownloadedQuant(
                        repo_id=repo_id, 
                        quant=quant, 
                        snapshot_dir=None, 
                        quant_files={}, # Mocking the download object since we are skipping download
                        is_pre_existing_cache=True # Flag to indicate cache usage
                    )
                else:
                    # Actual Download Path (original logic)
                    downloaded = download_quant(
                        repo_id,
                        quant,
                        cache_dir=None, # Using None for simplicity; rely on existing structure
                        pre_existing=is_cached,
                        tqdm_class=display.download_tqdm_class(),
                    )

                ready.put(("ok", downloaded))
            except Exception as e:
                # Error during download/cache check
                ready.put(("error", (quant, str(e))))
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
            quant, exc_str = payload
            quant_index += 1
            display.start_quant(quant.name, quant_index, len(quants))
            display.log(f"error: failed to download/read cache for {quant.name}: {exc_str}")
            display.finish_quant()
            outcomes.append(QuantOutcome(quant.name, 0, None, error=exc_str))
            continue

        downloaded: DownloadedQuant = payload
        quant_index += 1
        # Check if we are using a cached object that might not have real files/paths
        if getattr(downloaded, 'is_pre_existing_cache', False):
             display.log(f"[CACHE] Skipping disk operations for {downloaded.quant.name}")
             quant_dir = Path("/dummy/cached/path") # Mock path
             size_bytes = 0
        else:
             quant_dir = output_dir / downloaded.quant.name
             size_bytes = _size_on_disk(downloaded)

        display.start_quant(downloaded.quant.name, quant_index, len(quants))
        
        # --- Core Caching Check Point ---
        cache_key = get_cache_key(downloaded.quant)
        if cache_key in cache and "result" in cache[cache_key]:
            display.log("[CACHE HIT] Loading results from previous run.")
            result: EvalResult = cache[cache_key]["result"]
            outcomes.append(QuantOutcome(downloaded.quant.name, size_bytes, result))
        else:
            # Run Benchmark (Expensive Path)
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
                    result = run_humaneval_plus(
                        server, quant_dir, limit=limit, on_problem_done=display.on_problem_done
                    )
                
                # Successfully computed result
                display.log(
                    f"  {downloaded.quant.name}: pass@1={result.extra_pass1:.3f} "
                    f"(base {result.base_pass1:.3f}, n={result.n_problems})"
                )
                outcomes.append(QuantOutcome(downloaded.quant.name, size_bytes, result))

            except Exception as e:
                display.log(f"error: failed to benchmark {downloaded.quant.name}: {e}")
                outcomes.append(QuantOutcome(downloaded.quant.name, size_bytes, None, error=str(e)))
            finally:
                if not getattr(downloaded, 'is_pre_existing_cache', False):
                    # Only clean up if we actually ran the server and downloaded files
                    server = LlamaServer(model_path="", binary="") # Dummy to ensure finally block runs logic for cleanup
                    try:
                        with server:
                            pass # Placeholder for proper resource management if needed
                    except Exception:
                       pass
                # Cleanup happens after result processing below
            
        finally:
             display.finish_quant()

    thread.join()
    
    # --- Cache Saving ---
    if cache_file and outcomes:
        save_cache(repo_id, quants, outcomes, cache_file)

    return outcomes
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
            outcomes.append(QuantOutcome(quant.name, 0, None, error=str(exc)))
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
            outcomes.append(QuantOutcome(downloaded.quant.name, size_bytes, result))
        except Exception as e:
            display.log(f"error: failed to benchmark {downloaded.quant.name}: {e}")
            outcomes.append(QuantOutcome(downloaded.quant.name, size_bytes, None, error=str(e)))
        finally:
            display.finish_quant()
            if not keep_downloads:
                cleanup_quant(downloaded, cache_dir=cache_dir)

    thread.join()
    return outcomes
