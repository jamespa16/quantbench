# TODO

Improvements for quantbench, ordered by priority. **All items implemented and tested** (227 tests, ruff + mypy clean).

## High impact

- [x] **Persist throughput per quant** — `EvalResult` now carries `avg_tok_s` (total tokens / wall time of the generation phase) and `wall_time_s`; both flow through `summary.json` into `results.csv` (new `avg_tok_s`, `wall_time_s` columns) and the per-quant summary log line.
- [x] **Parallel generation** — `run_humaneval_plus(..., parallel=N)` fans out (task, sample) pairs over a `ThreadPoolExecutor`; `LlamaServer(parallel=N)` starts llama-server with `-np N`. `--parallel` CLI flag (default 1). `on_problem_done` fires when *all* of a task's samples are done, with a monotonic completed count. Output order stays task-major regardless of finish order.
- [x] **Resume bug: `--gpu-layers` missing from the run key** — `_run_key` now includes `gpu_layers` and `seed` (8 components: `limit:n_samples:temperature:pass_at_k:ctx_size:max_tokens:gpu_layers:seed`). `--parallel` is deliberately excluded (changes throughput only, never accuracy). Hashing `llama-server --version` into the key was considered and deferred: builds usually change results in small, explainable ways and it would force full re-benchmarks on any llama.cpp update.
- [x] **`generate()` robustness** —
  - Request timeout is scaled by the orchestrator: `max(600, max_tokens / 5)` (32000 tokens → ~21 min headroom at ~5 tok/s).
  - `generate()` retries transient failures (`requests.ConnectionError`, `requests.Timeout`, HTTP 5xx) up to `max_retries` (default 2, i.e. 3 total attempts) with linear backoff (2s, 4s). 4xx are not retried. `elapsed_s` covers the whole call including retries, so throughput accounting stays correct.

## Medium

- [x] **Use a paired significance test** — per-task scores are now keyed by task id (`dict[int, dict[str, float]]`), and `_pairwise_p_values` runs `scipy.stats.ttest_rel` on the *common* task ids of each pair. Pairs sharing < 2 tasks get `nan`. A pair whose per-task difference is constant (zero difference variance) is resolved analytically — `p=1.0` if no difference, `p=0.0` for a perfectly consistent offset — which also eliminates scipy's "catastrophic cancellation" `RuntimeWarning` (test suite now runs clean under `-W error::RuntimeWarning`).
- [x] **Warn on `--pass-at-k k>1` with `--temperature 0`** — stderr warning: greedy samples are identical, so pass@k equals pass@1.
- [x] **`--retry-failed` flag** — drops cached error outcomes before computing remaining quants, so previously-failed quants are re-run.
- [x] **Exit code hides partial failure** — `main()` returns 1 if *any* selected quant failed (success or cached error) and 0 only when all succeeded; a stderr warning names `--retry-failed`. CI can now gate on the exit code.

## Quick fixes

- [x] **README drift** — `--max-tokens` documented as `32000`; options table gained `--temperature`, `--seed`, `--parallel`, `--retry-failed`; new "Resuming and retrying" section covers the run key, retry semantics, and exit codes.
- [x] **Duplicate API call at startup** — `discover_quants_with_sizes()` makes a single `repo_info(..., files_metadata=True)` call returning both the quant list and the file sizes; `discover_quants` / `fetch_file_sizes` are thin wrappers kept for compatibility.
- [x] **Lazy-import matplotlib/scipy** — both are imported inside the functions that use them, so `--list-quants`, `--version`, and CSV-only paths never pay for them.
- [x] **`--seed` flag for reproducibility** — `--seed` (default 42) is passed through `LlamaServer.generate` into the chat-completion request body and included in the run key.
- [x] **`cleanup_quant` rescans the whole HF cache after every quant** — replaced the full-cache `scan_cache_dir` refcount with a repo-scoped refcount: cleanup now only walks the symlinks of *this* repo's snapshot dirs. (Trade-off, documented in `hf_cache.py`: a blob shared with *another* repo may be deleted even though the other repo still references it — the blob cache self-heals on re-download, and we never delete files we didn't download.)

## Bugs found along the way (fixed)

- `ax.bar(..., errhatches=None)` — `errhatches` is not a valid kwarg in matplotlib 3.11; the chart crashed for real users but tests never caught it because `plt` was fully mocked. Removed the kwarg and added unmocked render tests (`TestWriteChartReal`) that produce a real PNG.
- The old chart tests patched `quantbench.report.plt`, which no longer exists after the lazy import; replaced with a fixture that patches `sys.modules["matplotlib.pyplot"]` *and* the package attribute (both lookup paths `import ... as` uses) and surgically restores only that one key — `patch.dict("sys.modules", ...)` had been corrupting later tests by deleting unrelated modules (e.g. `numpy.fft._pocketfft`) added during the test.

## Verification

```
uv run pytest -q -W "error::RuntimeWarning"   # 227 passed
uv run ruff check .                            # All checks passed
uv run mypy .                                  # no issues in 14 source files (baseline had 6 pre-existing test errors, now 0)
```
