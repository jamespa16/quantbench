# quantbench

Benchmark every GGUF quantization in a Hugging Face repo (e.g.
`unsloth/Qwen3.5-0.8B-GGUF`) against the HumanEval+ coding benchmark, and
produce a CSV + bar chart comparing them.

For each quant it discovers, quantbench downloads the GGUF (if not already
in your Hugging Face cache), spins up `llama-server`, runs pass@1 greedy
generation against all 164 HumanEval+ problems, scores the results with
[evalplus](https://github.com/evalplus/evalplus), and tears the server back
down. Downloading the *next* quant overlaps with benchmarking the *current*
one, but only one `llama-server` (and therefore GPU) runs at a time. Any
quant that wasn't already in your cache before the run is deleted afterward
to keep disk usage bounded.

## Prerequisites

- **`llama-server`** (from [llama.cpp](https://github.com/ggml-org/llama.cpp))
  must be built and on your `PATH`, or pointed at via `--llama-server-bin`.
  Use a reasonably recent build: multi-shard GGUF quants (files split across
  `-00001-of-000NN.gguf` parts) hit a shard-ordering bug in older builds when
  loaded from a Hugging Face cache directory
  ([llama.cpp#21016](https://github.com/ggml-org/llama.cpp/issues/21016),
  fixed in #21019) — if you see an "illegal split file idx" error on a
  multi-shard quant, update your build.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.

## Usage

```sh
# see what quants a repo has without downloading anything
uv run quantbench unsloth/Qwen3.5-0.8B-GGUF --list-quants

# quick smoke test: one quant, 5 problems
uv run quantbench unsloth/Qwen3.5-0.8B-GGUF --quants Q4_K_M --limit 5

# full run across every discovered quant, skipping a couple
uv run quantbench unsloth/Qwen3.5-0.8B-GGUF --quants-except Q2_K,IQ1_S

# full run across every discovered quant
uv run quantbench unsloth/Qwen3.5-0.8B-GGUF
```

Results land in `results/<org>__<repo>/`: `results.csv` (pass@1 per quant,
base HumanEval and HumanEval+ separately, file size, aggregate throughput
and wall time), `results.png` (bar chart, best to worst, with error bars and
significance vs the best quant when pass@k stats are available), and a
per-quant subfolder with the raw generations, sanitized samples, evalplus
results, and `llama-server` log.

## Options

| Flag | Default | What it does |
|---|---|---|
| `--quants` | all discovered | Comma-separated subset of quant names to run |
| `--quants-except` | none | Comma-separated quant names to exclude |
| `--list-quants` | | Print discovered quants and exit |
| `--limit N` | all 164 | Only run the first N HumanEval+ problems |
| `--n-samples N` | 1 | Number of generation samples (for pass@k with k > 1) |
| `--pass-at-k` | 1 | Comma-separated k values for pass@k reporting (must not exceed --n-samples) |
| `--temperature` | 0.0 | Sampling temperature (0.0 = greedy) |
| `--seed` | 42 | Random seed for sampling (reproducible pass@k) |
| `--parallel N` | 1 | Concurrent generations per quant; also sets `llama-server -np`. Speeds up runs (especially pass@k); the reported tok/s is then aggregate throughput, and KV cache memory scales with N |
| `--ctx-size` | model's own | `llama-server --ctx-size` override |
| `--gpu-layers` | `all` | `llama-server -ngl` value |
| `--max-tokens` | 32000 | Maximum tokens generated per problem |
| `--llama-server-bin` | `llama-server` | Path to the binary |
| `--cache-dir` | HF default | Override the Hugging Face cache directory |
| `--output-dir` | `results/<org>__<repo>` | Where per-quant output and the report go |
| `--keep-downloads` | off | Don't delete quants this run downloaded |
| `--retry-failed` | off | Re-run quants whose previous attempt failed (errors are otherwise cached on resume) |
| `--version` | | Print quantbench version and exit |

## How quant grouping works

A repo's GGUF files are grouped into quants either by subfolder (the common
unsloth/TheBloke convention for multi-shard quants) or by parsing a quant
token (`Q4_K_M`, `IQ2_XS`, unsloth's `UD-...` dynamic quants, etc.) out of
the filename. Run `--list-quants` first on an unfamiliar repo to sanity-check
the grouping before committing to a full run.

## Resuming and retrying

Completed quants are resumed from their per-quant `summary.json`, so an
interrupted run picks up where it left off. The cache is keyed on the
parameters that affect results (`--limit`, `--n-samples`, `--temperature`,
`--pass-at-k`, `--ctx-size`, `--max-tokens`, `--gpu-layers`, `--seed`); change
any of them and all quants are re-benchmarked. Quants that *failed* are also
cached, so a plain rerun skips them too — pass `--retry-failed` to give them
another shot. The exit code is 0 only if every selected quant succeeded,
so a partially failed run is detectable in CI.
