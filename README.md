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
base HumanEval and HumanEval+ separately, file size), `results.png` (bar
chart, best to worst), and a per-quant subfolder with the raw generations,
sanitized samples, evalplus results, and `llama-server` log.

## Options

| Flag | Default | What it does |
|---|---|---|
| `--quants` | all discovered | Comma-separated subset of quant names to run |
| `--quants-except` | none | Comma-separated quant names to exclude |
| `--list-quants` | | Print discovered quants and exit |
| `--limit N` | all 164 | Only run the first N HumanEval+ problems |
| `--ctx-size` | model's own | `llama-server --ctx-size` override |
| `--gpu-layers` | `all` | `llama-server -ngl` value |
| `--llama-server-bin` | `llama-server` | Path to the binary |
| `--cache-dir` | HF default | Override the Hugging Face cache directory |
| `--output-dir` | `results/<org>__<repo>` | Where per-quant output and the report go |
| `--keep-downloads` | off | Don't delete quants this run downloaded |

## How quant grouping works

A repo's GGUF files are grouped into quants either by subfolder (the common
unsloth/TheBloke convention for multi-shard quants) or by parsing a quant
token (`Q4_K_M`, `IQ2_XS`, unsloth's `UD-...` dynamic quants, etc.) out of
the filename. Run `--list-quants` first on an unfamiliar repo to sanity-check
the grouping before committing to a full run.
