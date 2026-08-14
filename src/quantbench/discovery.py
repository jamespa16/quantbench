"""Discover and group the GGUF quantizations hosted in a Hugging Face repo."""

from __future__ import annotations

import re
from dataclasses import dataclass

from huggingface_hub import HfApi

_SHARD_SUFFIX_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
# Structural pattern instead of an enumerated whitelist: quant naming grows
# new size-suffix combinations upstream regularly (e.g. unsloth's "_XL"), and
# an enumerated list silently mis-groups anything it doesn't recognize (e.g.
# matching unlisted "Q2_K_L" as bare "Q2_K", merging two distinct quants).
# Greedy matching here always prefers the longest valid token at a position.
_IQUANT = r"IQ[1-4]_(?:XXS|XS|S|M|L|NL)"
_KQUANT = r"Q[2-8](?:_[01]|_K(?:_(?:XXS|XS|S|M|L|XL))?)?"
_FLOAT_QUANT = r"BF16|F16|F32"
_QUANT_TOKEN_RE = re.compile(
    rf"(?:^|[._-])(UD-)?({_IQUANT}|{_KQUANT}|{_FLOAT_QUANT})(?:[._-]|$)"
)


@dataclass(frozen=True)
class Quant:
    """One quantization: a name and the ordered GGUF file(s) that make it up."""

    name: str
    files: tuple[str, ...]  # repo-relative paths, sorted by shard index

    @property
    def primary_file(self) -> str:
        """The file to pass to `llama-server -m` (first shard, or the only file)."""
        return self.files[0]


def _shard_index(path: str) -> int:
    m = _SHARD_SUFFIX_RE.search(path)
    return int(m.group(1)) if m else 0


def _match_quant_token(filename: str) -> str | None:
    stripped = _SHARD_SUFFIX_RE.sub(".gguf", filename)
    m = _QUANT_TOKEN_RE.search(stripped)
    if not m:
        return None
    prefix, token = m.groups()
    return f"{prefix or ''}{token}"


def fetch_file_sizes(repo_id: str) -> dict[str, int]:
    """Repo-relative path -> size in bytes, for every file in the repo.

    Used to show a real download percentage instead of an indeterminate
    spinner (`snapshot_download` itself doesn't expose the total upfront).
    """
    info = HfApi().repo_info(repo_id, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings if s.size is not None}


def discover_quants(repo_id: str) -> list[Quant]:
    """List a repo's GGUF files and group them into quants.

    Grouping heuristic (matches common unsloth/TheBloke conventions):
      - a file inside a subfolder is grouped by that subfolder's name
      - a flat file is grouped by a known quant token parsed out of its name
    Files that can't be confidently grouped are skipped with a warning rather
    than raising, since new/unknown quant naming shows up regularly upstream.
    """
    # mmproj-*.gguf files are a multimodal vision projector shared across quants,
    # not a quant of the LLM itself, and would otherwise merge into whichever
    # group happens to share its precision suffix (e.g. mmproj-BF16 into BF16).
    files = [
        f
        for f in HfApi().list_repo_files(repo_id)
        if f.lower().endswith(".gguf") and not f.rsplit("/", 1)[-1].lower().startswith("mmproj")
    ]

    groups: dict[str, list[str]] = {}
    for path in files:
        key = path.split("/", 1)[0] if "/" in path else _match_quant_token(path.rsplit("/", 1)[-1])
        if key is None:
            print(f"warning: could not determine quant for {path!r}, skipping")
            continue
        groups.setdefault(key, []).append(path)

    return [
        Quant(name=name, files=tuple(sorted(paths, key=_shard_index)))
        for name, paths in sorted(groups.items())
    ]
