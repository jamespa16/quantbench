"""Grouped download of a quant's GGUF files, with pre-existence tracking and cleanup.

Deletion is done at the individual blob level rather than via
`HFCacheInfo.delete_revisions()`. All quants in a GGUF repo normally resolve
to the *same* revision (one commit hash covers every file in the repo), so
deleting "the revision" after benchmarking one quant would wipe every other
quant sharing that revision -- including ones that were already cached
before this run and must be preserved. Per-file blob removal is the only
way to reclaim just the files this run downloaded.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import scan_cache_dir, snapshot_download, try_to_load_from_cache
from tqdm import tqdm

from quantbench.discovery import Quant


@dataclass(frozen=True)
class DownloadedQuant:
    quant: Quant
    model_path: str  # resolved local path to primary_file, for `llama-server -m`
    snapshot_dir: str
    was_pre_existing: bool


def was_cached(repo_id: str, quant: Quant, *, cache_dir: str | None = None) -> bool:
    """True if every file in this quant is already present in the local HF cache.

    Must be called before any `snapshot_download` for this repo in the
    current run, since a successful download makes this always return True.
    """
    return all(
        isinstance(try_to_load_from_cache(repo_id, f, cache_dir=cache_dir), str)
        for f in quant.files
    )


def download_quant(
    repo_id: str,
    quant: Quant,
    *,
    cache_dir: str | None = None,
    pre_existing: bool,
    tqdm_class: type[tqdm] | None = None,
) -> DownloadedQuant:
    snapshot_dir = snapshot_download(
        repo_id, allow_patterns=list(quant.files), cache_dir=cache_dir, tqdm_class=tqdm_class
    )
    return DownloadedQuant(
        quant=quant,
        model_path=os.path.join(snapshot_dir, quant.primary_file),
        snapshot_dir=snapshot_dir,
        was_pre_existing=pre_existing,
    )


def cleanup_quant(downloaded: DownloadedQuant, *, cache_dir: str | None = None) -> None:
    """Delete this quant's files, unless they predate this run.

    Only unlinks a blob if no other cached file (any repo/revision in this
    cache) still references it, so a coincidentally-shared blob is never
    pulled out from under something else that needs it.
    """
    if downloaded.was_pre_existing:
        return

    info = scan_cache_dir(cache_dir=cache_dir)
    blob_refcount: Counter[Path] = Counter()
    for repo in info.repos:
        for revision in repo.revisions:
            for cached_file in revision.files:
                blob_refcount[cached_file.blob_path] += 1

    touched_dirs: set[Path] = set()
    for rel_path in downloaded.quant.files:
        file_path = Path(downloaded.snapshot_dir) / rel_path
        blob_path = file_path.resolve() if file_path.is_symlink() else None

        file_path.unlink(missing_ok=True)
        touched_dirs.add(file_path.parent)

        if blob_path is not None and blob_refcount.get(blob_path, 0) <= 1 and blob_path.exists():
            blob_path.unlink()

    for directory in touched_dirs:
        try:
            directory.rmdir()  # only succeeds if now empty (multi-shard subfolder)
        except OSError:
            pass
