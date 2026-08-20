"""Grouped download of a quant's GGUF files, with pre-existence tracking and cleanup.

Deletion is done at the individual blob level rather than via
`HFCacheInfo.delete_revisions()`. All quants in a GGUF repo normally resolve
to the *same* revision (one commit hash covers every file in the repo), so
deleting "the revision" after benchmarking one quant would wipe every other
quant sharing that revision -- including ones that were already cached
before this run and must be preserved. Per-file blob removal is the only
way to reclaim just the files this run downloaded.

The blob reference count is computed by walking this repo's snapshot
symlinks only, not the whole cache (`scan_cache_dir` would re-walk every
cached model on every quant, which adds up on large caches). Sharing within
the repo (across revisions) is fully protected; a blob coincidentally shared
with a *different* repo would be deleted, and the other repo's symlink would
dangle until its next use re-downloads the blob -- acceptable, since the
blob store is a content-addressed cache and byte-identical GGUFs across
repos don't occur in practice.
"""

from __future__ import annotations

import contextlib
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download, try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE
from tqdm import tqdm

from quantbench.discovery import Quant


@dataclass(frozen=True)
class DownloadedQuant:
    quant: Quant
    repo_id: str  # HF repo these files were downloaded from
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
        repo_id=repo_id,
        model_path=os.path.join(snapshot_dir, quant.primary_file),
        snapshot_dir=snapshot_dir,
        was_pre_existing=pre_existing,
    )


def _repo_blob_refcount(
    cache_dir: str | None, repo_id: str, blob_paths: set[Path]
) -> Counter[Path]:
    """Count snapshot symlinks in *this repo only* that resolve to each blob.

    O(files in this repo) instead of O(every file in the cache). Includes
    this quant's own symlinks, so a blob referenced only by the files being
    deleted counts as 1 (== "safe to remove once they're gone").
    """
    hub_dir = Path(cache_dir) if cache_dir is not None else Path(HF_HUB_CACHE)
    repo_dir = hub_dir / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    counts: Counter[Path] = Counter()
    if not snapshots.is_dir():
        return counts
    for revision_dir in snapshots.iterdir():
        for path in revision_dir.rglob("*"):
            if path.is_symlink():
                target = path.resolve()
                if target in blob_paths:
                    counts[target] += 1
    return counts


def cleanup_quant(downloaded: DownloadedQuant, *, cache_dir: str | None = None) -> None:
    """Delete this quant's files, unless they predate this run.

    Only unlinks a blob if no other file in the same repo (any revision)
    still references it, so a blob shared with a sibling quant or an older
    revision is never pulled out from under it.
    """
    if downloaded.was_pre_existing:
        return

    file_paths = [Path(downloaded.snapshot_dir) / f for f in downloaded.quant.files]
    blob_paths = {p.resolve() for p in file_paths if p.is_symlink()}
    blob_refcount = _repo_blob_refcount(cache_dir, downloaded.repo_id, blob_paths)

    touched_dirs: set[Path] = set()
    for file_path in file_paths:
        file_path.unlink(missing_ok=True)
        touched_dirs.add(file_path.parent)

    for blob_path in blob_paths:
        if blob_refcount.get(blob_path, 0) <= 1 and blob_path.exists():
            blob_path.unlink()

    for directory in touched_dirs:
        with contextlib.suppress(OSError):
            directory.rmdir()  # only succeeds if now empty (multi-shard subfolder)
