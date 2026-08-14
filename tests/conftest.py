"""Shared fixtures for quantbench tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def repo_id() -> str:
    """A fake Hugging Face repo ID used across tests."""
    return "author/model"
