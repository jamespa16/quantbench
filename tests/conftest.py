"""Shared fixtures for quantbench tests."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from quantbench.discovery import Quant
from quantbench.eval_runner import EvalResult, PassAtKStats
from quantbench.orchestrator import QuantOutcome

# ---------------------------------------------------------------------------
# Repo ID
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo_id() -> str:
    """A fake Hugging Face repo ID used across tests."""
    return "author/model"


# ---------------------------------------------------------------------------
# Mocked HfApi
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_hf_api(request: pytest.FixtureRequest) -> MagicMock:
    """Return a mocked HfApi instance.

    Configure the response via ``mock_hf_api.repo_info.return_value = ModelInfo(...)``
    (discovery goes through ``repo_info(..., files_metadata=True)``).
    Use with ``patch("quantbench.<module>.HfApi", return_value=mock_hf_api)``.
    """
    return MagicMock()


@pytest.fixture()
def patched_hf_api(mock_hf_api: MagicMock) -> Iterator[MagicMock]:
    """Patch ``HfApi`` in ``quantbench.discovery`` with a mock instance.

    Yields the mock so tests can configure ``.return_value`` before use.
    """
    with patch("quantbench.discovery.HfApi", return_value=mock_hf_api):
        yield mock_hf_api


# ---------------------------------------------------------------------------
# Sample Quant objects
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_quant() -> Quant:
    """A single-file Quant (Q4_K_M)."""
    return Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",))


@pytest.fixture()
def sample_quant_sharded() -> Quant:
    """A sharded Quant (Q8_0 with 4 shards)."""
    return Quant(
        name="Q8_0",
        files=(
            "model-Q8_0-00001-of-00004.gguf",
            "model-Q8_0-00002-of-00004.gguf",
            "model-Q8_0-00003-of-00004.gguf",
            "model-Q8_0-00004-of-00004.gguf",
        ),
    )


@pytest.fixture()
def sample_quants() -> list[Quant]:
    """A list of three Quants for testing filtering and grouping."""
    return [
        Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",)),
        Quant(name="Q8_0", files=("model-Q8_0-00001-of-00002.gguf", "model-Q8_0-00002-of-00002.gguf")),
        Quant(name="BF16", files=("model-BF16.gguf",)),
    ]


# ---------------------------------------------------------------------------
# Sample PassAtKStats objects
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_pass_at_k_stats() -> PassAtKStats:
    """A PassAtKStats instance with typical values."""
    return PassAtKStats(mean=0.70, std_dev=0.1, std_error=0.02)


# ---------------------------------------------------------------------------
# Sample EvalResult objects
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_eval_result() -> EvalResult:
    """An EvalResult without pass@k data."""
    return EvalResult(
        base_pass1=0.5, extra_pass1=0.4, n_problems=100,
        avg_tok_s=50.0, wall_time_s=60.0,
    )


@pytest.fixture()
def sample_eval_result_with_pass_at_k() -> EvalResult:
    """An EvalResult with full pass@k data (k=1)."""
    return EvalResult(
        base_pass1=0.5,
        extra_pass1=0.4,
        n_problems=100,
        pass_at_k={1: 0.70},
        pass_at_k_stats={1: PassAtKStats(mean=0.70, std_dev=0.1, std_error=0.02)},
        pass_at_k_per_task={
            1: {
                "HumanEval/0": 0.8,
                "HumanEval/1": 0.7,
                "HumanEval/2": 0.6,
                "HumanEval/3": 0.7,
                "HumanEval/4": 0.7,
            }
        },
        avg_tok_s=50.0,
        wall_time_s=60.0,
    )


# ---------------------------------------------------------------------------
# Sample QuantOutcome objects
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_quant_outcome(sample_eval_result: EvalResult) -> QuantOutcome:
    """A QuantOutcome with a successful result."""
    return QuantOutcome(quant_name="Q4_K_M", size_bytes=4_000_000_000, result=sample_eval_result)


@pytest.fixture()
def sample_quant_outcome_with_pass_at_k(sample_eval_result_with_pass_at_k: EvalResult) -> QuantOutcome:
    """A QuantOutcome with full pass@k data."""
    return QuantOutcome(quant_name="Q4_K_M", size_bytes=4_000_000_000, result=sample_eval_result_with_pass_at_k)


@pytest.fixture()
def sample_quant_outcome_with_error() -> QuantOutcome:
    """A QuantOutcome with an error (no result)."""
    return QuantOutcome(quant_name="Q2_K", size_bytes=0, result=None, error="timeout")


@pytest.fixture()
def sample_quant_outcomes() -> list[QuantOutcome]:
    """Three QuantOutcome instances with pass@k data for CSV/chart tests."""
    q4 = QuantOutcome(
        quant_name="Q4_K_M",
        size_bytes=4_000_000_000,
        result=EvalResult(
            base_pass1=0.5,
            extra_pass1=0.4,
            n_problems=100,
            pass_at_k={1: 0.70},
            pass_at_k_stats={1: PassAtKStats(mean=0.70, std_dev=0.1, std_error=0.02)},
            pass_at_k_per_task={1: {f"HumanEval/{i}": s for i, s in enumerate([0.8, 0.7, 0.6, 0.7, 0.7])}},
        ),
    )
    q8 = QuantOutcome(
        quant_name="Q8_0",
        size_bytes=8_000_000_000,
        result=EvalResult(
            base_pass1=0.5,
            extra_pass1=0.4,
            n_problems=100,
            pass_at_k={1: 0.60},
            pass_at_k_stats={1: PassAtKStats(mean=0.60, std_dev=0.12, std_error=0.025)},
            pass_at_k_per_task={1: {f"HumanEval/{i}": s for i, s in enumerate([0.7, 0.6, 0.5, 0.6, 0.6])}},
        ),
    )
    q2 = QuantOutcome(
        quant_name="Q2_K",
        size_bytes=2_000_000_000,
        result=EvalResult(
            base_pass1=0.3,
            extra_pass1=0.2,
            n_problems=100,
            pass_at_k={1: 0.40},
            pass_at_k_stats={1: PassAtKStats(mean=0.40, std_dev=0.15, std_error=0.03)},
            pass_at_k_per_task={1: {f"HumanEval/{i}": s for i, s in enumerate([0.5, 0.4, 0.3, 0.4, 0.4])}},
        ),
    )
    return [q4, q8, q2]
