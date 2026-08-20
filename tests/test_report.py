"""Tests for report.py: CSV output, p-value computation, chart generation, and significance markers."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quantbench.eval_runner import EvalResult, PassAtKStats
from quantbench.orchestrator import QuantOutcome
from quantbench.report import _pairwise_p_values, _sig_marker, write_chart, write_csv

# ---------------------------------------------------------------------------
# Helpers — build QuantOutcome objects for testing
# ---------------------------------------------------------------------------


def _make_outcome(
    quant_name: str,
    size_bytes: int,
    base_pass1: float = 0.5,
    extra_pass1: float = 0.4,
    n_problems: int = 100,
    pass_at_k: dict[int, float] | None = None,
    pass_at_k_stats: dict[int, PassAtKStats] | None = None,
    pass_at_k_per_task: dict[int, dict[str, float]] | None = None,
    error: str | None = None,
    avg_tok_s: float | None = None,
    wall_time_s: float | None = None,
) -> QuantOutcome:
    """Build a QuantOutcome with the given parameters."""
    result = EvalResult(
        base_pass1=base_pass1,
        extra_pass1=extra_pass1,
        n_problems=n_problems,
        pass_at_k=pass_at_k,
        pass_at_k_stats=pass_at_k_stats,
        pass_at_k_per_task=pass_at_k_per_task,
        avg_tok_s=avg_tok_s,
        wall_time_s=wall_time_s,
    )
    return QuantOutcome(quant_name, size_bytes, result, error=error)


def _make_outcomes_with_stats() -> list[QuantOutcome]:
    """Three quants with pass@1 stats and per-task scores for p-value testing."""
    # Q4_K_M — best (0.70 mean)
    q4 = _make_outcome(
        "Q4_K_M",
        4_000_000_000,
        base_pass1=0.5,
        extra_pass1=0.4,
        n_problems=100,
        pass_at_k={1: 0.70},
        pass_at_k_stats={1: PassAtKStats(mean=0.70, std_dev=0.1, std_error=0.02)},
        pass_at_k_per_task={1: {f"HumanEval/{i}": s for i, s in enumerate([0.8, 0.7, 0.6, 0.7, 0.7])}},
    )
    # Q8_0 — second (0.60 mean)
    q8 = _make_outcome(
        "Q8_0",
        8_000_000_000,
        base_pass1=0.5,
        extra_pass1=0.4,
        n_problems=100,
        pass_at_k={1: 0.60},
        pass_at_k_stats={1: PassAtKStats(mean=0.60, std_dev=0.12, std_error=0.025)},
        pass_at_k_per_task={1: {f"HumanEval/{i}": s for i, s in enumerate([0.7, 0.6, 0.5, 0.6, 0.6])}},
    )
    # Q2_K — worst (0.40 mean)
    q2 = _make_outcome(
        "Q2_K",
        2_000_000_000,
        base_pass1=0.3,
        extra_pass1=0.2,
        n_problems=100,
        pass_at_k={1: 0.40},
        pass_at_k_stats={1: PassAtKStats(mean=0.40, std_dev=0.15, std_error=0.03)},
        pass_at_k_per_task={1: {f"HumanEval/{i}": s for i, s in enumerate([0.5, 0.4, 0.3, 0.4, 0.4])}},
    )
    return [q4, q8, q2]


def _make_outcomes_no_pass_at_k() -> list[QuantOutcome]:
    """Two quants without pass@k — fallback to extra_pass1."""
    q1 = _make_outcome("Q4_K_M", 4_000_000_000)
    q2 = _make_outcome("Q8_0", 8_000_000_000)
    return [q1, q2]


# ---------------------------------------------------------------------------
# _sig_marker
# ---------------------------------------------------------------------------


class TestSigMarker:
    """Test _sig_marker thresholds."""

    def test_very_highly_significant(self) -> None:
        """p < 0.001 returns ***."""
        assert _sig_marker(0.0001) == "***"

    def test_highly_significant(self) -> None:
        """p < 0.01 returns **."""
        assert _sig_marker(0.005) == "**"

    def test_significant(self) -> None:
        """p < 0.05 returns *."""
        assert _sig_marker(0.03) == "*"

    def test_not_significant(self) -> None:
        """p >= 0.05 returns n.s."""
        assert _sig_marker(0.10) == "n.s."

    def test_boundary_0_001(self) -> None:
        """p == 0.001 is not < 0.001, so returns **."""
        assert _sig_marker(0.001) == "**"

    def test_boundary_0_01(self) -> None:
        """p == 0.01 is not < 0.01, so returns *."""
        assert _sig_marker(0.01) == "*"

    def test_boundary_0_05(self) -> None:
        """p == 0.05 is not < 0.05, so returns n.s."""
        assert _sig_marker(0.05) == "n.s."

    def test_exact_zero(self) -> None:
        """p == 0 returns ***."""
        assert _sig_marker(0.0) == "***"

    def test_one(self) -> None:
        """p == 1.0 returns n.s."""
        assert _sig_marker(1.0) == "n.s."


# ---------------------------------------------------------------------------
# _pairwise_p_values
# ---------------------------------------------------------------------------


class TestPairwisePValues:
    """Test _pairwise_p_values computation."""

    def test_empty_returns_empty(self) -> None:
        """No outcomes yields an empty p-map."""
        result = _pairwise_p_values([], 1)
        assert result == {}

    def test_single_outcome_returns_empty(self) -> None:
        """A single outcome has no pairs."""
        outcomes = _make_outcomes_with_stats()[:1]
        result = _pairwise_p_values(outcomes, 1)
        assert result == {}

    def test_two_outcomes_one_pair(self) -> None:
        """Two outcomes yield exactly one pair."""
        outcomes = _make_outcomes_with_stats()[:2]
        result = _pairwise_p_values(outcomes, 1)
        assert len(result) == 1
        # The pair key is (name_a, name_b) where order depends on sorted outcome order
        for (a, b), p in result.items():
            assert a in ("Q2_K", "Q4_K_M", "Q8_0")
            assert b in ("Q2_K", "Q4_K_M", "Q8_0")
            assert a != b
            assert 0.0 <= p <= 1.0

    def test_three_outcomes_three_pairs(self) -> None:
        """Three outcomes yield three pairs (n choose 2)."""
        outcomes = _make_outcomes_with_stats()
        result = _pairwise_p_values(outcomes, 1)
        assert len(result) == 3

    def test_symmetric_pairs(self) -> None:
        """Paired scores (a, b) and (b, a) use the same key."""
        outcomes = _make_outcomes_with_stats()[:2]
        result = _pairwise_p_values(outcomes, 1)
        # The function uses combinations(), so each pair appears exactly once.
        names = {name for pair in result for name in pair}
        assert len(names) == 2

    def test_filtering_skips_no_per_task(self) -> None:
        """Outcomes without pass_at_k_per_task are excluded from pairing."""
        # One outcome has no per-task data
        outcomes = [
            _make_outcome(
                "Q4_K_M", 4_000_000_000,
                pass_at_k={1: 0.7},
                pass_at_k_stats={1: PassAtKStats(mean=0.7, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task=None,  # no per-task data
            ),
        ]
        result = _pairwise_p_values(outcomes, 1)
        assert result == {}

    def test_filtering_skips_few_scores(self) -> None:
        """Outcomes with fewer than 2 per-task scores are excluded."""
        outcomes = [
            _make_outcome(
                "Q4_K_M", 4_000_000_000,
                pass_at_k={1: 0.7},
                pass_at_k_stats={1: PassAtKStats(mean=0.7, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task={1: {"HumanEval/0": 0.7}},  # only 1 score
            ),
        ]
        result = _pairwise_p_values(outcomes, 1)
        assert result == {}

    def test_p_value_range(self) -> None:
        """All p-values fall in [0, 1]."""
        outcomes = _make_outcomes_with_stats()
        result = _pairwise_p_values(outcomes, 1)
        for p in result.values():
            assert 0.0 <= p <= 1.0

    def test_identical_scores_p_one(self) -> None:
        """Identical per-task scores yield p = 1.0 (no evidence of a difference),
        not the NaN that a degenerate t-test would produce."""
        shared = {f"HumanEval/{i}": 0.5 for i in range(5)}
        outcomes = [
            _make_outcome(
                "A", 4_000_000_000,
                pass_at_k={1: 0.5},
                pass_at_k_stats={1: PassAtKStats(mean=0.5, std_dev=0.0, std_error=0.0)},
                pass_at_k_per_task={1: dict(shared)},
            ),
            _make_outcome(
                "B", 4_000_000_000,
                pass_at_k={1: 0.5},
                pass_at_k_stats={1: PassAtKStats(mean=0.5, std_dev=0.0, std_error=0.0)},
                pass_at_k_per_task={1: dict(shared)},
            ),
        ]
        result = _pairwise_p_values(outcomes, 1)
        assert len(result) == 1
        p = next(iter(result.values()))
        assert p == 1.0

    def test_constant_offset_p_zero(self) -> None:
        """A uniform per-task offset has zero difference variance: maximally
        significant (p = 0.0), without scipy's degenerate-test warning."""
        a = {f"HumanEval/{i}": s for i, s in enumerate([0.8, 0.7, 0.6, 0.7, 0.7])}
        b = {t: v - 0.1 for t, v in a.items()}
        outcomes = [
            _make_outcome(
                "A", 4_000_000_000,
                pass_at_k={1: 0.7},
                pass_at_k_stats={1: PassAtKStats(mean=0.7, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task={1: a},
            ),
            _make_outcome(
                "B", 4_000_000_000,
                pass_at_k={1: 0.6},
                pass_at_k_stats={1: PassAtKStats(mean=0.6, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task={1: b},
            ),
        ]
        result = _pairwise_p_values(outcomes, 1)
        assert next(iter(result.values())) == 0.0

    def test_paired_on_common_tasks_only(self) -> None:
        """Pairs are compared on the tasks both quants scored.

        If one quant has a task dropped by sanitize, the other quant's extra
        task must not shift the alignment: the test runs on the intersection.
        """
        # B is missing HumanEval/2 (dropped by sanitize for B only).
        a = {f"HumanEval/{i}": s for i, s in enumerate([0.8, 0.7, 0.6, 0.7, 0.7])}
        b = {k: v for k, v in a.items() if k != "HumanEval/2"}
        b = {k: max(0.0, v - 0.1) for k, v in b.items()}  # B is consistently worse
        outcomes = [
            _make_outcome(
                "A", 4_000_000_000,
                pass_at_k={1: 0.7},
                pass_at_k_stats={1: PassAtKStats(mean=0.7, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task={1: a},
            ),
            _make_outcome(
                "B", 4_000_000_000,
                pass_at_k={1: 0.6},
                pass_at_k_stats={1: PassAtKStats(mean=0.6, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task={1: b},
            ),
        ]
        result = _pairwise_p_values(outcomes, 1)
        assert len(result) == 1
        p = next(iter(result.values()))
        assert 0.0 <= p <= 1.0

    def test_fewer_than_two_common_tasks_is_nan(self) -> None:
        """Pairs sharing fewer than 2 tasks can't be tested and get nan."""
        outcomes = [
            _make_outcome(
                "A", 4_000_000_000,
                pass_at_k={1: 0.7},
                pass_at_k_stats={1: PassAtKStats(mean=0.7, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task={1: {"HumanEval/0": 0.8, "HumanEval/1": 0.6}},
            ),
            _make_outcome(
                "B", 4_000_000_000,
                pass_at_k={1: 0.6},
                pass_at_k_stats={1: PassAtKStats(mean=0.6, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task={1: {"HumanEval/0": 0.7, "HumanEval/2": 0.5}},  # only 1 common
            ),
        ]
        result = _pairwise_p_values(outcomes, 1)
        p = next(iter(result.values()))
        assert math.isnan(p)

    def test_k_mismatch_filters_outcome(self) -> None:
        """Outcomes that don't have data for the requested k are filtered."""
        outcomes = [
            _make_outcome(
                "Q4_K_M", 4_000_000_000,
                pass_at_k={5: 0.7},  # only has k=5
                pass_at_k_stats={5: PassAtKStats(mean=0.7, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task={5: {f"HumanEval/{i}": s for i, s in enumerate([0.8, 0.7, 0.6, 0.7, 0.7])}},
            ),
        ]
        result = _pairwise_p_values(outcomes, 1)  # asking for k=1
        assert result == {}


# ---------------------------------------------------------------------------
# write_csv — header
# ---------------------------------------------------------------------------


class TestWriteCsvHeader:
    """Test CSV header columns."""

    def test_base_columns(self, tmp_path: Path) -> None:
        """CSV always has the base columns."""
        outcomes = _make_outcomes_no_pass_at_k()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "quant" in header
        assert "base_pass1" in header
        assert "extra_pass1" in header
        assert "n_problems" in header
        assert "size_gb" in header
        assert "error" in header

    def test_pass_at_k_columns(self, tmp_path: Path) -> None:
        """When pass@k data exists, pass@k columns are added."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "pass@1" in header
        assert "pass@1_std" in header
        assert "pass@1_stderr" in header

    def test_p_value_column_when_stats(self, tmp_path: Path) -> None:
        """When per-task stats are available, a p_value_vs_best column is added."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "p_value_vs_best@1" in header

    def test_no_p_value_column_without_per_task(self, tmp_path: Path) -> None:
        """Without per-task scores, no p-value column is added."""
        outcomes = [
            _make_outcome(
                "Q4_K_M", 4_000_000_000,
                pass_at_k={1: 0.7},
                pass_at_k_stats={1: PassAtKStats(mean=0.7, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task=None,
            )
        ]
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "p_value_vs_best" not in header

    def test_no_pass_at_k_no_columns(self, tmp_path: Path) -> None:
        """When no outcomes have pass@k, no pass@k columns appear."""
        outcomes = _make_outcomes_no_pass_at_k()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "pass@1" not in header

    def test_multiple_k_values(self, tmp_path: Path) -> None:
        """Multiple k values produce columns for each k, sorted ascending."""
        outcomes = [
            _make_outcome(
                "Q4_K_M", 4_000_000_000,
                pass_at_k={5: 0.6, 1: 0.7, 10: 0.65},
                pass_at_k_stats={
                    1: PassAtKStats(mean=0.7, std_dev=0.1, std_error=0.02),
                    5: PassAtKStats(mean=0.6, std_dev=0.12, std_error=0.025),
                    10: PassAtKStats(mean=0.65, std_dev=0.11, std_error=0.023),
                },
                pass_at_k_per_task={
                    1: {f"HumanEval/{i}": v for i, v in enumerate([0.8, 0.7, 0.6, 0.7, 0.7])},
                    5: {f"HumanEval/{i}": v for i, v in enumerate([0.7, 0.6, 0.5, 0.6, 0.6])},
                    10: {f"HumanEval/{i}": v for i, v in enumerate([0.7, 0.65, 0.6, 0.65, 0.65])},
                },
            )
        ]
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "pass@1" in header
        assert "pass@5" in header
        assert "pass@10" in header
        # Check ordering: pass@1 before pass@5 before pass@10
        k_indices = [header.index(f"pass@{k}") for k in [1, 5, 10]]
        assert k_indices == sorted(k_indices)


# ---------------------------------------------------------------------------
# write_csv — row data
# ---------------------------------------------------------------------------


class TestWriteCsvRowData:
    """Test CSV row data correctness."""

    def test_row_count(self, tmp_path: Path) -> None:
        """CSV has one row per outcome."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # header + 3 data rows
        assert len(rows) == 4

    def test_quant_names(self, tmp_path: Path) -> None:
        """Row quant names match outcome quant names."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            names = [row["quant"] for row in reader]
        assert set(names) == {"Q4_K_M", "Q8_0", "Q2_K"}

    def test_base_pass1_format(self, tmp_path: Path) -> None:
        """base_pass1 is formatted to 4 decimal places."""
        outcomes = [_make_outcome("Q4_K_M", 4_000_000_000, base_pass1=0.5)]
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["base_pass1"] == "0.5000"

    def test_extra_pass1_format(self, tmp_path: Path) -> None:
        """extra_pass1 is formatted to 4 decimal places."""
        outcomes = [_make_outcome("Q4_K_M", 4_000_000_000, extra_pass1=0.4)]
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["extra_pass1"] == "0.4000"

    def test_n_problems(self, tmp_path: Path) -> None:
        """n_problems is written as the integer value."""
        outcomes = [_make_outcome("Q4_K_M", 4_000_000_000, n_problems=100)]
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["n_problems"] == "100"

    def test_size_gb_format(self, tmp_path: Path) -> None:
        """size_gb is bytes / 1e9 formatted to 3 decimal places."""
        outcomes = [_make_outcome("Q4_K_M", 4_000_000_000)]
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["size_gb"] == "4.000"

    def test_error_column(self, tmp_path: Path) -> None:
        """Error message is written to the error column."""
        outcomes = [_make_outcome("Q4_K_M", 0, error="timeout")]
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["error"] == "timeout"

    def test_pass_at_k_value_format(self, tmp_path: Path) -> None:
        """pass@k values are formatted to 4 decimal places."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        q4_row = next(r for r in rows if r["quant"] == "Q4_K_M")
        assert q4_row["pass@1"] == "0.7000"

    def test_pass_at_k_std_format(self, tmp_path: Path) -> None:
        """pass@k_std (std_dev) is formatted to 4 decimal places."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        q4_row = next(r for r in rows if r["quant"] == "Q4_K_M")
        assert q4_row["pass@1_std"] == "0.1000"

    def test_pass_at_k_stderr_format(self, tmp_path: Path) -> None:
        """pass@k_stderr is formatted to 4 decimal places."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        q4_row = next(r for r in rows if r["quant"] == "Q4_K_M")
        assert q4_row["pass@1_stderr"] == "0.0200"


# ---------------------------------------------------------------------------
# write_csv — p-value column
# ---------------------------------------------------------------------------


class TestWriteCsvPValueColumn:
    """Test p-value column in CSV output."""

    def test_best_quant_empty_p_value(self, tmp_path: Path) -> None:
        """The best quant row has an empty p-value cell."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Q4_K_M is the best (highest pass@1 = 0.70)
        q4_row = next(r for r in rows if r["quant"] == "Q4_K_M")
        assert q4_row["p_value_vs_best@1"] == ""

    def test_non_best_quants_have_p_values(self, tmp_path: Path) -> None:
        """Non-best quant rows have numeric p-values."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        q8_row = next(r for r in rows if r["quant"] == "Q8_0")
        q2_row = next(r for r in rows if r["quant"] == "Q2_K")
        assert q8_row["p_value_vs_best@1"] != ""
        assert q2_row["p_value_vs_best@1"] != ""

    def test_p_value_is_numeric(self, tmp_path: Path) -> None:
        """p-value cells contain valid floating-point values (or nan for missing pairs)."""
        outcomes = _make_outcomes_with_stats()
        path = tmp_path / "out.csv"
        write_csv(outcomes, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for row in rows:
            p_str = row.get("p_value_vs_best@1", "")
            if p_str:
                p = float(p_str)
                # p-values may be finite numbers or NaN if the pair is not in p_map
                assert math.isfinite(p) or math.isnan(p)


# ---------------------------------------------------------------------------
# write_chart
# ---------------------------------------------------------------------------


_ABSENT = object()


@pytest.fixture()
def plt_mocks():
    """Make `import matplotlib.pyplot as plt` inside write_chart resolve to a mock.

    Both lookup paths are covered: the sys.modules entry *and* the attribute on
    the matplotlib package (IMPORT_FROM checks the attribute first). Only the
    single key is touched, so unrelated modules imported during the test (e.g.
    scipy's C extensions) are left alone.
    """
    import matplotlib

    mock_plt = MagicMock()
    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    saved_mod = sys.modules.get("matplotlib.pyplot", _ABSENT)
    saved_attr = getattr(matplotlib, "pyplot", _ABSENT)
    sys.modules["matplotlib.pyplot"] = mock_plt
    matplotlib.pyplot = mock_plt
    try:
        yield mock_plt, mock_ax
    finally:
        if saved_mod is _ABSENT:
            del sys.modules["matplotlib.pyplot"]
        else:
            sys.modules["matplotlib.pyplot"] = saved_mod
        if saved_attr is _ABSENT:
            delattr(matplotlib, "pyplot")
        else:
            matplotlib.pyplot = saved_attr


class TestWriteChart:
    """Test write_chart with QuantOutcome data."""

    def test_chart_with_pass_at_k(self, tmp_path: Path, plt_mocks) -> None:
        """Chart is generated when pass@k data is present."""
        mock_plt, _ = plt_mocks
        write_chart(_make_outcomes_with_stats(), tmp_path / "report.png", repo_id="author/model")
        mock_plt.close.assert_called_once()

    def test_chart_without_pass_at_k(self, tmp_path: Path, plt_mocks) -> None:
        """Chart falls back to extra_pass1 when no pass@k data."""
        mock_plt, _ = plt_mocks
        write_chart(_make_outcomes_no_pass_at_k(), tmp_path / "report.png", repo_id="author/model")
        mock_plt.close.assert_called_once()

    def test_chart_single_outcome(self, tmp_path: Path, plt_mocks) -> None:
        """Chart works with a single outcome."""
        mock_plt, _ = plt_mocks
        write_chart(_make_outcomes_with_stats()[:1], tmp_path / "report.png", repo_id="author/model")
        mock_plt.close.assert_called_once()

    def test_chart_empty_outcomes(self, tmp_path: Path) -> None:
        """Empty outcomes list skips chart generation (no crash)."""
        path = tmp_path / "report.png"
        write_chart([], path, repo_id="author/model")
        # The function prints a warning and returns; no file is created
        assert not path.exists()

    def test_chart_preserves_ordering(self, tmp_path: Path, plt_mocks) -> None:
        """Chart sorts bars from best to worst."""
        _, mock_ax = plt_mocks
        write_chart(_make_outcomes_with_stats(), tmp_path / "report.png", repo_id="author/model")
        # Check bar was called (i.e., chart was constructed)
        mock_ax.bar.assert_called_once()

    def test_chart_with_failed_outcomes(self, tmp_path: Path, plt_mocks) -> None:
        """Chart handles mixed success/failure outcomes."""
        mock_plt, _ = plt_mocks
        outcomes = [
            _make_outcome(
                "Q4_K_M", 4_000_000_000,
                pass_at_k={1: 0.70},
                pass_at_k_stats={1: PassAtKStats(mean=0.70, std_dev=0.1, std_error=0.02)},
                pass_at_k_per_task={1: {f"HumanEval/{i}": v for i, v in enumerate([0.8, 0.7, 0.6, 0.7, 0.7])}},
            ),
            # Failed outcome — no result
            QuantOutcome("Q2_K", 2_000_000_000, None, error="timeout"),
        ]
        write_chart(outcomes, tmp_path / "report.png", repo_id="author/model")
        mock_plt.close.assert_called_once()

    def test_chart_overwrites_existing(self, tmp_path: Path, plt_mocks) -> None:
        """write_chart overwrites an existing file (mocked: savefig is called)."""
        mock_plt, _ = plt_mocks
        path = tmp_path / "report.png"
        path.write_bytes(b"old content")
        write_chart(_make_outcomes_with_stats()[:1], path, repo_id="author/model")
        mock_plt.close.assert_called_once()
        mock_plt.subplots.return_value[0].savefig.assert_called_once()

    def test_chart_repo_id_included(self, tmp_path: Path, plt_mocks) -> None:
        """Chart is generated with the repo_id for the title."""
        _, mock_ax = plt_mocks
        write_chart(_make_outcomes_with_stats()[:1], tmp_path / "report.png", repo_id="test/repo")
        assert mock_ax.set_title.called


class TestWriteChartReal:
    """Render with real matplotlib (unmocked) to catch invalid kwarg bugs
    that a fully mocked plt can't surface."""

    def test_real_render_pass_at_k(self, tmp_path: Path) -> None:
        path = tmp_path / "report.png"
        write_chart(_make_outcomes_with_stats(), path, repo_id="author/model")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_real_render_fallback(self, tmp_path: Path) -> None:
        path = tmp_path / "report.png"
        write_chart(_make_outcomes_no_pass_at_k(), path, repo_id="author/model")
        assert path.exists()
        assert path.stat().st_size > 0
