"""Tests for cli.py: argument parsing, validation, and _run_key generation."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quantbench import __version__
from quantbench.cli import _parse_args, _run_key, main

# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

class TestParseArgsBasic:
    """Basic argument parsing."""

    def test_required_repo_id(self) -> None:
        args = _parse_args(["author/model"])
        assert args.repo_id == "author/model"

    def test_defaults(self) -> None:
        args = _parse_args(["author/model"])
        assert args.n_samples == 1
        assert args.temperature == 0.0
        assert args.gpu_layers == "all"
        assert args.llama_server_bin == "llama-server"
        assert args.quants is None
        assert args.quants_except is None
        assert args.list_quants is False
        assert args.limit is None
        assert args.ctx_size is None
        assert args.cache_dir is None
        assert args.output_dir is None
        assert args.pass_at_k is None
        assert args.keep_downloads is False
        assert args.max_tokens == 32000
        assert args.seed == 42
        assert args.parallel == 1
        assert args.retry_failed is False

    def test_all_flags(self) -> None:
        args = _parse_args(
            [
                "author/model",
                "--quants",
                "Q4_K_M,Q8_0",
                "--limit",
                "5",
                "--ctx-size",
                "8192",
                "--gpu-layers",
                "99",
                "--llama-server-bin",
                "/usr/bin/llama-server",
                "--cache-dir",
                "/tmp/cache",
                "--output-dir",
                "/tmp/results",
                "--n-samples",
                "10",
                "--temperature",
                "0.5",
                "--pass-at-k",
                "1,10",
                "--keep-downloads",
                "--list-quants",
                "--max-tokens",
                "8192",
                "--seed",
                "7",
                "--parallel",
                "4",
                "--retry-failed",
            ]
        )
        assert args.repo_id == "author/model"
        assert args.quants == "Q4_K_M,Q8_0"
        assert args.limit == 5
        assert args.ctx_size == 8192
        assert args.gpu_layers == "99"
        assert args.llama_server_bin == "/usr/bin/llama-server"
        assert args.cache_dir == "/tmp/cache"
        assert args.output_dir == "/tmp/results"
        assert args.n_samples == 10
        assert args.temperature == 0.5
        assert args.pass_at_k == "1,10"
        assert args.keep_downloads is True
        assert args.list_quants is True
        assert args.max_tokens == 8192
        assert args.seed == 7
        assert args.parallel == 4
        assert args.retry_failed is True


class TestParseArgsVersion:
    """--version flag."""

    def test_version_flag(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_version_output(self) -> None:
        output = StringIO()
        with patch.object(sys, "stdout", output), pytest.raises(SystemExit):
            _parse_args(["--version"])
        assert __version__ in output.getvalue()


# ---------------------------------------------------------------------------
# main(): mutually exclusive --quants and --quants-except
# ---------------------------------------------------------------------------

class TestMutuallyExclusiveFlags:
    """--quants and --quants-except cannot be used together."""

    def test_both_provided_returns_error(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--quants", "Q4_K_M", "--quants-except", "Q8_0"])
        assert code == 1
        assert "mutually exclusive" in capsys.readouterr().err

    def test_both_with_spaces(self, capsys: pytest.CaptureFixture) -> None:
        """Test with extra whitespace in quant lists."""
        code = main(["author/model", "--quants", "Q4_K_M, Q8_0", "--quants-except", "BF16"])
        assert code == 1
        assert "mutually exclusive" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main(): --n-samples validation
# ---------------------------------------------------------------------------

class TestNSamplesValidation:
    """--n-samples must be >= 1."""

    def test_zero_samples(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--n-samples", "0"])
        assert code == 1
        assert "must be >= 1" in capsys.readouterr().err

    def test_negative_samples(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--n-samples", "-1"])
        assert code == 1
        assert "must be >= 1" in capsys.readouterr().err

    def test_default_one_is_valid(self) -> None:
        """Default --n-samples of 1 passes parsing."""
        args = _parse_args(["author/model"])
        assert args.n_samples == 1

    def test_large_samples(self) -> None:
        args = _parse_args(["author/model", "--n-samples", "100"])
        assert args.n_samples == 100


# ---------------------------------------------------------------------------
# main(): --pass-at-k validation
# ---------------------------------------------------------------------------

class TestPassAtKValidation:
    """--pass-at-k validation: format, range, and n-samples constraint."""

    def test_non_integer(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--pass-at-k", "1,abc"])
        assert code == 1
        assert "comma-separated integers" in capsys.readouterr().err

    def test_float_values(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--pass-at-k", "1.0,2.0"])
        assert code == 1
        assert "comma-separated integers" in capsys.readouterr().err

    def test_empty_string(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--pass-at-k", ","])
        assert code == 1

    def test_zero_value(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--pass-at-k", "0"])
        assert code == 1
        assert "must be >= 1" in capsys.readouterr().err

    def test_negative_value(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--pass-at-k", "1,-1"])
        assert code == 1
        assert "must be >= 1" in capsys.readouterr().err

    def test_exceeds_n_samples(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--n-samples", "5", "--pass-at-k", "1,10"])
        assert code == 1
        assert "requires --n-samples" in capsys.readouterr().err

    def test_exceeds_default_n_samples(self, capsys: pytest.CaptureFixture) -> None:
        """Default --n-samples is 1, so pass@k > 1 should fail."""
        code = main(["author/model", "--pass-at-k", "10"])
        assert code == 1
        assert "requires --n-samples" in capsys.readouterr().err

    def test_valid_single_k(self) -> None:
        """pass@k=1 with default n_samples=1 should parse OK."""
        args = _parse_args(["author/model", "--pass-at-k", "1"])
        assert args.pass_at_k == "1"

    def test_valid_multiple_k(self) -> None:
        args = _parse_args(["author/model", "--pass-at-k", "1,10,100"])
        assert args.pass_at_k == "1,10,100"

    def test_valid_with_matching_n_samples(self) -> None:
        """k=10 with n-samples=10 should be accepted by arg parsing."""
        args = _parse_args(["author/model", "--n-samples", "10", "--pass-at-k", "1,10"])
        assert args.pass_at_k == "1,10"
        assert args.n_samples == 10

    def test_whitespace_trimming(self) -> None:
        """Whitespace around values should be trimmed during parsing."""
        args = _parse_args(["author/model", "--pass-at-k", "1, 10, 100"])
        assert args.pass_at_k == "1, 10, 100"


# ---------------------------------------------------------------------------
# _run_key generation
# ---------------------------------------------------------------------------

class TestRunKey:
    """_run_key generates deterministic keys from benchmark parameters."""

    def test_default_values(self) -> None:
        args = _parse_args(["author/model"])
        key = _run_key(args)
        assert key == "None:1:0.0:None:None:32000:all:42"

    def test_custom_values(self) -> None:
        args = _parse_args(
            [
                "author/model",
                "--limit",
                "5",
                "--n-samples",
                "10",
                "--temperature",
                "0.5",
                "--pass-at-k",
                "1,10",
                "--ctx-size",
                "8192",
            ]
        )
        key = _run_key(args)
        assert key == "5:10:0.5:1,10:8192:32000:all:42"

    def test_deterministic(self) -> None:
        """Same arguments produce the same key."""
        args1 = _parse_args(["author/model", "--n-samples", "10"])
        args2 = _parse_args(["different/model", "--n-samples", "10"])
        assert _run_key(args1) == _run_key(args2)

    def test_different_n_samples_changes_key(self) -> None:
        args1 = _parse_args(["author/model", "--n-samples", "1"])
        args2 = _parse_args(["author/model", "--n-samples", "10"])
        assert _run_key(args1) != _run_key(args2)

    def test_different_limit_changes_key(self) -> None:
        args1 = _parse_args(["author/model", "--limit", "5"])
        args2 = _parse_args(["author/model", "--limit", "10"])
        assert _run_key(args1) != _run_key(args2)

    def test_different_temperature_changes_key(self) -> None:
        args1 = _parse_args(["author/model", "--temperature", "0.0"])
        args2 = _parse_args(["author/model", "--temperature", "0.5"])
        assert _run_key(args1) != _run_key(args2)

    def test_different_pass_at_k_changes_key(self) -> None:
        args1 = _parse_args(["author/model", "--pass-at-k", "1"])
        args2 = _parse_args(["author/model", "--pass-at-k", "1,10"])
        assert _run_key(args1) != _run_key(args2)

    def test_different_ctx_size_changes_key(self) -> None:
        args1 = _parse_args(["author/model", "--ctx-size", "4096"])
        args2 = _parse_args(["author/model", "--ctx-size", "8192"])
        assert _run_key(args1) != _run_key(args2)

    def test_repo_id_not_included(self) -> None:
        """repo_id should not affect the run key."""
        args1 = _parse_args(["author/model", "--n-samples", "5"])
        args2 = _parse_args(["other/model", "--n-samples", "5"])
        assert _run_key(args1) == _run_key(args2)

    def test_quants_not_included(self) -> None:
        """--quants flag should not affect the run key."""
        args1 = _parse_args(["author/model", "--quants", "Q4_K_M"])
        args2 = _parse_args(["author/model"])
        assert _run_key(args1) == _run_key(args2)

    def test_format_is_colon_separated(self) -> None:
        args = _parse_args(["author/model"])
        key = _run_key(args)
        parts = key.split(":")
        assert len(parts) == 8  # limit:n_samples:temperature:pass_at_k:ctx_size:max_tokens:gpu_layers:seed
        # (pass_at_k may be a comma-joined string, so this count is exact only
        # for the default args where pass_at_k is None)

    def test_different_gpu_layers_changes_key(self) -> None:
        """Offloading changes numerics, so cached results must not be reused."""
        args1 = _parse_args(["author/model", "--gpu-layers", "all"])
        args2 = _parse_args(["author/model", "--gpu-layers", "32"])
        assert _run_key(args1) != _run_key(args2)

    def test_different_seed_changes_key(self) -> None:
        args1 = _parse_args(["author/model", "--seed", "1"])
        args2 = _parse_args(["author/model", "--seed", "2"])
        assert _run_key(args1) != _run_key(args2)

    def test_parallel_not_in_key(self) -> None:
        """Parallelism changes throughput, never accuracy: it must not invalidate the cache."""
        args1 = _parse_args(["author/model", "--parallel", "1"])
        args2 = _parse_args(["author/model", "--parallel", "8"])
        assert _run_key(args1) == _run_key(args2)


# ---------------------------------------------------------------------------
# main(): no quants found
# ---------------------------------------------------------------------------

class TestNoQuantsFound:
    """Error when discover_quants returns empty list."""

    def test_no_quants_error(self, capsys: pytest.CaptureFixture) -> None:
        with patch("quantbench.cli.discover_quants_with_sizes", return_value=([], {})):
            code = main(["author/model"])
        assert code == 1
        assert "no GGUF quants found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main(): --list-quants mode
# ---------------------------------------------------------------------------

class TestListQuants:
    """--list-quants prints quants and exits."""

    def test_list_quants_output(self, capsys: pytest.CaptureFixture) -> None:
        from quantbench.discovery import Quant

        mock_quants = [
            Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",)),
            Quant(name="Q8_0", files=("model-Q8_0-00001-of-00002.gguf", "model-Q8_0-00002-of-00002.gguf")),
        ]
        with patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})):
            code = main(["author/model", "--list-quants"])
        assert code == 0
        output = capsys.readouterr().out
        assert "Q4_K_M" in output
        assert "Q8_0" in output
        assert "model-Q4_K_M.gguf" in output


# ---------------------------------------------------------------------------
# main(): unknown quant validation
# ---------------------------------------------------------------------------

class TestUnknownQuantValidation:
    """Error when --quants references non-existent quants."""

    def test_unknown_quant_in_quants(self, capsys: pytest.CaptureFixture) -> None:
        from quantbench.discovery import Quant

        mock_quants = [Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",))]
        with patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})):
            code = main(["author/model", "--quants", "Q8_0"])
        assert code == 1
        err = capsys.readouterr().err
        assert "unknown quant" in err

    def test_unknown_quant_in_quants_except(self, capsys: pytest.CaptureFixture) -> None:
        from quantbench.discovery import Quant

        mock_quants = [Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",))]
        with patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})):
            code = main(["author/model", "--quants-except", "Q8_0"])
        assert code == 1
        err = capsys.readouterr().err
        assert "unknown quant" in err


# ---------------------------------------------------------------------------
# main(): quant selection filtering
# ---------------------------------------------------------------------------

class TestQuantSelection:
    """--quants and --quants-except correctly filter the quant list."""

    def test_quants_filter(self) -> None:
        from quantbench.discovery import Quant

        mock_quants = [
            Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",)),
            Quant(name="Q8_0", files=("model-Q8_0.gguf",)),
        ]
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli.create_display") as mock_display,
            patch("quantbench.cli.run_pipeline", return_value=[]),
            patch("quantbench.cli._load_existing_outcomes", return_value={}),
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            mock_display.return_value.__enter__ = MagicMock(return_value=None)
            mock_display.return_value.__exit__ = MagicMock(return_value=None)
            main(["author/model", "--quants", "Q4_K_M"])

    def test_quants_except_filter(self) -> None:
        from quantbench.discovery import Quant

        mock_quants = [
            Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",)),
            Quant(name="Q8_0", files=("model-Q8_0.gguf",)),
        ]
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli.create_display") as mock_display,
            patch("quantbench.cli.run_pipeline", return_value=[]),
            patch("quantbench.cli._load_existing_outcomes", return_value={}),
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            mock_display.return_value.__enter__ = MagicMock(return_value=None)
            mock_display.return_value.__exit__ = MagicMock(return_value=None)
            main(["author/model", "--quants-except", "Q8_0"])


# ---------------------------------------------------------------------------
# main(): resume detection
# ---------------------------------------------------------------------------

class TestResumeDetection:
    """Resume logic: existing outcomes are loaded and remaining quants are run."""

    def test_no_existing_outcomes(self) -> None:
        from quantbench.discovery import Quant

        mock_quants = [Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",))]
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli._load_existing_outcomes", return_value={}),
            patch("quantbench.cli.create_display") as mock_display,
            patch("quantbench.cli.run_pipeline") as mock_run,
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            mock_display.return_value.__enter__ = MagicMock(return_value=None)
            mock_display.return_value.__exit__ = MagicMock(return_value=None)
            mock_run.return_value = []
            main(["author/model"])
            mock_run.assert_called_once()
            args, _ = mock_run.call_args
            passed_quants = args[1]  # remaining_quants is positional arg #2
            assert len(passed_quants) == 1

    def test_resume_skips_completed(self) -> None:
        from quantbench.discovery import Quant
        from quantbench.orchestrator import QuantOutcome

        mock_quants = [
            Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",)),
            Quant(name="Q8_0", files=("model-Q8_0.gguf",)),
        ]
        existing = {"Q4_K_M": QuantOutcome("Q4_K_M", 1000, None, None)}
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli._load_existing_outcomes", return_value=existing),
            patch("quantbench.cli.create_display") as mock_display,
            patch("quantbench.cli.run_pipeline") as mock_run,
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            mock_display.return_value.__enter__ = MagicMock(return_value=None)
            mock_display.return_value.__exit__ = MagicMock(return_value=None)
            mock_run.return_value = []
            main(["author/model"])
            args, _ = mock_run.call_args
            passed_quants = args[1]  # remaining_quants is positional arg #2
            assert len(passed_quants) == 1
            assert passed_quants[0].name == "Q8_0"


# ---------------------------------------------------------------------------
# main(): --parallel validation
# ---------------------------------------------------------------------------

class TestParallelValidation:
    """--parallel must be >= 1."""

    def test_zero_parallel(self, capsys: pytest.CaptureFixture) -> None:
        code = main(["author/model", "--parallel", "0"])
        assert code == 1
        assert "must be >= 1" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main(): exit codes
# ---------------------------------------------------------------------------

@contextmanager
def _patch_full_run(mock_quants, new_outcomes):
    """Patch everything main() touches after argument validation."""
    with (
        patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
        patch("quantbench.cli._load_existing_outcomes", return_value={}),
        patch("quantbench.cli.create_display"),
        patch("quantbench.cli.run_pipeline", return_value=new_outcomes),
        patch("quantbench.cli.write_csv"),
        patch("quantbench.cli.write_chart"),
        patch.object(Path, "mkdir"),
        patch.object(Path, "exists", return_value=False),
    ):
        yield


class TestExitCodes:
    """main() returns 0 only when every selected quant succeeded."""

    def test_all_success_returns_zero(self, capsys: pytest.CaptureFixture) -> None:
        from quantbench.discovery import Quant
        from quantbench.eval_runner import EvalResult
        from quantbench.orchestrator import QuantOutcome

        mock_quants = [Quant(name="Q4_K_M", files=("model.gguf",))]
        new_outcomes = [
            QuantOutcome("Q4_K_M", 1000, EvalResult(base_pass1=0.5, extra_pass1=0.4, n_problems=100), None)
        ]
        with _patch_full_run(mock_quants, new_outcomes):
            code = main(["author/model"])
        assert code == 0
        assert "1/1 quants benchmarked successfully" in capsys.readouterr().out

    def test_partial_failure_returns_one(self, capsys: pytest.CaptureFixture) -> None:
        from quantbench.discovery import Quant
        from quantbench.eval_runner import EvalResult
        from quantbench.orchestrator import QuantOutcome

        mock_quants = [
            Quant(name="Q4_K_M", files=("a.gguf",)),
            Quant(name="Q8_0", files=("b.gguf",)),
        ]
        new_outcomes = [
            QuantOutcome("Q4_K_M", 1000, EvalResult(base_pass1=0.5, extra_pass1=0.4, n_problems=100), None),
            QuantOutcome("Q8_0", 2000, None, error="server crashed"),
        ]
        with _patch_full_run(mock_quants, new_outcomes):
            code = main(["author/model"])
        err = capsys.readouterr().err
        assert code == 1
        assert "1 quant(s) failed" in err
        assert "--retry-failed" in err

    def test_failure_in_existing_outcome_counts(self, capsys: pytest.CaptureFixture) -> None:
        """Cached error outcomes from a previous run also make the exit code 1."""
        from quantbench.discovery import Quant
        from quantbench.orchestrator import QuantOutcome

        mock_quants = [Quant(name="Q4_K_M", files=("a.gguf",))]
        existing = {"Q4_K_M": QuantOutcome("Q4_K_M", 1000, None, error="old failure")}
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli._load_existing_outcomes", return_value=existing),
            patch("quantbench.cli.create_display"),
            patch("quantbench.cli.run_pipeline", return_value=[]),
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            code = main(["author/model"])
        err = capsys.readouterr().err
        assert code == 1
        assert "1 quant(s) failed" in err


# ---------------------------------------------------------------------------
# main(): --retry-failed
# ---------------------------------------------------------------------------

class TestRetryFailed:
    """--retry-failed drops cached error outcomes so those quants are re-run."""

    def _existing(self):
        from quantbench.eval_runner import EvalResult
        from quantbench.orchestrator import QuantOutcome

        return {
            "Q4_K_M": QuantOutcome("Q4_K_M", 1000, None, error="old failure"),
            "Q8_0": QuantOutcome(
                "Q8_0", 2000,
                EvalResult(base_pass1=0.6, extra_pass1=0.5, n_problems=100), None,
            ),
        }

    def test_retry_failed_reruns_only_errors(self) -> None:
        from quantbench.discovery import Quant

        mock_quants = [
            Quant(name="Q4_K_M", files=("a.gguf",)),
            Quant(name="Q8_0", files=("b.gguf",)),
        ]
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli._load_existing_outcomes", return_value=self._existing()),
            patch("quantbench.cli.create_display"),
            patch("quantbench.cli.run_pipeline") as mock_run,
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            mock_run.return_value = []
            main(["author/model", "--retry-failed"])
            passed_quants = mock_run.call_args.args[1]
            assert [q.name for q in passed_quants] == ["Q4_K_M"]

    def test_without_flag_errors_are_cached(self) -> None:
        from quantbench.discovery import Quant

        mock_quants = [
            Quant(name="Q4_K_M", files=("a.gguf",)),
            Quant(name="Q8_0", files=("b.gguf",)),
        ]
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli._load_existing_outcomes", return_value=self._existing()),
            patch("quantbench.cli.create_display"),
            patch("quantbench.cli.run_pipeline") as mock_run,
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            mock_run.return_value = []
            main(["author/model"])
            passed_quants = mock_run.call_args.args[1]
            assert passed_quants == []


# ---------------------------------------------------------------------------
# main(): greedy pass@k warning
# ---------------------------------------------------------------------------

class TestGreedyPassAtKWarning:
    """pass@k > 1 with temperature 0 warns that samples will be identical."""

    def test_warning_printed(self, capsys: pytest.CaptureFixture) -> None:
        from quantbench.discovery import Quant

        mock_quants = [Quant(name="Q4_K_M", files=("a.gguf",))]
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli._load_existing_outcomes", return_value={}),
            patch("quantbench.cli.create_display"),
            patch("quantbench.cli.run_pipeline", return_value=[]),
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            code = main(["author/model", "--n-samples", "5", "--pass-at-k", "1,5"])
        assert code == 0
        err = capsys.readouterr().err
        assert "identical" in err

    def test_no_warning_with_temperature(self, capsys: pytest.CaptureFixture) -> None:
        from quantbench.discovery import Quant

        mock_quants = [Quant(name="Q4_K_M", files=("a.gguf",))]
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli._load_existing_outcomes", return_value={}),
            patch("quantbench.cli.create_display"),
            patch("quantbench.cli.run_pipeline", return_value=[]),
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            main(["author/model", "--n-samples", "5", "--pass-at-k", "1,5", "--temperature", "0.7"])
        assert "identical" not in capsys.readouterr().err

    def test_no_warning_for_k1_only(self, capsys: pytest.CaptureFixture) -> None:
        from quantbench.discovery import Quant

        mock_quants = [Quant(name="Q4_K_M", files=("a.gguf",))]
        with (
            patch("quantbench.cli.discover_quants_with_sizes", return_value=(mock_quants, {})),
            patch("quantbench.cli._load_existing_outcomes", return_value={}),
            patch("quantbench.cli.create_display"),
            patch("quantbench.cli.run_pipeline", return_value=[]),
            patch("quantbench.cli.write_csv"),
            patch("quantbench.cli.write_chart"),
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", return_value=False),
        ):
            main(["author/model", "--pass-at-k", "1"])
        assert "identical" not in capsys.readouterr().err
