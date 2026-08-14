"""Tests for discovery.py: quant token regex, grouping logic, shard ordering, and file sizes."""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest
from huggingface_hub import ModelInfo

from quantbench.discovery import Quant, _match_quant_token, _shard_index, discover_quants, fetch_file_sizes

# ---------------------------------------------------------------------------
# _match_quant_token
# ---------------------------------------------------------------------------

class TestMatchQuantTokenStandard:
    """Standard GGML quants: Q4_K_M, Q8_0, BF16, F16, F32."""

    def test_q4_k_m(self) -> None:
        assert _match_quant_token("model-Q4_K_M.gguf") == "Q4_K_M"

    def test_q8_0(self) -> None:
        assert _match_quant_token("model-Q8_0.gguf") == "Q8_0"

    def test_q2_k(self) -> None:
        assert _match_quant_token("model-Q2_K.gguf") == "Q2_K"

    def test_q2_k_s(self) -> None:
        assert _match_quant_token("model-Q2_K_S.gguf") == "Q2_K_S"

    def test_q3_k_l(self) -> None:
        assert _match_quant_token("model-Q3_K_L.gguf") == "Q3_K_L"

    def test_q5_k_xl(self) -> None:
        assert _match_quant_token("model-Q5_K_XL.gguf") == "Q5_K_XL"

    def test_q6_k(self) -> None:
        assert _match_quant_token("model-Q6_K.gguf") == "Q6_K"

    def test_q5_0(self) -> None:
        assert _match_quant_token("model-Q5_0.gguf") == "Q5_0"

    def test_q5_1(self) -> None:
        assert _match_quant_token("model-Q5_1.gguf") == "Q5_1"

    def test_bf16(self) -> None:
        assert _match_quant_token("model-BF16.gguf") == "BF16"

    def test_f16(self) -> None:
        assert _match_quant_token("model-F16.gguf") == "F16"

    def test_f32(self) -> None:
        assert _match_quant_token("model-F32.gguf") == "F32"


class TestMatchQuantTokenIquant:
    """IQ quantizations: IQ1_XXS through IQ4_NL."""

    def test_iq1_xxs(self) -> None:
        assert _match_quant_token("model-IQ1_XXS.gguf") == "IQ1_XXS"

    def test_iq1_xs(self) -> None:
        assert _match_quant_token("model-IQ1_XS.gguf") == "IQ1_XS"

    def test_iq1_s(self) -> None:
        assert _match_quant_token("model-IQ1_S.gguf") == "IQ1_S"

    def test_iq1_m(self) -> None:
        assert _match_quant_token("model-IQ1_M.gguf") == "IQ1_M"

    def test_iq1_l(self) -> None:
        assert _match_quant_token("model-IQ1_L.gguf") == "IQ1_L"

    def test_iq1_nl(self) -> None:
        assert _match_quant_token("model-IQ1_NL.gguf") == "IQ1_NL"

    def test_iq2_xxs(self) -> None:
        assert _match_quant_token("model-IQ2_XXS.gguf") == "IQ2_XXS"

    def test_iq2_xs(self) -> None:
        assert _match_quant_token("model-IQ2_XS.gguf") == "IQ2_XS"

    def test_iq2_s(self) -> None:
        assert _match_quant_token("model-IQ2_S.gguf") == "IQ2_S"

    def test_iq2_m(self) -> None:
        assert _match_quant_token("model-IQ2_M.gguf") == "IQ2_M"

    def test_iq2_l(self) -> None:
        assert _match_quant_token("model-IQ2_L.gguf") == "IQ2_L"

    def test_iq2_nl(self) -> None:
        assert _match_quant_token("model-IQ2_NL.gguf") == "IQ2_NL"

    def test_iq3_xxs(self) -> None:
        assert _match_quant_token("model-IQ3_XXS.gguf") == "IQ3_XXS"

    def test_iq3_xs(self) -> None:
        assert _match_quant_token("model-IQ3_XS.gguf") == "IQ3_XS"

    def test_iq3_s(self) -> None:
        assert _match_quant_token("model-IQ3_S.gguf") == "IQ3_S"

    def test_iq3_m(self) -> None:
        assert _match_quant_token("model-IQ3_M.gguf") == "IQ3_M"

    def test_iq3_l(self) -> None:
        assert _match_quant_token("model-IQ3_L.gguf") == "IQ3_L"

    def test_iq3_nl(self) -> None:
        assert _match_quant_token("model-IQ3_NL.gguf") == "IQ3_NL"

    def test_iq4_xxs(self) -> None:
        assert _match_quant_token("model-IQ4_XXS.gguf") == "IQ4_XXS"

    def test_iq4_xs(self) -> None:
        assert _match_quant_token("model-IQ4_XS.gguf") == "IQ4_XS"

    def test_iq4_s(self) -> None:
        assert _match_quant_token("model-IQ4_S.gguf") == "IQ4_S"

    def test_iq4_m(self) -> None:
        assert _match_quant_token("model-IQ4_M.gguf") == "IQ4_M"

    def test_iq4_l(self) -> None:
        assert _match_quant_token("model-IQ4_L.gguf") == "IQ4_L"

    def test_iq4_nl(self) -> None:
        assert _match_quant_token("model-IQ4_NL.gguf") == "IQ4_NL"


class TestMatchQuantTokenUd:
    """UD- prefix variants."""

    def test_ud_q4_k_m(self) -> None:
        assert _match_quant_token("model-UD-Q4_K_M.gguf") == "UD-Q4_K_M"

    def test_ud_q8_0(self) -> None:
        assert _match_quant_token("model-UD-Q8_0.gguf") == "UD-Q8_0"

    def test_ud_q2_k(self) -> None:
        assert _match_quant_token("model-UD-Q2_K.gguf") == "UD-Q2_K"

    def test_ud_iq2_xs(self) -> None:
        assert _match_quant_token("model-UD-IQ2_XS.gguf") == "UD-IQ2_XS"

    def test_ud_f16(self) -> None:
        assert _match_quant_token("model-UD-F16.gguf") == "UD-F16"


class TestMatchQuantTokenEdgeCases:
    """Edge cases: shard suffixes, separators, underscores."""

    def test_shard_file_q4_k_m(self) -> None:
        assert _match_quant_token("model-Q4_K_M-00001-of-00004.gguf") == "Q4_K_M"

    def test_underscore_separator(self) -> None:
        assert _match_quant_token("model_Q4_K_M.gguf") == "Q4_K_M"

    def test_dot_separator(self) -> None:
        assert _match_quant_token("model.Q8_0.gguf") == "Q8_0"

    def test_leading_quant(self) -> None:
        assert _match_quant_token("Q6_K.gguf") == "Q6_K"

    def test_case_insensitive_shard(self) -> None:
        assert _match_quant_token("model-Q4_K_M-00001-OF-00004.GGUF") == "Q4_K_M"


class TestMatchQuantTokenRejectsUnknown:
    """Unknown tokens must return None."""

    def test_unknown_prefix(self) -> None:
        assert _match_quant_token("model-XYZ_99.gguf") is None

    def test_random_string(self) -> None:
        assert _match_quant_token("model-foobar.gguf") is None

    def test_q1_invalid(self) -> None:
        assert _match_quant_token("model-Q1_K_M.gguf") is None

    def test_q9_invalid(self) -> None:
        assert _match_quant_token("model-Q9_0.gguf") is None

    def test_iq5_invalid(self) -> None:
        assert _match_quant_token("model-IQ5_M.gguf") is None

    def test_no_quant_token(self) -> None:
        assert _match_quant_token("just-a-name.gguf") is None

    def test_empty(self) -> None:
        assert _match_quant_token("") is None


# ---------------------------------------------------------------------------
# _shard_index
# ---------------------------------------------------------------------------

class TestShardIndex:
    """_shard_index returns correct shard ordering."""

    def test_single_file_returns_zero(self) -> None:
        assert _shard_index("model-Q4_K_M.gguf") == 0

    def test_first_shard(self) -> None:
        assert _shard_index("model-Q4_K_M-00001-of-00004.gguf") == 1

    def test_second_shard(self) -> None:
        assert _shard_index("model-Q4_K_M-00002-of-00004.gguf") == 2

    def test_last_shard(self) -> None:
        assert _shard_index("model-Q4_K_M-00004-of-00004.gguf") == 4

    def test_large_shard_numbers(self) -> None:
        assert _shard_index("model-Q8_0-01234-of-01234.gguf") == 1234

    def test_case_insensitive(self) -> None:
        assert _shard_index("model-00001-OF-00002.gguf") == 1

    def test_sorting_order(self) -> None:
        files = [
            "model-Q4_K_M-00003-of-00004.gguf",
            "model-Q4_K_M-00001-of-00004.gguf",
            "model-Q4_K_M-00004-of-00004.gguf",
            "model-Q4_K_M-00002-of-00004.gguf",
        ]
        assert sorted(files, key=_shard_index) == [
            "model-Q4_K_M-00001-of-00004.gguf",
            "model-Q4_K_M-00002-of-00004.gguf",
            "model-Q4_K_M-00003-of-00004.gguf",
            "model-Q4_K_M-00004-of-00004.gguf",
        ]

    def test_mixed_single_and_sharded(self) -> None:
        """Single files (index 0) sort before sharded files (index >= 1)."""
        files = [
            "model-Q8_0-00001-of-00002.gguf",
            "model-Q4_K_M.gguf",
        ]
        assert sorted(files, key=_shard_index) == [
            "model-Q4_K_M.gguf",
            "model-Q8_0-00001-of-00002.gguf",
        ]


# ---------------------------------------------------------------------------
# discover_quants
# ---------------------------------------------------------------------------

class TestDiscoverQuantsSubfolder:
    """Subfolder grouping: files in subfolders are grouped by folder name."""

    def test_subfolder_grouping(self) -> None:
        files = [
            "Q4_K_M/model-Q4_K_M.gguf",
            "Q4_K_M/model-Q4_K_M-00001-of-00002.gguf",
            "Q8_0/model-Q8_0.gguf",
        ]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        names = [q.name for q in quants]
        assert names == ["Q4_K_M", "Q8_0"]

    def test_subfolder_shard_ordering(self) -> None:
        files = [
            "Q4_K_M/model-00003-of-00004.gguf",
            "Q4_K_M/model-00001-of-00004.gguf",
            "Q4_K_M/model-00004-of-00004.gguf",
            "Q4_K_M/model-00002-of-00004.gguf",
        ]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        assert len(quants) == 1
        q = quants[0]
        assert q.name == "Q4_K_M"
        assert q.files == tuple(sorted(files, key=_shard_index))

    def test_subfolder_sorted_names(self) -> None:
        """Quant names are returned sorted alphabetically."""
        files = ["Z_quant/model.gguf", "A_quant/model.gguf"]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        assert [q.name for q in quants] == ["A_quant", "Z_quant"]


class TestDiscoverQuantsFlat:
    """Flat-file grouping: files without subfolders are grouped by parsed quant token."""

    def test_flat_grouping(self) -> None:
        files = [
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "model-IQ2_XS.gguf",
        ]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        names = [q.name for q in quants]
        assert names == ["IQ2_XS", "Q4_K_M", "Q8_0"]

    def test_flat_mixed_shards(self) -> None:
        files = [
            "model-Q4_K_M.gguf",
            "model-Q8_0-00001-of-00002.gguf",
            "model-Q8_0-00002-of-00002.gguf",
        ]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        assert len(quants) == 2
        q8 = next(q for q in quants if q.name == "Q8_0")
        assert len(q8.files) == 2


class TestDiscoverQuantsUnknow:
    """Unrecognizable files are skipped with a warning, not an error."""

    def test_unknown_file_skipped(self, capsys: pytest.CaptureFixture) -> None:
        files = ["model-Q4_K_M.gguf", "model-UNKNOWN.gguf"]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        assert len(quants) == 1
        assert quants[0].name == "Q4_K_M"
        assert "could not determine quant" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# mmproj filtering
# ---------------------------------------------------------------------------

class TestMmprojFilter:
    """mmproj-*.gguf files must be excluded from discovery."""

    def test_mmproj_excluded(self) -> None:
        files = [
            "model-Q4_K_M.gguf",
            "mmproj-BF16.gguf",
        ]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        names = [q.name for q in quants]
        assert names == ["Q4_K_M"]

    def test_mmproj_in_subfolder_excluded(self) -> None:
        files = [
            "model-Q4_K_M.gguf",
            "mmproj/mmproj-F16.gguf",
        ]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        names = [q.name for q in quants]
        assert names == ["Q4_K_M"]

    def test_mmproj_case_insensitive(self) -> None:
        """MMproj with different casing is still excluded."""
        files = [
            "model-Q4_K_M.gguf",
            "MMproj-F16.gguf",
            "MMPROJ-BF16.gguf",
        ]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        names = [q.name for q in quants]
        assert names == ["Q4_K_M"]

    def test_only_mmproj_returns_empty(self) -> None:
        files = ["mmproj-BF16.gguf"]
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.list_repo_files.return_value = files
            quants = discover_quants("author/model")
        assert quants == []


# ---------------------------------------------------------------------------
# fetch_file_sizes
# ---------------------------------------------------------------------------

class TestFetchFileSizes:
    """fetch_file_sizes returns correct sizes from HfApi repo_info."""

    def test_basic(self) -> None:
        mock_info = ModelInfo(
            id="author/model",
            siblings=[
                {"rfilename": "model-Q4_K_M.gguf", "size": 4000},
                {"rfilename": "model-Q8_0.gguf", "size": 8000},
                {"rfilename": "README.md", "size": 100},
            ],
        )
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.repo_info.return_value = mock_info
            sizes = fetch_file_sizes("author/model")
        assert sizes == {
            "model-Q4_K_M.gguf": 4000,
            "model-Q8_0.gguf": 8000,
            "README.md": 100,
        }

    def test_no_siblings(self) -> None:
        mock_info = ModelInfo(
            id="author/model",
            siblings=None,
        )
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.repo_info.return_value = mock_info
            sizes = fetch_file_sizes("author/model")
        assert sizes == {}

    def test_null_size_filtered(self) -> None:
        """Files with size=None are excluded from the result."""
        mock_info = ModelInfo(
            id="author/model",
            siblings=[
                {"rfilename": "model-Q4_K_M.gguf", "size": 4000},
                {"rfilename": "model-Q8_0.gguf", "size": None},
            ],
        )
        with patch("quantbench.discovery.HfApi") as mock_api:
            mock_api.return_value.repo_info.return_value = mock_info
            sizes = fetch_file_sizes("author/model")
        assert sizes == {"model-Q4_K_M.gguf": 4000}


# ---------------------------------------------------------------------------
# Quant dataclass
# ---------------------------------------------------------------------------

class TestQuantDataclass:
    """Quant dataclass properties."""

    def test_primary_file_single(self) -> None:
        q = Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",))
        assert q.primary_file == "model-Q4_K_M.gguf"

    def test_primary_file_sharded(self) -> None:
        q = Quant(
            name="Q4_K_M",
            files=(
                "model-Q4_K_M-00001-of-00004.gguf",
                "model-Q4_K_M-00002-of-00004.gguf",
                "model-Q4_K_M-00003-of-00004.gguf",
                "model-Q4_K_M-00004-of-00004.gguf",
            ),
        )
        assert q.primary_file == "model-Q4_K_M-00001-of-00004.gguf"

    def test_frozen(self) -> None:
        q = Quant(name="Q4_K_M", files=("model-Q4_K_M.gguf",))
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.name = "Q8_0"  # type: ignore
