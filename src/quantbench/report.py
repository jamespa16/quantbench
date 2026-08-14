"""CSV + bar chart output for a completed quantbench run."""

from __future__ import annotations

import csv
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from quantbench.eval_runner import PassAtKStats
from quantbench.orchestrator import QuantOutcome

# Single-hue categorical slot 1 + chart chrome from the project's validated
# default palette.
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SURFACE = "#fcfcfb"
_BAR = "#2a78d6"


def _pairwise_p_values(
    outcomes: list[QuantOutcome], k: int
) -> dict[tuple[str, str], float]:
    """Compute Welch's t-test p-values for all pairs of scored quants at a given k."""
    scored = [(o.quant_name, o.result.pass_at_k_per_task[k])
              for o in outcomes
              if o.result and o.result.pass_at_k_per_task and k in o.result.pass_at_k_per_task
              and len(o.result.pass_at_k_per_task[k]) >= 2]
    p_map: dict[tuple[str, str], float] = {}
    for (a, scores_a), (b, scores_b) in combinations(scored, 2):
        _t_stat, p_val = stats.ttest_ind(scores_a, scores_b, equal_var=False)
        p_map[(a, b)] = p_val
    return p_map


def write_csv(outcomes: list[QuantOutcome], path: Path) -> None:
    all_k: list[int] = []
    for o in outcomes:
        if o.result and o.result.pass_at_k:
            for k in o.result.pass_at_k:
                if k not in all_k:
                    all_k.append(k)
    all_k.sort()

    # Determine primary k for p-value column (smallest k requested, or 1 if none).
    primary_k = all_k[0] if all_k else None
    has_stats = primary_k is not None and any(
        o.result and o.result.pass_at_k_per_task and primary_k in o.result.pass_at_k_per_task
        for o in outcomes
    )
    assert primary_k is not None if has_stats else True
    p_map = _pairwise_p_values(outcomes, primary_k) if has_stats and primary_k is not None else {}

    # Best quant for "vs best" p-value column.
    best_name: str | None = None
    if all_k:
        first_k = all_k[0]
        scored = [
            (o, o.result.pass_at_k[first_k])
            for o in outcomes
            if o.result and o.result.pass_at_k and first_k in o.result.pass_at_k
        ]
        if scored:
            best_name = max(scored, key=lambda x: x[1])[0].quant_name

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["quant", "base_pass1", "extra_pass1", "n_problems", "size_gb", "error"]
        for k in all_k:
            header += [f"pass@{k}", f"pass@{k}_std", f"pass@{k}_stderr"]
        if has_stats:
            assert primary_k is not None
            header.append(f"p_value_vs_best@{primary_k}")
        writer.writerow(header)
        for o in outcomes:
            result = o.result
            row = [
                o.quant_name,
                f"{result.base_pass1:.4f}" if result else "",
                f"{result.extra_pass1:.4f}" if result else "",
                result.n_problems if result else "",
                f"{o.size_bytes / 1e9:.3f}",
                o.error or "",
            ]
            for k in all_k:
                if result and result.pass_at_k and k in result.pass_at_k:
                    row.append(f"{result.pass_at_k[k]:.4f}")
                else:
                    row.append("")
                st: PassAtKStats | None = (
                    result.pass_at_k_stats[k]
                    if result and result.pass_at_k_stats and k in result.pass_at_k_stats
                    else None
                )
                row.append(f"{st.std_dev:.4f}" if st else "")
                row.append(f"{st.std_error:.4f}" if st else "")
            if has_stats and best_name is not None and o.quant_name != best_name:
                pair: tuple[str, str] = (
                    (best_name, o.quant_name)
                    if best_name < o.quant_name
                    else (o.quant_name, best_name)
                )
                row.append(f"{p_map.get(pair, float('nan')):.4f}")
            elif has_stats:
                row.append("")
            writer.writerow(row)


def _sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def write_chart(outcomes: list[QuantOutcome], path: Path, *, repo_id: str) -> None:
    """Bar chart of pass@k per quant, sorted best to worst, with error bars and significance.

    When pass@k stats are available, charts the primary k value with ±1 SE error
    bars and significance markers (Welch's t-test vs the best quant).
    Falls back to extra_pass1 when no pass@k was requested.
    """
    has_pass_at_k = any(
        o.result and o.result.pass_at_k for o in outcomes
    )

    p_map: dict[tuple[str, str], float] = {}
    best_name: str | None = None
    sig_markers: list[str] = []

    if has_pass_at_k:
        all_k = sorted({
            k for o in outcomes
            if o.result and o.result.pass_at_k
            for k in o.result.pass_at_k
        })
        primary_k = all_k[0]
        scored_list = [
            (o, o.result.pass_at_k[primary_k])
            for o in outcomes
            if o.result and o.result.pass_at_k and primary_k in o.result.pass_at_k
        ]
        scored = sorted(scored_list, key=lambda x: x[1], reverse=True)
        names = [o.quant_name for o, _ in scored]
        values = [v for _, v in scored]
        stderrs = [
            o.result.pass_at_k_stats[primary_k].std_error
            if o.result is not None and o.result.pass_at_k_stats
               and primary_k in o.result.pass_at_k_stats
            else 0.0
            for o, _ in scored
        ]
        ylabel = f"HumanEval+ pass@{primary_k}"

        # Compute significance vs best.
        p_map = _pairwise_p_values(outcomes, primary_k)
        best_name = scored[0][0].quant_name if scored else None
        sig_markers = []
        if best_name is not None:
            for o, _ in scored:
                if o.quant_name == best_name:
                    sig_markers.append("")
                else:
                    pair: tuple[str, str] = (
                        (best_name, o.quant_name)
                        if best_name < o.quant_name
                        else (o.quant_name, best_name)
                    )
                    p = p_map.get(pair)
                    sig_markers.append(_sig_marker(p) if p is not None else "")
        else:
            sig_markers = []
    else:
        scored_list = [
            (o, o.result.extra_pass1)
            for o in outcomes
            if o.result is not None
        ]
        scored = sorted(scored_list, key=lambda x: x[1], reverse=True)
        names = [o.quant_name for o, _ in scored]
        values = [v for _, v in scored]
        stderrs = [0.0] * len(scored)
        ylabel = "HumanEval+ pass@1"

    n_failed = len(outcomes) - len(scored)
    if not scored:
        print("warning: no successful quants to chart, skipping report.png")
        return

    fig_width = max(6.0, 0.5 * len(names) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    x_positions = range(len(names))
    ax.bar(
        x_positions, values,
        yerr=stderrs if any(stderrs) else None,
        capsize=4,
        color=_BAR,
        width=0.6,
        zorder=3,
        ecolor=_INK_MUTED,
        errhatches=None,
    )

    ax.set_ylim(0, min(1.05, max(values) + 0.1 * max(values) + 0.05))
    ax.set_ylabel(ylabel, color=_INK_SECONDARY, fontsize=10)
    title = f"Coding benchmark by quantization — {repo_id}"
    if n_failed:
        title += f" ({n_failed} quant{'s' if n_failed != 1 else ''} failed, not shown)"
    wrap_width = max(30, int(fig_width * 8.5))
    ax.set_title(textwrap.fill(title, width=wrap_width), color=_INK, fontsize=12, pad=14)

    ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", colors=_INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", colors=_INK_MUTED, labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i, (pos, value, stderr, marker) in enumerate(zip(x_positions, values, stderrs, sig_markers, strict=False)):
        label_y = value + stderr + 0.02
        text_parts = [f"{value:.2f}"]
        if marker and best_name is not None:
            chart_pair: tuple[str, str] = (
                (best_name, names[i])
                if best_name < names[i]
                else (names[i], best_name)
            )
            text_parts.append(f"{marker} ({p_map.get(chart_pair, 1):.3f})")
        ax.text(
            pos, label_y, " ".join(text_parts),
            ha="center", va="bottom", fontsize=7, color=_INK_SECONDARY,
        )

    # Legend for significance markers.
    if any(sig_markers):
        legend_text = ("*** p<0.001  ** p<0.01  * p<0.05  "
                       "n.s. p≥0.05  (vs best, Welch's t-test)")
        ax.text(
            0.5, -0.18, legend_text,
            ha="center", va="top", transform=ax.transAxes,
            fontsize=7, color=_INK_MUTED,
        )

    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
