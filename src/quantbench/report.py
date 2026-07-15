"""CSV + bar chart output for a completed quantbench run."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantbench.orchestrator import QuantOutcome

# Single-hue categorical slot 1 + chart chrome from the project's validated
# default palette (references/palette.md in the dataviz skill).
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SURFACE = "#fcfcfb"
_BAR = "#2a78d6"


def write_csv(outcomes: list[QuantOutcome], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quant", "base_pass1", "extra_pass1", "n_problems", "size_gb", "error"])
        for o in outcomes:
            writer.writerow(
                [
                    o.quant_name,
                    f"{o.result.base_pass1:.4f}" if o.result else "",
                    f"{o.result.extra_pass1:.4f}" if o.result else "",
                    o.result.n_problems if o.result else "",
                    f"{o.size_bytes / 1e9:.3f}",
                    o.error or "",
                ]
            )


def write_chart(outcomes: list[QuantOutcome], path: Path, *, repo_id: str) -> None:
    """Bar chart of HumanEval+ pass@1 per quant, sorted best to worst.

    Quants that failed to download/benchmark are left out of the chart (a
    zero-height bar would be indistinguishable from a real 0.0 score) but are
    still present in the CSV and noted in the title if any occurred.
    """
    scored = sorted(
        (o for o in outcomes if o.result is not None),
        key=lambda o: o.result.extra_pass1,
        reverse=True,
    )
    n_failed = len(outcomes) - len(scored)
    if not scored:
        print("warning: no successful quants to chart, skipping report.png")
        return

    names = [o.quant_name for o in scored]
    values = [o.result.extra_pass1 for o in scored]

    fig_width = max(6.0, 0.5 * len(names) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    bars = ax.bar(names, values, color=_BAR, width=0.6, zorder=3)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("HumanEval+ pass@1", color=_INK_SECONDARY, fontsize=10)
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

    ax.tick_params(axis="x", colors=_INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", colors=_INK_MUTED, labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=_INK_SECONDARY,
        )

    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
