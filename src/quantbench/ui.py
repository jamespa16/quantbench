"""Terminal dashboard: startup splash, download/benchmark progress, tokens/sec.

All Rich usage is confined to this module -- other modules talk to a
`BenchDisplay` purely through plain callables/methods, never importing Rich
themselves.
"""

from __future__ import annotations

from rich.console import Console
from rich.console import Group as _Group
from rich.live import Live
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TaskID, TextColumn, TransferSpeedColumn
from rich.text import Text
from tqdm import tqdm as _tqdm_base

# --- splash ------------------------------------------------------------

_GLYPH_HEIGHT = 5
_BLOCK = "██"
_FONT: dict[str, tuple[str, ...]] = {
    "Q": (".###.", "#...#", "#...#", "#..#.", ".####"),
    "U": ("#...#", "#...#", "#...#", "#...#", ".###."),
    "A": (".###.", "#...#", "#####", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#"),
    "T": ("#####", "..#..", "..#..", "..#..", "..#.."),
    "B": ("####.", "#...#", "####.", "#...#", "####."),
    "E": ("#####", "#....", "###..", "#....", "#####"),
    "C": (".####", "#....", "#....", "#....", ".####"),
    "H": ("#...#", "#...#", "#####", "#...#", "#...#"),
    " ": ("...", "...", "...", "...", "..."),
}
_FALLBACK: tuple[str, ...] = ("#####", "#####", "#####", "#####", "#####")


def _render_glyph_row(glyph_row: str, block: str) -> str:
    off = " " * len(block)
    return "".join(block if p == "#" else off for p in glyph_row)


def _render_text(text: str, *, block: str, letter_gap: str) -> list[str]:
    glyphs = [_FONT.get(char.upper(), _FALLBACK) for char in text]
    return [
        letter_gap.join(_render_glyph_row(glyph[row], block) for glyph in glyphs)
        for row in range(_GLYPH_HEIGHT)
    ]


def print_banner(console: Console, *, subtitle: str | None = None) -> None:
    # "██" reads best but needs ~118 cols for "QUANTBENCH"; fall back to a
    # single-column block on narrower terminals so rows never wrap (wrapping
    # would break each row at a different point and destroy the alignment).
    wide = _render_text("QUANTBENCH", block=_BLOCK, letter_gap="  ")
    lines = wide if len(wide[0]) <= console.width else _render_text("QUANTBENCH", block="█", letter_gap=" ")

    console.print()
    for line in lines:
        console.print(line, style="bold cyan", no_wrap=True, overflow="crop")
    if subtitle:
        console.print(subtitle, style="dim", justify="left")
    console.print()


# --- download progress: tqdm shim for huggingface_hub -------------------


def make_tqdm_shim(progress: Progress, task_id: TaskID) -> type[_tqdm_base]:
    """A tqdm subclass that forwards byte-count updates into a Rich task.

    huggingface_hub's `snapshot_download` instantiates this once per file it
    downloads (and once more for a non-byte "Fetching N files" overview bar).
    Only byte-unit instances are forwarded, so the overview bar can't
    double-count and corrupt the aggregate percentage.
    """

    class _RichTqdmShim(_tqdm_base):
        def __init__(self, *args, **kwargs):
            self._is_bytes = kwargs.get("unit") == "B"
            kwargs["disable"] = True  # suppress tqdm's own stderr rendering
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> bool | None:
            if self._is_bytes:
                progress.update(task_id, advance=n)
            return super().update(n)

    return _RichTqdmShim


# --- benchmark dashboard --------------------------------------------------


class _RichDisplay:
    def __init__(self, console: Console) -> None:
        self._console = console

        self._quants_progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} quants"),
            console=console,
        )
        self._quants_task = self._quants_progress.add_task("waiting to start...", total=1)

        self._download_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
        )
        self._download_task = self._download_progress.add_task(
            "waiting to download...", total=None, start=False
        )

        self._eval_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[readout]}"),
            console=console,
            speed_estimate_period=10.0,
        )
        self._problem_task = self._eval_progress.add_task(
            "waiting to benchmark...", total=1, readout="0/0 problems", start=False
        )
        self._tokens_task = self._eval_progress.add_task(
            "tok/s", total=None, readout="-- tok/s", start=False, visible=False
        )

        # Log panel rendered inside Live so it doesn't corrupt the terminal.
        self._log_lines: list[str] = []
        self._log_panel = Progress(
            TextColumn("{task.fields[msg]}"),
            console=console,
        )
        self._log_task = self._log_panel.add_task("", msg=Text("waiting to start..."), start=False)

        self._live = Live(
            _Group(self._quants_progress, self._download_progress, self._eval_progress, self._log_panel),
            console=console,
            refresh_per_second=10,
        )

    def __enter__(self) -> _RichDisplay:
        self._live.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._live.stop()

    def log(self, message: str) -> None:
        self._log_lines.append(message)
        # Keep last 5 lines in the panel; show the most recent one.
        display = "\n".join(self._log_lines[-5:])
        self._log_panel.update(self._log_task, msg=Text(display, style="dim"), description="")

    def start_quant(self, name: str, index: int, total: int) -> None:
        self._quants_progress.update(
            self._quants_task, total=total, completed=index - 1, description=f"current: {name}"
        )

    def finish_quant(self) -> None:
        task = next(t for t in self._quants_progress.tasks if t.id == self._quants_task)
        self._quants_progress.update(self._quants_task, completed=(task.completed or 0) + 1)

    def start_download(self, name: str, total_bytes: int | None) -> None:
        self._download_progress.reset(
            self._download_task, total=total_bytes, completed=0, description=f"downloading {name}"
        )

    def download_tqdm_class(self) -> type[_tqdm_base]:
        return make_tqdm_shim(self._download_progress, self._download_task)

    def start_eval(self, name: str) -> None:
        self._eval_progress.reset(
            self._problem_task,
            total=None,
            completed=0,
            description=f"{name} problems",
            readout="0/? problems",
        )
        self._eval_progress.reset(
            self._tokens_task, total=None, completed=0, description=f"{name} tok/s", readout="-- tok/s", visible=True
        )

    def on_problem_done(self, index: int, total: int, completion_tokens: int, elapsed_s: float) -> None:
        self._eval_progress.update(
            self._problem_task, total=total, completed=index, readout=f"{index}/{total} problems"
        )
        self._eval_progress.update(self._tokens_task, advance=completion_tokens)
        speed = next(t for t in self._eval_progress.tasks if t.id == self._tokens_task).speed
        self._eval_progress.update(
            self._tokens_task, readout=f"{speed:.1f} tok/s" if speed else "-- tok/s"
        )


class _PlainDisplay:
    """Throttled print()-based fallback for non-TTY output (piped/CI)."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def __enter__(self) -> _PlainDisplay:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def start_quant(self, name: str, index: int, total: int) -> None:
        self._console.print(f"[{index}/{total}] {name}")

    def finish_quant(self) -> None:
        pass

    def start_download(self, name: str, total_bytes: int | None) -> None:
        size = f"{total_bytes / 1e9:.2f} GB" if total_bytes else "unknown size"
        self._console.print(f"  downloading {name} ({size})...")

    def download_tqdm_class(self) -> type[_tqdm_base]:
        console = self._console

        class _ThrottledTqdm(_tqdm_base):
            def __init__(self, *args, **kwargs):
                self._is_bytes = kwargs.get("unit") == "B"
                self._last_pct = -10
                kwargs["disable"] = True  # suppress tqdm's own stderr rendering
                super().__init__(*args, **kwargs)

            def update(self, n: int = 1) -> bool | None:
                result = super().update(n)
                if self._is_bytes and self.total:
                    pct = int(self.n / self.total * 100)
                    if pct >= self._last_pct + 10:
                        self._last_pct = pct
                        console.print(f"    {pct}% ({self.n / 1e9:.2f}/{self.total / 1e9:.2f} GB)")
                return result

        return _ThrottledTqdm

    def start_eval(self, name: str) -> None:
        self._console.print(f"  benchmarking {name}...")

    def on_problem_done(self, index: int, total: int, completion_tokens: int, elapsed_s: float) -> None:
        step = max(1, total // 10)
        if index == total or index % step == 0:
            tok_s = completion_tokens / elapsed_s if elapsed_s else 0.0
            self._console.print(f"  {index}/{total} problems ({tok_s:.1f} tok/s)")

    def log(self, message: str) -> None:
        self._console.print(message)


BenchDisplay = _RichDisplay | _PlainDisplay


def create_display(console: Console) -> BenchDisplay:
    return _RichDisplay(console) if console.is_terminal else _PlainDisplay(console)
