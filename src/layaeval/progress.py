"""Progress reporting and real-time process visibility for LayaEval."""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
from typing import IO, Any, Callable, Iterable, Iterator, TypeVar

from .hardware import HardwareInfo, detect_hardware

T = TypeVar("T")

_STARTUP_SHOWN = False
_STARTUP_LOCK = threading.Lock()


def is_progress_enabled(override: bool | None = None) -> bool:
    """Determine whether progress display is enabled.

    Args:
        override: Explicit boolean override passed by user, or None to auto-detect.

    Returns:
        True if progress should be displayed.
    """
    if override is not None:
        return bool(override)

    env_val = os.environ.get("LAYAEVAL_PROGRESS") or os.environ.get("LAYAEVAL_VERBOSE")
    if env_val is not None:
        return env_val.lower() not in ("0", "false", "no", "off")

    # Enabled by default in interactive shells or Jupyter/Colab notebooks
    if _in_notebook():
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _in_notebook() -> bool:
    """Check if code is running in a Jupyter or Google Colab notebook."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell", "Shell") or "google.colab" in sys.modules
    except NameError:
        return "google.colab" in sys.modules
    except Exception:
        return False


def show_hardware_banner(
    *,
    hardware: HardwareInfo | None = None,
    model: str = "auto",
    force: bool = False,
) -> None:
    """Display detected hardware acceleration and server initialization banner.

    Guaranteed to run once per process unless force=True.
    """
    global _STARTUP_SHOWN
    with _STARTUP_LOCK:
        if _STARTUP_SHOWN and not force:
            return
        _STARTUP_SHOWN = True

    hw = hardware or detect_hardware()
    print(f"[LayaEval] Hardware: {hw.description}")
    print(f"[LayaEval] Initializing Laya MCP Server (model: {model})...")
    sys.stdout.flush()


class StderrWatcher:
    """Background reader that buffers and streams subprocess stderr in real time.

    Captures HuggingFace model download progress, weight loading logs, and
    diagnostics, forwarding them to the user while keeping a circular buffer
    for error reporting.
    """

    def __init__(
        self,
        stream: IO[str],
        on_line: Callable[[str], None] | None = None,
        max_buffer_lines: int = 60,
        show_progress: bool = True,
    ):
        self.stream = stream
        self.on_line = on_line
        self.show_progress = show_progress
        self.buffer: collections.deque[str] = collections.deque(maxlen=max_buffer_lines)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for raw_line in iter(self.stream.readline, ""):
                if not raw_line:
                    break
                line = raw_line.rstrip()
                self.buffer.append(line)

                # Check if this line looks like a download or model loading progress
                if self.show_progress and self._is_progress_line(line):
                    print(f"[LayaEval] {line}", flush=True)

                if self.on_line:
                    try:
                        self.on_line(line)
                    except Exception:
                        pass
        except Exception:
            pass

    @staticmethod
    def _is_progress_line(line: str) -> bool:
        """Detect lines from huggingface_hub, torch, or laya indicating progress."""
        lower = line.lower()
        progress_keywords = (
            "downloading",
            "downloaded",
            "fetching",
            "loading checkpoint",
            "loading weights",
            "extracting",
            "resolving weights",
            "100%",
            "%|",
        )
        return any(kw in lower for kw in progress_keywords)

    def get_buffered_stderr(self) -> str:
        """Return all buffered stderr lines joined by newline."""
        return "\n".join(self.buffer)


class ProgressBar:
    """Universal progress bar that automatically leverages tqdm when available,

    with an elegant fallback for environments without tqdm.
    """

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        total: int | None = None,
        desc: str = "Evaluating",
        unit: str = "it",
        enabled: bool = True,
    ):
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)  # type: ignore
        self.desc = desc
        self.unit = unit
        self.enabled = enabled
        self.current = 0
        self._start_time = time.time()
        self._tqdm_instance: Any = None

        if self.enabled:
            try:
                from tqdm.auto import tqdm

                self._tqdm_instance = tqdm(
                    iterable=iterable,
                    total=self.total,
                    desc=self.desc,
                    unit=self.unit,
                    leave=True,
                )
            except ImportError:
                self._tqdm_instance = None

    def __iter__(self) -> Iterator[T]:
        if not self.enabled or self.iterable is None:
            if self.iterable is not None:
                yield from self.iterable
            return

        if self._tqdm_instance is not None:
            yield from self._tqdm_instance
            return

        # Fallback text progress bar
        self._start_time = time.time()
        self._render(0)
        for item in self.iterable:
            yield item
            self.current += 1
            self._render(self.current)
        self._finish()

    def update(self, n: int = 1, desc: str | None = None) -> None:
        """Manually advance progress by n items."""
        if not self.enabled:
            return

        if desc:
            self.desc = desc

        if self._tqdm_instance is not None:
            if desc:
                self._tqdm_instance.set_description(desc)
            self._tqdm_instance.update(n)
            return

        self.current += n
        self._render(self.current)
        if self.total and self.current >= self.total:
            self._finish()

    def set_postfix_str(self, text: str) -> None:
        """Set dynamic postfix text for the progress bar."""
        if not self.enabled:
            return
        if self._tqdm_instance is not None and hasattr(self._tqdm_instance, "set_postfix_str"):
            self._tqdm_instance.set_postfix_str(text)

    def close(self) -> None:
        """Close the progress bar."""
        if self._tqdm_instance is not None:
            self._tqdm_instance.close()

    def _render(self, current: int) -> None:
        if self.total and self.total > 0:
            fraction = min(1.0, current / self.total)
            pct = int(fraction * 100)
            bar_len = 24
            filled = int(fraction * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            status = f"\r[LayaEval] {self.desc}: [{bar}] {pct}% ({current}/{self.total} {self.unit})"
        else:
            status = f"\r[LayaEval] {self.desc}: {current} {self.unit}"

        sys.stdout.write(status)
        sys.stdout.flush()

    def _finish(self) -> None:
        elapsed = time.time() - self._start_time
        rate = (self.current / elapsed) if elapsed > 0 else 0
        if self.total:
            sys.stdout.write(f"\r[LayaEval] {self.desc}: Done in {elapsed:.2f}s ({rate:.1f} {self.unit}/s)\n")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()
