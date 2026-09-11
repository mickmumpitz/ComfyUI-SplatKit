"""One subprocess runner for every backend call.

What it guarantees, because each point once cost a run:

* **Progress reaches the ComfyUI bar.** The child's merged output is split on both `\\r` and
  `\\n`, so tqdm's in-place updates arrive line by line; a parser turns them into
  (done, total). Generation and download bars remain visible in the terminal.
* **Cancel actually stops the work.** The loop checks ComfyUI's interrupt flag twice a
  second and on any exception kills the whole process tree: 4DAnyone spawns worker
  processes and a plain `proc.kill()` leaves them holding the GPU.
* **Failures carry evidence.** The last forty lines travel with the exception, so the error
  shown in ComfyUI says what the backend said.
"""

from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from .constants import LOG
from .backend import BackendError, environment

Progress = Callable[[str], "tuple[int, int] | None"]
# A preview callback returns a PIL image for a log line that announces a new artefact
# (a trained frame, a matted camera, a published view); the image is sent to the node
# through the same channel KSampler uses for its sampling preview.
Preview = Callable[[str], "object | None"]
PREVIEW_MAX = 512

_TQDM = re.compile(r"(\d+)/(\d+)\s*\[")
_DOWNLOAD_BAR = re.compile(r"\d+%\|.*(?:B/s|s/B)\]")
_FRAME = re.compile(r"^\s*frame\s+\d+\s+(cold|warm|player)\s+\d+\s+it\b", re.IGNORECASE)


def tqdm_progress(line: str):
    """4DAnyone prints tqdm bars like 'Generate 6 target views:  75%|###| 18/24 [37:03<...'."""
    m = _TQDM.search(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


class FrameProgress:
    """The trainer prints one line per finished frame: '  frame   30  cold 30000 it  2.61 min ...'."""

    def __init__(self, total: int):
        self.total = max(1, int(total))
        self.done = 0

    def __call__(self, line: str):
        if _FRAME.match(line):
            self.done += 1
            return (self.done, self.total)
        return None


class PhaseProgress:
    """Two counted phases in one bar, for the frameset export (cameras, then frames)."""

    def __init__(self, phases: list[tuple[re.Pattern, int]]):
        self.phases = [(p, max(1, n)) for p, n in phases]
        self.done = [0] * len(self.phases)
        self.total = sum(n for _, n in self.phases)

    def __call__(self, line: str):
        for i, (pattern, _) in enumerate(self.phases):
            if pattern.search(line):
                self.done[i] += 1
                return (sum(self.done), self.total)
        return None


def _drain(proc: subprocess.Popen, lines: queue.Queue[str | None]) -> None:
    try:
        buffer = b""
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            buffer += chunk
            while True:
                cuts = [i for i in (buffer.find(b"\r"), buffer.find(b"\n")) if i != -1]
                if not cuts:
                    break
                cut = min(cuts)
                line = buffer[:cut].decode("utf-8", errors="replace").strip()
                buffer = buffer[cut + 1:]
                if line:
                    lines.put(line)
        rest = buffer.decode("utf-8", errors="replace").strip()
        if rest:
            lines.put(rest)
    finally:
        lines.put(None)


def kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        result = subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                capture_output=True, timeout=10)
        if result.returncode and proc.poll() is None:
            proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=10)


def run(command: list[str], cwd: str | Path, extra_env: dict | None = None,
        progress: Progress | None = None, preview: Preview | None = None,
        label: str = LOG, tail_lines: int = 40) -> list[str]:
    """Run a backend command to completion. Returns the last lines of its output.

    `progress(line)` turns log lines into (done, total) for ComfyUI's bar; `preview(line)`
    may return a PIL image for a line, which is shown on the node like a sampling preview.
    """
    try:
        import comfy.model_management as mm
        from comfy.utils import ProgressBar
    except Exception:                      # outside ComfyUI (tests): no bar, no interrupt
        mm = None
        ProgressBar = None

    popen_kwargs = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True     # so killpg reaches the workers
    proc = subprocess.Popen([str(c) for c in command], cwd=str(cwd),
                            env=environment(extra_env), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, **popen_kwargs)
    lines: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=_drain, args=(proc, lines), daemon=True).start()

    tail: list[str] = []
    bar = None
    closed = False
    terminal_bar_width = 0
    try:
        while not closed:
            if mm is not None:
                mm.throw_exception_if_processing_interrupted()
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if mm is not None:
                    mm.throw_exception_if_processing_interrupted()
                continue
            if line is None:
                closed = True
                continue
            step = progress(line) if progress is not None else None
            image = None
            if preview is not None:
                try:
                    image = preview(line)
                except Exception as exc:    # a preview must never fail the run
                    print(f"{label} preview skipped: {exc}", flush=True)
            if step is not None:
                done, total = step
                if ProgressBar is not None:
                    if bar is None or bar.total != total:
                        bar = ProgressBar(total)
                    bar.update_absolute(min(done, total), total,
                                        ("JPEG", image, PREVIEW_MAX) if image is not None else None)
                    image = None
            if image is not None and ProgressBar is not None:
                if bar is None:
                    bar = ProgressBar(1)
                bar.update_absolute(bar.current, bar.total, ("JPEG", image, PREVIEW_MAX))
            tail.append(line)
            del tail[:-tail_lines]
            if (_TQDM.search(line) or _DOWNLOAD_BAR.search(line)) and sys.stdout.isatty():
                text = f"{label} {line}"
                print("\r" + text.ljust(terminal_bar_width), end="", flush=True)
                terminal_bar_width = len(text)
            else:
                if terminal_bar_width:
                    print(flush=True)
                    terminal_bar_width = 0
                print(f"{label} {line}", flush=True)
            if mm is not None:
                mm.throw_exception_if_processing_interrupted()
        while proc.poll() is None:
            if mm is not None:
                mm.throw_exception_if_processing_interrupted()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        code = proc.returncode
    except BaseException:
        kill_tree(proc)
        raise
    finally:
        proc.stdout.close()
        if terminal_bar_width:
            print(flush=True)
    if code != 0:
        raise BackendError(f"backend exited with code {code}:\n" + "\n".join(tail[-15:]))
    return tail
