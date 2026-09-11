"""Downloading large model files without a helper package.

Hugging Face answers range requests, and on a throttled connection eight parallel ranges
turned 0.3 MB/s into 33 MB/s (2.83 GB in 1.2 minutes instead of two hours, measured on the
SAM 3D Body weights). So this fetches in parallel when the server allows it and falls back
to a plain stream when it does not. The file is written beside its destination and moved
into place only when complete and the right size, so an interrupted download can never be
mistaken for a finished model.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ..splatting.constants import PACK_NAME, PACK_VERSION

USER_AGENT = f"{PACK_NAME}/{PACK_VERSION}"
CHUNK = 32 << 20            # 32 MB per range request
RETRIES = 4


def _request(url: str, method: str = "GET", headers: dict | None = None):
    if not url.lower().startswith("https://"):
        raise ValueError(f"refusing a non-HTTPS download: {url}")
    # the scheme is checked above, so only https reaches urllib
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    req = urllib.request.Request(url, method=method, headers=hdrs)  # noqa: S310
    return urllib.request.urlopen(req, timeout=60)  # noqa: S310


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def probe(url: str) -> tuple[int, bool]:
    """(content length, supports ranges) for a URL, following redirects."""
    with _request(url, "HEAD") as r:
        size = int(r.headers.get("Content-Length") or 0)
        ranges = "bytes" in (r.headers.get("Accept-Ranges") or "").lower()
    return size, ranges


def download(url: str, dest: str | Path, threads: int = 8,
             progress: Callable[[int, int], None] | None = None,
             sha256: str | None = None) -> Path:
    """Fetch `url` to `dest`; when `sha256` is given the file must match it or it is removed."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        # A cached file is only trusted if it still matches its pinned hash. Without a
        # pin we cannot verify it, so we keep the previous behaviour and reuse it.
        if not sha256 or _sha256_file(dest) == sha256.lower():
            return dest
        dest.unlink()
    tmp = dest.with_name(dest.name + ".partial")
    size, ranges = probe(url)
    done = 0
    lock = threading.Lock()

    def report(n: int) -> None:
        nonlocal done
        with lock:
            done += n
            if progress is not None:
                progress(done, size)

    if size and ranges and threads > 1:
        with open(tmp, "wb") as f:
            f.truncate(size)
        spans = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]

        def fetch(span):
            start, end = span
            for attempt in range(RETRIES):
                try:
                    with _request(url, headers={"Range": f"bytes={start}-{end}"}) as r, \
                            open(tmp, "r+b") as f:
                        f.seek(start)
                        pos = start
                        while True:
                            block = r.read(1 << 20)
                            if not block:
                                break
                            f.write(block)
                            pos += len(block)
                    if pos != end + 1:
                        raise OSError(f"short range {start}-{end}: got {pos - start} bytes")
                    report(pos - start)
                    return
                except Exception:
                    if attempt == RETRIES - 1:
                        raise
            return

        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(fetch, spans))
    else:
        with _request(url) as r, open(tmp, "wb") as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                report(len(block))

    got = tmp.stat().st_size
    if size and got != size:
        tmp.unlink(missing_ok=True)
        raise OSError(f"download of {url} is incomplete: {got} of {size} bytes")
    if sha256:
        if _sha256_file(tmp) != sha256.lower():
            tmp.unlink(missing_ok=True)
            raise OSError(f"download of {url} does not match its published SHA-256; "
                          "the file was removed. Try again, or check the network path.")
    os.replace(tmp, dest)
    return dest


def free_space_gb(path: str | Path) -> float:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9
