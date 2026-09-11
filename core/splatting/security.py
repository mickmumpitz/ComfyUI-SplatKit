"""What a UI user may ask this pack to do, depending on who can reach the server.

ComfyUI itself draws this line: when the server listens on a non-loopback address, ComfyUI
Manager refuses install actions at its default security level. This pack follows suit.
Free-form path widgets (a clip anywhere on disk, a frameset folder, a model tree) are the
normal way to work on one's own machine, so on a loopback server they stay open; on a
server that listens on the network they are restricted to ComfyUI's own input, output,
temp and models folders, and the Setup node refuses to download and install anything.

`SPLATKIT_ALLOW_REMOTE=1` in the environment lifts both restrictions for people who run
ComfyUI behind their own authentication.
"""

from __future__ import annotations

import os
from pathlib import Path

from .backend import BackendError

LOOPBACK = {"127.0.0.1", "localhost", "::1", ""}


def server_is_public() -> bool:
    if os.environ.get("SPLATKIT_ALLOW_REMOTE") == "1":
        return False
    try:
        from server import PromptServer
        address = str(getattr(PromptServer.instance, "address", "") or "")
    except Exception:
        return False
    return address not in LOOPBACK


def allowed_roots() -> list[Path]:
    import folder_paths
    roots = [folder_paths.get_input_directory(), folder_paths.get_output_directory(),
             folder_paths.get_temp_directory(), folder_paths.models_dir]
    return [Path(r).resolve() for r in roots if r]


def check_path(path: str | Path, what: str = "path") -> Path:
    """Resolve a user-supplied path; on a public server it must sit inside ComfyUI."""
    p = Path(str(path).strip().strip('"')).expanduser()
    try:
        resolved = p.resolve()
    except OSError as exc:
        raise BackendError(f"{what} cannot be resolved: {p} ({exc})") from exc
    if server_is_public():
        for root in allowed_roots():
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise BackendError(
            f"{what} is outside ComfyUI's input, output, temp and models folders: {resolved}. "
            "This server listens on the network, so only those folders are allowed. Set "
            "SPLATKIT_ALLOW_REMOTE=1 to lift the restriction on a server you have secured.")
    return resolved


def check_setup_allowed() -> None:
    if server_is_public():
        raise BackendError(
            "The backend install is refused while ComfyUI listens on a network address: it "
            "downloads and runs software. Run it from a ComfyUI bound to localhost, or set "
            "SPLATKIT_ALLOW_REMOTE=1 on a server you have secured.")
