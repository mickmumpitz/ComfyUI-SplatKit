"""The sequence playing inside the node, with the camera under the user's control.

Preview Splat (core) shows one file; Preview Sequence renders from a camera fixed in
advance. This is the third answer: a real splat renderer embedded in the node, so the clip
plays while you orbit it. The renderer is antimatter15/splat (MIT), vendored under web/,
pointed at the trained folder through a small route. Nothing renders on the server.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...core.splatting.constants import CATEGORY, NODE_PREFIX, TYPE_SEQUENCE
from ...core.splatting.backend import BackendError


_REGISTERED: dict[str, Path] = {}


def register(folder: str | Path) -> str:
    path = Path(folder).resolve()
    token = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    _REGISTERED[token] = path
    return token


def _setup() -> bool:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return False

    routes = PromptServer.instance.routes

    @routes.get("/splatkit/seq/{token}/{name}")
    async def splatkit_file(request):
        folder = _REGISTERED.get(request.match_info["token"])
        if folder is None:
            return web.Response(status=404, text="unknown sequence")
        name = request.match_info["name"]
        # Only a plain file name inside the registered folder. On Windows a name like
        # "C:foo" is drive-relative and Path joining would replace the folder with it, so
        # the check is on the parsed path, not on the characters.
        parsed = Path(name)
        if parsed.name != name or parsed.drive or parsed.anchor or name in (".", ".."):
            return web.Response(status=400, text="bad name")
        target = (folder / name).resolve()
        if target.parent != folder.resolve() or not target.is_file():
            return web.Response(status=404, text="no such frame")
        content_type = "application/json" if name.endswith(".json") else "application/octet-stream"
        return web.FileResponse(target, headers={"Content-Type": content_type,
                                                 "Cache-Control": "no-store"})

    return True


ROUTE_READY = _setup()


class SplatKitPlayer:
    CATEGORY = CATEGORY
    FUNCTION = "play"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = ("Plays the sequence in the node. Drag to orbit, right-drag to pan, wheel to "
                   "zoom, space to play, arrows to step, F to reset the view.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sequence": (TYPE_SEQUENCE,)}}

    def play(self, sequence):
        folder = Path(sequence["dir"]) / "splat"
        if not (folder / "index.json").is_file():
            raise BackendError(f"No playable sequence in {folder}. Train writes one there.")
        if not ROUTE_READY:
            raise BackendError("The SplatKit player route did not register. Restart ComfyUI.")
        token = register(folder)
        count = len(json.loads((folder / "index.json").read_text(encoding="utf-8")).get("frames", []))
        return {"ui": {"splatkit": [{"token": token, "frames": count}]}}


NODE_CLASS_MAPPINGS = {NODE_PREFIX + "SequencePlayer": SplatKitPlayer}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_PREFIX + "SequencePlayer": "Sequence Player"}
