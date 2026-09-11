"""Disk cache identities include the selected files and effective runtime."""
import hashlib
import json
from pathlib import Path

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def fingerprint(files, root):
    h = hashlib.sha256()
    for file in sorted(set(map(Path, files))):
        h.update(file.relative_to(root).as_posix().encode())
        h.update(file_hash(file).encode())
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    stage = path.with_suffix(path.suffix + ".partial")
    stage.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    stage.replace(path)
