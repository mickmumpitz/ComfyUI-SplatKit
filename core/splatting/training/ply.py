"""Minimal binary PLY reader and writer.

Deliberately dependency-free. nerfstudio reads the seed point cloud with open3d and
several splat tools use plyfile; open3d is a 400 MB dependency for one file read and
plyfile is GPLv3, which would be incompatible with this project's Apache-2.0 licence.
Both are replaced by the two functions below.
"""

from __future__ import annotations

import numpy as np

_NP_FROM_PLY = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


def read_ply_table(path) -> np.ndarray:
    """The vertex element of a binary_little_endian PLY as a structured array."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt, count, fields = None, 0, []
        in_vertex = False
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == b"format":
                fmt = parts[1].decode()
            elif key == b"element":
                in_vertex = parts[1] == b"vertex"
                if in_vertex:
                    count = int(parts[2])
            elif key == b"property" and in_vertex:
                if parts[1] == b"list":
                    raise ValueError("list properties are not supported on the vertex element")
                fields.append((parts[2].decode(), _NP_FROM_PLY[parts[1].decode()]))
            elif key == b"end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"{path} is '{fmt}'; only binary_little_endian is supported")
        data = np.frombuffer(fh.read(count * np.dtype(fields).itemsize), dtype=np.dtype(fields), count=count)
    return data


def read_ply(path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz float32 [N,3], rgb uint8 [N,3]) from a binary_little_endian PLY.

    Colours default to mid grey when the file carries none.
    """
    data = read_ply_table(path)
    names = data.dtype.names
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    return xyz, rgb


def read_ply_comments(path) -> list[str]:
    """The `comment` lines of a PLY header, in order."""
    out = []
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        while True:
            line = fh.readline()
            if not line or line.strip() == b"end_header":
                break
            if line.startswith(b"comment "):
                out.append(line[8:].strip().decode(errors="replace"))
    return out


def write_ply(path, arrays: dict[str, np.ndarray], comments: tuple[str, ...] = ()) -> None:
    """Write one vertex element. `arrays` maps property name to a [N] or [N,k] column."""
    cols, names = [], []
    n = None
    for name, value in arrays.items():
        value = np.asarray(value)
        value = value[:, None] if value.ndim == 1 else value
        n = value.shape[0] if n is None else n
        if value.shape[0] != n:
            raise ValueError(f"property {name} has {value.shape[0]} rows, expected {n}")
        for k in range(value.shape[1]):
            cols.append(value[:, k])
            names.append(name if value.shape[1] == 1 else f"{name}_{k}")
    dtype = np.dtype([(nm, c.dtype.newbyteorder("<").str) for nm, c in zip(names, cols)])
    out = np.empty(n, dtype=dtype)
    for nm, c in zip(names, cols):
        out[nm] = c
    ply_of = {("f", 4): "float", ("f", 8): "double", ("u", 1): "uchar",
              ("i", 4): "int", ("u", 4): "uint", ("i", 2): "short", ("u", 2): "ushort"}
    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment {c}" for c in comments]
    header.append(f"element vertex {n}")
    header += [f"property {ply_of[(dtype[nm].kind, dtype[nm].itemsize)]} {nm}" for nm in names]
    header.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        fh.write(out.tobytes())
