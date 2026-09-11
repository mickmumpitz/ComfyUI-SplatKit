"""Build ComfyUI workflow JSON from a running server's object_info.

Hand-written workflow JSON drifts: widget order is not INPUT_TYPES order (ComfyUI lists
link inputs first, then every widget as a connectable input), seed widgets grow a hidden
"control after generate" value, and a saved graph whose widgets do not line up loads as
UNKNOWN. So the graph is generated from what the server actually reports, and validated
against it. The same description also yields the API-format prompt used to execute the
graph headlessly.

    from graph import Graph, fetch_object_info
    info = fetch_object_info("http://127.0.0.1:8188")
    g = Graph(info)
    a = g.node("LoadVideo", (0, 0), values={"file": "clip.mp4"})
    ...
    g.dump(path); prompt = g.prompt()
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "LOAD_3D", "COLOR",
                "COMFY_DYNAMICCOMBO_V3"}
CONTROL_AFTER = ("seed", "noise_seed")


def fetch_object_info(server: str) -> dict:
    with urllib.request.urlopen(server.rstrip("/") + "/object_info", timeout=30) as r:
        return json.loads(r.read().decode())


def _is_widget(type_) -> bool:
    # V1 nodes report a combo as a list of options; V3 nodes as the string "COMBO" with the
    # options in the option dict.
    return isinstance(type_, list) or type_ == "COMBO" or type_ in WIDGET_TYPES


def _type_name(type_) -> str:
    return "COMBO" if isinstance(type_, list) else str(type_)


def _default(type_, opts: dict):
    if isinstance(type_, list):
        if "default" in opts:
            return opts["default"]
        return type_[0] if type_ else ""
    if "default" in opts:
        return opts["default"]
    if type_ == "COMBO":
        options = opts.get("options") or []
        return options[0] if options else ""
    if type_ == "COMFY_DYNAMICCOMBO_V3":
        options = opts.get("options") or []
        first = options[0] if options else ""
        return first.get("key", first.get("value", "")) if isinstance(first, dict) else first
    return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False, "LOAD_3D": "",
            "COLOR": "#000000"}.get(type_, "")


class Graph:
    def __init__(self, info: dict):
        self.info = info
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._node_id = 0
        self._link_id = 0
        self._widget_names: dict[int, list[str]] = {}

    def _inputs_in_order(self, type_: str):
        spec = self.info[type_]
        inputs = spec.get("input", {})
        order = spec.get("input_order", {})
        out = []
        for section in ("required", "optional"):
            names = order.get(section) or list(inputs.get(section, {}).keys())
            for name in names:
                entry = inputs[section][name]
                t = entry[0]
                opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
                out.append((name, t, opts, section == "optional"))
        return out

    def node(self, type_: str, pos, size=None, values: dict | None = None,
             title: str | None = None, mode: int = 0, color: str | None = None) -> int:
        if type_ not in self.info:
            raise KeyError(f"{type_} is not a node type on this server")
        values = dict(values or {})
        spec = self.info[type_]
        self._node_id += 1
        inputs, widgets, names = [], [], []
        for name, t, opts, optional in self._inputs_in_order(type_):
            if _is_widget(t):
                value = values.pop(name, _default(t, opts))
                widgets.append(value)
                names.append(name)
                if name in CONTROL_AFTER or opts.get("control_after_generate"):
                    widgets.append("fixed")
                inputs.append({"name": name, "type": _type_name(t), "link": None,
                               "widget": {"name": name}})
            else:
                entry = {"name": name, "type": _type_name(t), "link": None}
                if optional:
                    entry["shape"] = 7
                inputs.append(entry)
        if values:
            raise KeyError(f"{type_} has no widgets named {sorted(values)}")
        outputs = []
        for i, (t, n) in enumerate(zip(spec.get("output", []), spec.get("output_name", []))):
            outputs.append({"name": n, "type": _type_name(t), "links": [], "slot_index": i})
        n = {
            "id": self._node_id, "type": type_, "pos": list(pos),
            "size": list(size or self._guess_size(len(inputs), len(widgets))),
            "flags": {}, "order": len(self.nodes), "mode": mode,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": type_},
            "widgets_values": widgets,
        }
        if title:
            n["title"] = title
        if color:
            n["color"] = color
            n["bgcolor"] = color
        self.nodes.append(n)
        self._widget_names[n["id"]] = names
        return n["id"]

    @staticmethod
    def _guess_size(n_inputs: int, n_widgets: int):
        return (360, 60 + 26 * (n_inputs + 1))

    def raw_node(self, type_: str, pos, size, properties: dict | None = None,
                 widgets: list | None = None, title: str | None = None,
                 color: str | None = None, bgcolor: str | None = None) -> int:
        """A node the server does not describe: frontend-only (MarkdownNote) or from a pack
        that is not installed on the build server (MickmumpitzLabel). No inputs or outputs."""
        self._node_id += 1
        n = {
            "id": self._node_id, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": 0, "inputs": [], "outputs": [],
            "properties": {"Node name for S&R": type_, **(properties or {})},
            "widgets_values": list(widgets or []),
        }
        if title:
            n["title"] = title
        if color:
            n["color"] = color
        if bgcolor:
            n["bgcolor"] = bgcolor
        self.nodes.append(n)
        self._widget_names[n["id"]] = []
        return n["id"]

    def md_note(self, text: str, pos, size, title: str, color: str | None = None,
                bgcolor: str | None = None) -> int:
        return self.raw_node("MarkdownNote", pos, size, widgets=[text], title=title,
                             color=color, bgcolor=bgcolor)

    def note(self, text: str, pos, size, title: str) -> int:
        self._node_id += 1
        self.nodes.append({
            "id": self._node_id, "type": "Note", "pos": list(pos), "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": 0, "title": title,
            "inputs": [], "outputs": [], "properties": {"text": ""},
            "widgets_values": [text], "color": "#432", "bgcolor": "#653",
        })
        return self._node_id

    def _node(self, node_id: int) -> dict:
        return next(n for n in self.nodes if n["id"] == node_id)

    def _slot(self, node: dict, key: str, name_or_index) -> int:
        if isinstance(name_or_index, int):
            return name_or_index
        for i, entry in enumerate(node[key]):
            if entry["name"] == name_or_index:
                return i
        raise KeyError(f"{node['type']} has no {key[:-1]} named {name_or_index!r}")

    def link(self, src_id: int, src_out, dst_id: int, dst_in) -> None:
        src, dst = self._node(src_id), self._node(dst_id)
        s, d = self._slot(src, "outputs", src_out), self._slot(dst, "inputs", dst_in)
        stype, dtype = src["outputs"][s]["type"], dst["inputs"][d]["type"]
        if stype != dtype and stype not in dtype.split(",") and dtype != "*":
            raise TypeError(f"{src['type']}.{src['outputs'][s]['name']} ({stype}) -> "
                            f"{dst['type']}.{dst['inputs'][d]['name']} ({dtype})")
        self._link_id += 1
        self.links.append([self._link_id, src_id, s, dst_id, d, stype])
        src["outputs"][s]["links"].append(self._link_id)
        dst["inputs"][d]["link"] = self._link_id

    def group(self, title: str, bounding, color: str = "#3f789e") -> dict:
        return {"id": len(self.nodes) + 1000, "title": title, "bounding": list(bounding),
                "color": color, "font_size": 24, "flags": {}}

    def dump(self, path: Path, groups: list[dict] | None = None,
             free: tuple[int, ...] = ()) -> None:
        if groups:
            self.check_layout(groups, free=free)
        else:
            self.check_overlaps()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "last_node_id": self._node_id, "last_link_id": self._link_id,
            "nodes": self.nodes, "links": self.links, "groups": groups or [],
            "config": {}, "extra": {}, "version": 0.4,
        }, indent=2), encoding="utf-8")

    TITLE_H = 30            # LiteGraph draws the title bar above pos.y
    GROUP_TITLE_H = 50      # and a group's title band below its bounding top

    def _box(self, n: dict):
        return (n["pos"][0], n["pos"][1] - self.TITLE_H,
                n["pos"][0] + n["size"][0], n["pos"][1] + n["size"][1])

    def check_overlaps(self) -> None:
        boxes = [(n["id"], *self._box(n)) for n in self.nodes]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                    raise ValueError(f"nodes {a[0]} and {b[0]} overlap")

    def check_layout(self, groups: list[dict], margin: int = 10,
                     free: tuple[int, ...] = ()) -> None:
        """Every node sits fully inside one group (below its title band), no two groups
        touch. `free` lists node ids allowed outside any group (the title labels)."""
        self.check_overlaps()
        gb = [(g["title"], g["bounding"][0], g["bounding"][1], g["bounding"][0] + g["bounding"][2],
               g["bounding"][1] + g["bounding"][3]) for g in groups]
        for i, a in enumerate(gb):
            for b in gb[i + 1:]:
                if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                    raise ValueError(f"groups {a[0]!r} and {b[0]!r} overlap")
        for n in self.nodes:
            if n["id"] in free:
                continue
            x0, y0, x1, y1 = self._box(n)
            inside = [t for t, gx0, gy0, gx1, gy1 in gb
                      if x0 >= gx0 + margin and y0 >= gy0 + self.GROUP_TITLE_H
                      and x1 <= gx1 - margin and y1 <= gy1 - margin]
            if not inside:
                raise ValueError(f"node {n['id']} {n['type']} at {n['pos']} is not inside a group")

    def prompt(self, overrides: dict | None = None) -> dict:
        """API-format prompt for /prompt. Muted (mode 2) and Note nodes are left out."""
        overrides = overrides or {}
        out = {}
        for n in self.nodes:
            if n["type"] in ("Note", "MarkdownNote", "MickmumpitzLabel") or n.get("mode", 0) != 0:
                continue
            inputs = {}
            names = self._widget_names[n["id"]]
            # widgets_values carries a "fixed" after every seed-type widget; skip those
            vals = []
            wi = 0
            for nm in names:
                vals.append(n["widgets_values"][wi])
                wi += 1
                if nm in CONTROL_AFTER:
                    wi += 1
            for nm, v in zip(names, vals):
                inputs[nm] = v
            for entry in n["inputs"]:
                if entry.get("link") is not None:
                    link = next(lk for lk in self.links if lk[0] == entry["link"])
                    inputs[entry["name"]] = [str(link[1]), link[2]]
            inputs.update(overrides.get(n["id"], {}))
            out[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
        # drop links to muted nodes
        for node in out.values():
            for k, v in list(node["inputs"].items()):
                if isinstance(v, list) and len(v) == 2 and v[0] not in out:
                    del node["inputs"][k]
        return out


def validate(path: Path, info: dict) -> list[str]:
    """Problems a saved workflow would have on this server. Empty means clean."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    g = Graph(info)
    for n in data["nodes"]:
        if n["type"] in ("Note", "MarkdownNote", "MickmumpitzLabel"):
            continue
        if n["type"] not in info:
            problems.append(f"node {n['id']}: unknown type {n['type']}")
            continue
        expected = 0
        for name, t, opts, _ in g._inputs_in_order(n["type"]):
            if _is_widget(t):
                expected += 1
                if name in CONTROL_AFTER or opts.get("control_after_generate"):
                    expected += 1
        if len(n.get("widgets_values", [])) != expected:
            problems.append(f"node {n['id']} {n['type']}: {len(n.get('widgets_values', []))} "
                            f"widget values, server expects {expected}")
    ids = {n["id"] for n in data["nodes"]}
    for lk in data["links"]:
        if lk[1] not in ids or lk[3] not in ids:
            problems.append(f"link {lk[0]} dangles")
    return problems
