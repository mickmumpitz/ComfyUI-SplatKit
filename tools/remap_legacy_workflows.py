"""Remap workflows saved with the old Pano2Splat-Matrix pack onto SplatKit node ids.

This pack was previously called ComfyUI-Pano2Splat-Matrix and its node class ids were
prefixed ``P2SMatrix_``. They are now ``SplatKit_``. A workflow .json saved before the
rename therefore loads with "missing node" errors. This script rewrites those ids in
place.

It touches every place ComfyUI stores a class id:

  * graph format   -- nodes[].type, nodes[].properties["Node name for S&R"]
  * API format     -- <node_id>.class_type
  * provenance     -- nodes[].properties.cnr_id / .aux_id (so ComfyUI Manager doesn't
                      offer to reinstall the old pack to satisfy "missing" nodes)

Nodes that did NOT make it into the release (the experimental Gaussian Splash, image-to-pano,
detail-regen and GenRecon nodes) have no SplatKit equivalent. They are left untouched and
reported: a workflow that uses them still needs the old pack, which is kept on disk as
custom_nodes/ComfyUI-Pano2Splat-Matrix.disabled (drop the .disabled suffix to re-enable it).

Usage (dry run -- prints what WOULD change, writes nothing):

    python_embeded\\python.exe custom_nodes\\ComfyUI-SplatKit\\tools\\remap_legacy_workflows.py

Apply it, backing every rewritten file up to <name>.json.p2sbak first:

    python_embeded\\python.exe ...\\remap_legacy_workflows.py --apply

Options:
    <dir_or_file> ...   what to scan (default: ComfyUI/user/default/workflows)
    --apply             actually write (default is a dry run)
    --no-backup         skip the .p2sbak copies
"""

import argparse
import json
import os
import shutil
import sys

OLD_PREFIX = "P2SMatrix_"
NEW_PREFIX = "SplatKit_"

# The 20 nodes that shipped in SplatKit. A P2SMatrix_<suffix> id is remappable iff
# <suffix> appears here -- everything else was left behind in the old pack.
SHIPPED = {
    "WanI2VMaskedConditioning",
    "DatasetProject",
    "RenderControlInProcess",
    "CameraPlotRenderControl",
    "CameraPlotRenderControlGeo",
    "CameraPlotSceneReference",
    "BuildEquirectDataset",
    "BuildEquirectDatasetFused",
    "PanoToPerspectiveViews",
    "EquirectCameraView",
    "SphereSfMDataset",
    "SphereSfMAddToDataset",
    "SaveWanPanoFrames",
    "LoadWanPanoFrames",
    "MoGeModelLoader",
    "ResolveDatasetImages",
    "LoadDatasetImagesOrdered",
    "SaveUpscaledDataset",
    "HiResPanoFlythrough",
    "AddHiResViewsToDataset",
}

OLD_REPO = "mickmumpitz/ComfyUI-Pano2Splat-Matrix"
NEW_REPO = "mickmumpitz/ComfyUI-SplatKit"


def _default_workflow_dir():
    # this file -> tools/ -> pack root -> custom_nodes/ -> ComfyUI/
    here = os.path.abspath(__file__)
    comfy = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    return os.path.join(comfy, "user", "default", "workflows")


def _remap_id(value):
    """Return (new_id, status) where status is 'mapped', 'dropped' or 'skip'."""
    if not isinstance(value, str) or not value.startswith(OLD_PREFIX):
        return value, "skip"
    suffix = value[len(OLD_PREFIX):]
    if suffix in SHIPPED:
        return NEW_PREFIX + suffix, "mapped"
    return value, "dropped"


def remap(obj, stats):
    """Walk the loaded JSON and rewrite every class id in place.

    The pack-provenance keys (cnr_id / aux_id, which tell ComfyUI Manager where a node
    comes from) are only rewritten on nodes whose class id actually MOVED to SplatKit.
    A node that stayed behind must keep pointing at the old pack, or Manager will hunt
    for it in SplatKit, where it does not exist.
    """
    if isinstance(obj, dict):
        # Is this dict a node? (graph format: "type"; API format: "class_type")
        id_key = next((k for k in ("type", "class_type")
                       if isinstance(obj.get(k), str)
                       and obj[k].startswith(OLD_PREFIX)), None)
        if id_key:
            old_id = obj[id_key]
            new_id, status = _remap_id(old_id)
            if status == "dropped":
                stats["dropped"][old_id] = stats["dropped"].get(old_id, 0) + 1
                return  # leave this node, and its provenance, entirely alone
            obj[id_key] = new_id
            stats["mapped"][old_id] = stats["mapped"].get(old_id, 0) + 1
            props = obj.get("properties")
            if isinstance(props, dict):
                if props.get("Node name for S&R") == old_id:
                    props["Node name for S&R"] = new_id
                for key in ("cnr_id", "aux_id"):
                    if props.get(key) == OLD_REPO:
                        props[key] = NEW_REPO
                        stats["repo"] += 1
            return

        for val in obj.values():
            remap(val, stats)
    elif isinstance(obj, list):
        for item in obj:
            remap(item, stats)


def process(path, apply_changes, backup):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print("  !! %s -- not valid JSON, skipped (%s)" % (os.path.basename(path), e))
        return False

    stats = {"mapped": {}, "dropped": {}, "repo": 0}
    remap(data, stats)
    n_mapped = sum(stats["mapped"].values())
    if not n_mapped and not stats["dropped"] and not stats["repo"]:
        return False

    print("\n%s" % os.path.relpath(path))
    for node_id, count in sorted(stats["mapped"].items()):
        print("   %-40s -> %-40s x%d"
              % (node_id, NEW_PREFIX + node_id[len(OLD_PREFIX):], count))
    if stats["repo"]:
        print("   %-40s -> %-40s x%d" % (OLD_REPO, NEW_REPO, stats["repo"]))
    for node_id, count in sorted(stats["dropped"].items()):
        print("   !! %-37s NOT in SplatKit -- left as-is (x%d)" % (node_id, count))

    if apply_changes and (n_mapped or stats["repo"]):
        if backup:
            shutil.copy2(path, path + ".p2sbak")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("   written%s" % (" (backup: %s.p2sbak)" % os.path.basename(path) if backup else ""))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="*", help="workflow .json files and/or directories")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default: dry run, nothing is written)")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not leave a <name>.json.p2sbak copy of each rewritten file")
    args = ap.parse_args()

    targets = args.targets or [_default_workflow_dir()]
    files = []
    for t in targets:
        if os.path.isdir(t):
            for root, _dirs, names in os.walk(t):
                files += [os.path.join(root, n) for n in sorted(names) if n.endswith(".json")]
        elif os.path.isfile(t):
            files.append(t)
        else:
            print("not found: %s" % t)

    if not files:
        print("no .json files found in: %s" % ", ".join(targets))
        return 1

    print("%s %d file(s)%s"
          % ("Rewriting" if args.apply else "DRY RUN over", len(files),
             "" if args.apply else " -- nothing will be written; re-run with --apply"))

    touched = sum(1 for f in files if process(f, args.apply, not args.no_backup))
    print("\n%d of %d file(s) reference the old pack." % (touched, len(files)))
    if touched and not args.apply:
        print("Re-run with --apply to rewrite them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
