"""Pre-fetch / install the SphereSfM CUDA binary bundle into the pack's bin/.

You normally DON'T need to run this: the SplatKit_SphereSfMDataset node auto-downloads
the bundle from the GitHub Release into bin/ the first time it runs. Use this script only to:

  * fetch it ahead of time:        python tools/install_spheresfm.py
  * install from a local zip:       python tools/install_spheresfm.py --zip <bundle.zip>
                                    (for offline machines — copy the zip over first)

The bundle is a BSD-3-Clause CUDA build of github.com/json87/SphereSfM (sm_75..120 + PTX).
See docs/panosplat-workflow/SPHERESFM.md and bin/BUILD_INFO.txt for details.
"""
import argparse
import os
import sys
import zipfile

# import the node's resolver so we reuse one source of truth for paths + download logic.
# Run as a plain script, so the pack is not a package -- reach core/ by adding it to
# sys.path and importing the module by its (globally distinctive) bare name.
_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PACK)
sys.path.insert(0, os.path.join(_PACK, "core"))
import spheresfm_colmap as s  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Fetch/install the SphereSfM CUDA bundle into bin/")
    ap.add_argument("--zip", help="install from a local bundle zip instead of downloading")
    args = ap.parse_args()

    if args.zip:
        if not os.path.isfile(args.zip):
            sys.exit("[install_spheresfm] no such file: %s" % args.zip)
        os.makedirs(s._BIN_DIR, exist_ok=True)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(s._BIN_DIR)
        if os.path.isfile(s._PACK_BIN):
            print("[install_spheresfm] installed from %s -> %s" % (args.zip, s._PACK_BIN))
            return
        sys.exit("[install_spheresfm] FAILED: colmap_sphere.exe not found after extracting")

    if os.path.isfile(s._PACK_BIN):
        print("[install_spheresfm] already present -> %s" % s._PACK_BIN)
        return
    exe = s._download_colmap_sphere_bundle()
    if not exe:
        sys.exit("[install_spheresfm] download failed (see message above)")
    print("[install_spheresfm] OK -> %s" % exe)


if __name__ == "__main__":
    main()
