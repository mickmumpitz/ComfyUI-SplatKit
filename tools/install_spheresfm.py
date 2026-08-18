"""Pre-fetch / install the SphereSfM binary bundle into the pack's bin/.

You normally DON'T need to run this: the SplatKit_SphereSfMDataset node auto-downloads
the right bundle for your platform (Windows / Linux / macOS) from the GitHub Release
into bin/ the first time it runs. Use this script only to:

  * fetch it ahead of time:        python tools/install_spheresfm.py
  * install from a local archive:  python tools/install_spheresfm.py --archive <file>
                                   (zip or tar.gz -- for offline machines, copy it over
                                   first; --zip is the old name and still works)

Every bundle is a BSD-3-Clause build of github.com/json87/SphereSfM (COLMAP 3.8 +
sphere patches). See docs/SPHERESFM.md and bin/BUILD_INFO.txt for details.
"""
import argparse
import os
import sys

# import the node's resolver so we reuse one source of truth for paths + download logic.
# Run as a plain script, so the pack is not a package -- reach core/ by adding it to
# sys.path and importing the module by its (globally distinctive) bare name.
_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PACK)
sys.path.insert(0, os.path.join(_PACK, "core"))
import spheresfm_colmap as s  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Fetch/install the SphereSfM bundle into bin/")
    ap.add_argument("--archive", "--zip", dest="archive",
                    help="install from a local bundle archive (zip or tar.gz) "
                         "instead of downloading")
    args = ap.parse_args()

    if args.archive:
        if not os.path.isfile(args.archive):
            sys.exit("[install_spheresfm] no such file: %s" % args.archive)
        exe = s.extract_bundle_archive(args.archive)
        if exe:
            print("[install_spheresfm] installed from %s -> %s" % (args.archive, exe))
            return
        sys.exit("[install_spheresfm] FAILED: %s not found after extracting"
                 % s._EXE_NAME)

    if os.path.isfile(s._PACK_BIN):
        print("[install_spheresfm] already present -> %s" % s._PACK_BIN)
        return
    exe = s._download_colmap_sphere_bundle()
    if not exe:
        sys.exit("[install_spheresfm] download failed (see message above)")
    print("[install_spheresfm] OK -> %s" % exe)


if __name__ == "__main__":
    main()
