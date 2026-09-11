"""Private subprocess entry point for licensed body-model conditioning."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fdanyone.device import select_cuda_device
from fdanyone.skeleton.pipeline import build_skeleton_conditioning
from fdanyone.video import load_canonical_working_clip
from fdanyone.views import ViewPlan


def main(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text())
    device, _ = select_cuda_device(request["device"])
    clip = load_canonical_working_clip(request["working_video"], request["clip_metadata"])

    # Body pose comes from the npz written by tools/sam3d_mhr70.py.
    import numpy as np

    from fdanyone.skeleton.sam3d import load as load_sam
    from fdanyone.skeleton.sam3d import motion_from_sam, to_body_geometry

    data = np.load(request["sam_keypoints"])
    geometry = to_body_geometry(load_sam(request["sam_keypoints"],
                                         smooth_window=request.get("sam_smooth", 9)))
    motion = motion_from_sam(data, len(clip.frames))

    build_skeleton_conditioning(
        motion=motion,
        clip=clip,
        foreground_model_path=request["foreground_model_path"],
        output_dir=request["output_dir"],
        device=device,
        view_plan=ViewPlan.from_dict(request["view_plan"]),
        geometry=geometry,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m fdanyone.skeleton.worker REQUEST.json")
    main(sys.argv[1])
