"""Convert Matrix-3D's pano LoRA (DiffSynth/PEFT format) to ComfyUI Wan format.

DiffSynth/PEFT keys look like:
    blocks.{N}.self_attn.q.lora_A.default.weight   [rank, in]
    blocks.{N}.self_attn.q.lora_B.default.weight   [out, rank]

ComfyUI/Wan (same convention lightx2v uses, which loads natively) wants:
    diffusion_model.blocks.{N}.self_attn.q.lora_down.weight   [rank, in]
    diffusion_model.blocks.{N}.self_attn.q.lora_up.weight     [out, rank]

The internal module paths (self_attn/cross_attn.{q,k,v,o}, ffn.{0,2}) are identical
in both, so this is a pure key rename. lora_A == lora_down, lora_B == lora_up.

Usage:
    python convert_pano_lora.py <in.bin> <out.safetensors> [--ref lightx2v.safetensors] [--dtype bf16|fp16|fp32]
"""
import argparse, re, sys
import torch
from safetensors.torch import save_file, load_file

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def convert(sd):
    out = {}
    unmatched = []
    for k, v in sd.items():
        nk = k
        if nk.endswith(".lora_A.default.weight"):
            nk = nk[: -len(".lora_A.default.weight")] + ".lora_down.weight"
        elif nk.endswith(".lora_B.default.weight"):
            nk = nk[: -len(".lora_B.default.weight")] + ".lora_up.weight"
        else:
            unmatched.append(k)
            continue
        out["diffusion_model." + nk] = v
    return out, unmatched


def module_paths(keys):
    """Strip lora suffixes + diffusion_model prefix -> bare module paths for comparison."""
    paths = set()
    for k in keys:
        p = k
        p = re.sub(r"^diffusion_model\.", "", p)
        p = re.sub(r"\.(lora_down|lora_up|lora_A|lora_B)(\.default)?\.weight$", "", p)
        p = re.sub(r"\.(diff_b|diff)$", "", p)
        paths.add(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--ref", default=None, help="known-good ComfyUI LoRA for structural check")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    args = ap.parse_args()

    obj = torch.load(args.inp, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and not hasattr(next(iter(obj.values())), "shape"):
        obj = obj["state_dict"]

    out, unmatched = convert(obj)
    print(f"input keys:     {len(obj)}")
    print(f"converted keys: {len(out)}")
    print(f"unmatched keys: {len(unmatched)}")
    if unmatched:
        print("  WARNING sample unmatched:", unmatched[:5])

    dt = DTYPES[args.dtype]
    out = {k: v.to(dt).contiguous() for k, v in out.items()}

    if args.ref:
        ref = load_file(args.ref)
        pano_mods = module_paths(out.keys())
        ref_mods = module_paths(ref.keys())
        missing = sorted(pano_mods - ref_mods)
        print(f"\nstructural check vs ref ({args.ref.split('/')[-1]}):")
        print(f"  pano module paths:        {len(pano_mods)}")
        print(f"  also present in ref:      {len(pano_mods & ref_mods)}")
        print(f"  NOT in ref (would fail):  {len(missing)}")
        if missing:
            print("  sample missing:", missing[:8])
        else:
            print("  -> every pano-targeted module exists in the known-good ComfyUI LoRA. OK.")

    save_file(out, args.out, metadata={"format": "pt", "source": "matrix3d_pano_video_gen_720p"})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
