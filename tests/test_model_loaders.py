"""Local model selection and backend handoff without downloads or CUDA."""
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
for suffix in ("", ".nodes", ".nodes.four_d_anyone", ".nodes.splatting"):
    package = types.ModuleType("splatkit_loaders" + suffix)
    package.__path__ = [str(ROOT.joinpath(*suffix.strip(".").split("."))) if suffix else str(ROOT)]
    sys.modules[package.__name__] = package
generate = importlib.import_module("splatkit_loaders.nodes.four_d_anyone.generate")
train = importlib.import_module("splatkit_loaders.nodes.splatting.sequence")
paths = importlib.import_module("splatkit_loaders.core.four_d_anyone.paths")
weights = importlib.import_module("splatkit_loaders.core.splatting.training.weights")
sys.path.insert(0, str(ROOT / "vendored" / "4danyone"))
from fdanyone import assets


class ModelLoaderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.names = {"checkpoint": "model.safetensors", "vae": "Wan2.2_VAE.pth",
                      "prompt_context": "prompt_context.safetensors", "birefnet": "model.safetensors",
                      "sam3d_body": "sam_3d_body_dinov3_bf16.safetensors", "turbo_lora": "none"}
        self.generator = self.root / "splatkit" / "4danyone"
        self.birefnet = self.root / "splatkit" / "birefnet"
        self.detection = self.root / "detection"
        self.dirs = {"splatkit_4danyone": ([str(self.generator)], set()),
                     "splatkit_birefnet": ([str(self.birefnet)], set()),
                     "detection": ([str(self.detection)], {".safetensors"})}
        bases = {"sam3d_body": self.detection, "birefnet": self.birefnet}
        for kind, name in self.names.items():
            if name == "none":
                continue
            file = bases.get(kind, self.generator) / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(kind.encode())
        for name in paths.BIREFNET_FILES:
            self.birefnet.mkdir(parents=True, exist_ok=True)
            (self.birefnet / name).write_bytes(name.encode())
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = str(self.root)
        folder_paths.folder_names_and_paths = self.dirs
        folder_paths.add_model_folder_path = lambda key, path: self.dirs.setdefault(key, ([path], set()))
        folder_paths.get_filename_list = lambda key: sorted(
            p.relative_to(base).as_posix() for base in map(Path, self.dirs[key][0])
            for p in base.rglob("*") if p.is_file() and p.suffix in self.dirs[key][1])
        folder_paths.get_full_path_or_raise = lambda key, name: str(Path(self.dirs[key][0][0]) / name)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"folder_paths": folder_paths}).start()

    def test_local_loader_resolves_every_selected_file(self):
        bundle, = generate.FourDAnyoneModelLoader().load(**self.names)
        self.assertEqual(bundle["vae"], str(self.generator / self.names["vae"]))
        self.assertEqual(bundle["sam3d_body"], str(self.detection / self.names["sam3d_body"]))
        self.assertEqual(bundle["turbo_lora"], "")
        self.assertNotIn("auto (download if missing)", paths.model_options("checkpoint"))

    def test_unknown_or_traversal_selection_is_rejected(self):
        for name in ("../model.safetensors", "auto (download if missing)", "missing.safetensors"):
            with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                paths.resolve_model("checkpoint", name)

    def test_birefnet_requires_config_and_code_files(self):
        (self.birefnet / "config.json").unlink()
        with self.assertRaisesRegex(generate.BackendError, "config.json"):
            generate.FourDAnyoneModelLoader().load(**self.names)

    def test_partial_downloads_do_not_appear_in_choices(self):
        partial = self.birefnet / ".download/model.safetensors"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"partial")
        self.assertEqual(paths.model_options("birefnet"), ["model.safetensors"])

    def test_turbo_requires_selected_lora_before_backend_setup(self):
        bundle, = generate.FourDAnyoneModelLoader().load(**self.names)
        with patch.dict(sys.modules, {"comfy": types.ModuleType("comfy"),
                                      "comfy.model_management": Mock()}), \
             patch.object(generate, "load_config") as config:
            with self.assertRaisesRegex(generate.BackendError, "turbo_lora"):
                generate.FourDAnyoneGenerateViews().generate("full", 42, True, models=bundle)
        config.assert_not_called()

    def test_base_generation_passes_local_paths_and_pose_selection(self):
        bundle, = generate.FourDAnyoneModelLoader().load(**self.names)
        source = self.root / "clip.mp4"
        source.write_bytes(b"video")
        config = {"backend_root": str(self.root), "runtime_id": "runtime", "generator_source_id": "source", "python": "python"}
        with patch.dict(sys.modules, {"comfy": types.ModuleType("comfy"), "comfy.model_management": Mock()}), \
             patch.object(generate, "load_config", return_value=config), \
             patch.object(generate, "generated_root", return_value=self.root / "outputs"), \
             patch.object(generate, "materialize_video", return_value=source), \
             patch.object(generate, "probe_video", return_value={}), \
             patch.object(generate, "contract_warnings", return_value=[]), \
             patch.object(generate, "resolve_preset", return_value=(4, [15])), \
             patch.object(generate, "load_bundle", return_value={}), \
             patch.object(generate, "run") as run, \
             patch.object(generate.FourDAnyoneGenerateViews, "_pose", return_value=self.root / "pose.npz") as pose, \
             patch.object(generate.FourDAnyoneGenerateViews, "_write_run_info"):
            result, _ = generate.FourDAnyoneGenerateViews().generate("full", 42, False, models=bundle)
        args = run.call_args.args[0]
        for flag, value in (("--vae_path", bundle["vae"]), ("--prompt_context_path", bundle["prompt_context"]),
                            ("--checkpoint_path", bundle["checkpoint"]),
                            ("--foreground_model_dir", str(self.birefnet))):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn("--turbo_lora_path", args)
        self.assertEqual(pose.call_args.args[-1], Path(bundle["sam3d_body"]))
        self.assertEqual(result["foreground_model_dir"], str(self.birefnet))

    def test_backend_resolves_selected_vae_and_context(self):
        base = assets.resolve_base_assets(self.root, vae_path=self.generator / self.names["vae"],
                                         prompt_context_path=self.generator / self.names["prompt_context"])
        self.assertEqual(base.vae, self.generator / self.names["vae"])
        self.assertEqual(assets.resolve_foreground_model(path=self.birefnet), self.birefnet)

    def test_perceptual_weights_require_explicit_local_file(self):
        with self.assertRaises(FileNotFoundError):
            weights.perceptual_weights()
        with self.assertRaises(FileNotFoundError):
            weights.perceptual_weights(self.root / "missing")
        local = self.generator / "imagenet-vgg-verydeep-19-conv.safetensors"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"vgg")
        self.assertEqual(train.SplatKitPerceptualModelLoader().load(local.name), (str(local),))
        self.assertEqual(weights.perceptual_weights(local), local)

    def test_train_requires_loader_before_backend_setup(self):
        with patch.object(train, "load_config") as config:
            with self.assertRaisesRegex(train.BackendError, "Perceptual Model Loader"):
                train.SplatKitTrain().train({}, "draft", "test")
        config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
