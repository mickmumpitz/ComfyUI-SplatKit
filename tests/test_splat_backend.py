"""Regression checks for the optional backend boundary; no models or CUDA required."""
import importlib
import importlib.util
import io
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# A package namespace lets relative imports work without starting the ComfyUI adapters.
package = types.ModuleType("splatkit_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
from splatkit_test.core.splatting import runtime
from splatkit_test.core.splatting import backend, sequence, runner

spec = importlib.util.spec_from_file_location("splatkit_test.nodes.splatting.sequence", ROOT / "nodes" / "splatting" / "sequence.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
spec = importlib.util.spec_from_file_location("install_splat_backend", ROOT / "tools" / "install_splat_backend.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

class SplatBackendTests(unittest.TestCase):
    def test_generation_progress_reaches_terminal_and_comfy_bar(self):
        script = """
import sys
sys.stdout.write('Loading model\\n')
sys.stdout.write('\\rGenerate 6 target views:  50%|#####     | 2/4 [00:02<00:02, 1.00s/it]')
sys.stdout.write('\\rGenerate 6 target views: 100%|##########| 4/4 [00:04<00:00, 1.00s/it]\\n')
sys.stdout.write('Saved views\\n')
"""
        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")
        mm.throw_exception_if_processing_interrupted = Mock()
        comfy.model_management = mm
        utils = types.ModuleType("comfy.utils")
        utils.ProgressBar = Mock()
        for tty in (True, False):
            with self.subTest(tty=tty):
                output = io.StringIO()
                output.isatty = lambda: tty
                utils.ProgressBar.reset_mock()
                bar = Mock(total=4)
                utils.ProgressBar.return_value = bar
                with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils}), \
                     redirect_stdout(output):
                    tail = runner.run([sys.executable, "-c", script], ROOT, progress=runner.tqdm_progress)
                text = output.getvalue()
                self.assertIn("2/4 [00:02<00:02, 1.00s/it]", text)
                self.assertIn("4/4 [00:04<00:00, 1.00s/it]", text)
                self.assertIn("Loading model", text)
                self.assertIn("\n" + runner.LOG + " Saved views\n", text)
                self.assertEqual("\r" in text, tty)
                self.assertTrue(any("4/4" in line for line in tail))
                utils.ProgressBar.assert_called_once_with(4)
                self.assertEqual(bar.update_absolute.call_count, 2)
                bar.update_absolute.assert_any_call(2, 4, None)
                bar.update_absolute.assert_any_call(4, 4, None)

    def test_huggingface_download_bars_survive_the_backend_pipe(self):
        script = """
from huggingface_hub.utils.tqdm import _get_progress_bar_context
with _get_progress_bar_context(desc='Download test.bin', log_level=30, total=1024) as bar:
    bar.update(1024)
"""
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        output = Terminal()
        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")
        mm.throw_exception_if_processing_interrupted = Mock()
        comfy.model_management = mm
        utils = types.ModuleType("comfy.utils")
        utils.ProgressBar = Mock()
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils}), \
             redirect_stdout(output):
            runner.run([sys.executable, "-c", script], ROOT, progress=runner.tqdm_progress)
        self.assertIn("Download test.bin", output.getvalue())
        self.assertIn("100%", output.getvalue())
        self.assertIn("B/s", output.getvalue())
        self.assertIn("\r", output.getvalue())

    def test_download_progress_is_enabled_in_backend_environment(self):
        with patch.dict(os.environ, {"HF_HUB_DISABLE_PROGRESS_BARS": "1"}):
            env = backend.environment()
        self.assertEqual(env["HF_HUB_DISABLE_PROGRESS_BARS"], "0")
        self.assertEqual(env["TQDM_POSITION"], "-1")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def frame(self, index):
        d = self.root / f"frame_{index:03d}"
        (d / "images").mkdir(parents=True)
        (d / "images" / "00.png").write_bytes(b"image")
        (d / "sparse_pcd.ply").write_bytes(b"pointcloud")
        (d / "transforms.json").write_text(json.dumps({"frames":[{"file_path":"images/00.png"}]}))
        return d

    def test_narrow_frame_selection_and_content_changes(self):
        self.frame(0)
        d = self.frame(1)
        first = sequence.read_frameset(self.root, [1])
        self.assertEqual(first["frame_ids"], [1])
        self.assertEqual(first["frames"], 1)
        image = d / "images" / "00.png"
        stat = image.stat()
        image.write_bytes(b"other")
        os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(first["fingerprint"], sequence.read_frameset(self.root, [1])["fingerprint"])

    def test_missing_selected_frame_fails(self):
        self.frame(0)
        with self.assertRaisesRegex(backend.BackendError, "Missing selected frame"):
            sequence.read_frameset(self.root, [1])

    def test_missing_image_is_not_complete(self):
        d = self.frame(0)
        (d / "images" / "00.png").unlink()
        with self.assertRaisesRegex(backend.BackendError, "Incomplete frame"):
            sequence.read_frameset(self.root)

    def test_dataset_paths_cannot_escape_root(self):
        d = self.frame(0)
        (d / "transforms.json").write_text(json.dumps({"frames":[{"file_path":"../../../outside.png"}]}))
        with self.assertRaises(backend.BackendError):
            sequence.read_frameset(self.root)

    def test_cache_requires_runtime_content_and_exact_frames(self):
        folder = self.root / "shot"
        (folder / "ply").mkdir(parents=True)
        (folder / "preview").mkdir()
        (folder / "ply" / "frame_00000.ply").write_bytes(b"ply")
        (folder / "preview" / "frame_00000.png").write_bytes(b"png")
        signature = {"runtime":"new", "content":"a", "frames":[0], "quality":"standard"}
        (folder / "meta.json").write_text(json.dumps({"frames":[0]}))
        with patch.object(train, "output_root", return_value=self.root):
            self.assertIsNone(train._finished_sequence("shot", signature))
            (folder / "splatkit_cache.json").write_text(json.dumps(signature))
            self.assertEqual(train._finished_sequence("shot", signature), folder)
            for key, value in [("runtime", "old"), ("trainer", "changed"), ("content", "b"), ("frames", [1]), ("quality", "best")]:
                self.assertIsNone(train._finished_sequence("shot", dict(signature, **{key:value})))
            self.assertIsNone(train._finished_sequence("sho", signature))

    def test_training_passes_only_selected_frames(self):
        self.frame(0)
        self.frame(1)
        fs = sequence.read_frameset(self.root, [1])
        output = self.root / "out"
        output.mkdir()
        config = {"python":sys.executable, "runtime_id":"test"}
        seq = {"dir":str(output), "preview":["preview.png"]}
        weights = self.root / "vgg.safetensors"
        weights.write_bytes(b"weights")
        mm = types.ModuleType("comfy.model_management")
        mm.unload_all_models = Mock()
        mm.soft_empty_cache = Mock()
        comfy = types.ModuleType("comfy")
        comfy.model_management = mm
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": mm}), \
             patch.object(train, "load_config", return_value=config), \
             patch.object(train, "check_path", side_effect=lambda p, _:Path(p)), \
             patch.object(train, "_finished_sequence", return_value=None), \
             patch.object(train, "sequence_dir", return_value=output), \
             patch.object(train, "run") as run, \
             patch.object(train, "read_sequence", return_value=seq), \
             patch.object(train, "load_images", return_value="images"):
            train.SplatKitTrain().train(fs, "draft", "shot", perceptual_model=str(weights))
        args = run.call_args.args[0]
        self.assertEqual(args[1:3], [str(runtime.WORKER), "train"])
        self.assertEqual(args[args.index("--frames")+1], "1")
        self.assertEqual(args[args.index("--perceptual-weights")+1], str(weights))
        cached = json.loads((output / "splatkit_cache.json").read_text())
        self.assertEqual(cached["trainer"], runtime.trainer_id())
        self.assertEqual(cached["pack"], runtime.pack_version())

    def make_playback(self, folder, index):
        head = bytearray(32)
        head[:8] = b"SPLATSH1"
        struct.pack_into("<I", head, 8, 1)
        (folder / f"frame_{index:05d}.splatsh").write_bytes(head + bytes(77))

    def test_partial_or_truncated_playback_is_repaired(self):
        folder = self.root / "splat"
        folder.mkdir()
        seq = {"dir":str(self.root), "ply":["frame_00000.ply", "frame_00001.ply"]}
        names = ["frame_00000.splatsh", "frame_00001.splatsh"]
        (folder / "index.json").write_text(json.dumps({"frames":names, "format":"splatsh"}))
        self.make_playback(folder, 0)
        self.assertFalse(sequence.playback_complete(seq))
        with patch.object(backend, "load_config", return_value={"python":"python", "backend_root":str(self.root)}), \
             patch.object(runner, "run", side_effect=lambda *a, **k:self.make_playback(folder, 1)) as run, \
             patch.object(sequence, "read_sequence", return_value=seq):
            sequence.ensure_player_files(seq)
            run.assert_called_once()
        self.assertTrue(sequence.playback_complete(seq))
        (folder / names[1]).write_bytes(b"SPLATSH1")
        self.assertFalse(sequence.playback_complete(seq))

    def test_loader_notices_modified_playback_without_frame_count_change(self):
        folder = self.root / "splat"
        folder.mkdir()
        frame = folder / "frame_00000.splatsh"
        frame.write_bytes(b"first")
        with patch.object(train, "check_path", return_value=self.root):
            before = train.SplatKitLoadSequence.IS_CHANGED(str(self.root))
            frame.write_bytes(b"truncated")
            after = train.SplatKitLoadSequence.IS_CHANGED(str(self.root))
        self.assertNotEqual(before, after)

    def test_missing_or_old_manifest_is_never_ready(self):
        with patch.object(backend, "manifest", return_value={}), \
             patch.object(runtime, "expected", return_value={"source_id":"new"}):
            self.assertFalse(backend.is_installed())
        with patch.object(backend, "manifest", return_value={"contract":{"source_id":"old"}, "cuda_smoke_test":True}), \
             patch.object(runtime, "expected", return_value={"source_id":"new"}):
            self.assertFalse(backend.is_installed())

    def test_environment_does_not_leak_host_python(self):
        with patch.dict(os.environ, {"PYTHONPATH":"host", "PYTHONHOME":"host"}):
            env = backend.environment()
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)

    def test_trainer_edit_changes_cache_without_changing_environment(self):
        source = self.root / "training"
        source.mkdir()
        module = source / "model.py"
        module.write_text("value = 1")
        with patch.object(runtime, "TRAINER_SOURCE", source):
            before, contract = runtime.trainer_id(), runtime.expected()
            module.write_text("value = 2")
            self.assertNotEqual(before, runtime.trainer_id())
            self.assertEqual(contract, runtime.expected())

    def test_dependency_and_wheel_hash_changes_require_setup_but_url_does_not(self):
        before = runtime.expected()
        with patch.object(runtime, "WHEEL_URL", "https://example.invalid/mirror.whl"):
            self.assertEqual(before, runtime.expected())
        with patch.object(runtime, "WHEEL_SHA256", "0" * 64):
            self.assertNotEqual(before, runtime.expected())
        req = self.root / "requirements.txt"
        req.write_text(runtime.REQUIREMENTS.read_text() + "\npillow==1.0\n")
        with patch.object(runtime, "REQUIREMENTS", req):
            self.assertNotEqual(before, runtime.expected())

    def test_generator_edit_requires_copy_refresh_only(self):
        source = self.root / "generator"
        source.mkdir()
        (source / "requirements.txt").write_text("torch==2.8.0")
        module = source / "inference.py"
        module.write_text("value = 1")
        contract = runtime.expected()
        with patch.object(runtime, "GENERATOR_SOURCE", source), \
             patch.object(runtime, "CHECKOUT", source), \
             patch.object(runtime, "expected", return_value=contract), \
             patch.object(backend, "venv_python", return_value=Path(sys.executable)), \
             patch.object(backend, "manifest", return_value={"contract": contract, "cuda_smoke_test": True,
                          "generator_source_id": runtime.source_id(source)}):
            backend.load_config()
            trainer_id = runtime.trainer_id()
            module.write_text("value = 2")
            self.assertEqual(trainer_id, runtime.trainer_id())
            backend.load_config(require_generator=False)
            with self.assertRaisesRegex(backend.BackendError, "source needs an update"):
                backend.load_config()

    def test_host_adapters_and_training_package_do_not_import_cuda(self):
        script = """
import importlib, importlib.util, sys, types
from pathlib import Path
root = Path(sys.argv[1])
server = types.ModuleType('server')
routes = types.SimpleNamespace(get=lambda path: lambda handler: handler)
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))
sys.modules['server'] = server
package = types.ModuleType('splatkit_isolation')
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
for suffix in ('nodes', 'nodes.splatting', 'nodes.four_d_anyone'):
    namespace = types.ModuleType(package.__name__ + '.' + suffix)
    namespace.__path__ = [str(root.joinpath(*suffix.split('.')))]
    sys.modules[namespace.__name__] = namespace
for group in ('splatting', 'four_d_anyone'):
    for path in (root / 'nodes' / group).glob('*.py'):
        if path.name == '__init__.py':
            continue
        importlib.import_module(f'splatkit_isolation.nodes.{group}.{path.stem}')
importlib.import_module('splatkit_isolation.core.splatting.training')
assert not any(n == 'gsplat' or n.startswith('gsplat.') or n.endswith('.training.model') for n in sys.modules)
"""
        subprocess.run([sys.executable, "-c", script, str(ROOT)], cwd=self.root, check=True,
                       capture_output=True, text=True)

    def test_worker_entry_point_from_another_directory(self):
        # Substitute just the CLI to verify bootstrap and exit propagation without GPU dependencies.
        script = """
import runpy, sys, types
cli = types.ModuleType('core.splatting.training.cli')
def main():
    from core.splatting import runtime
    assert runtime.PACK_ROOT.is_dir()
    return 7
cli.main = main
sys.modules[cli.__name__] = cli
runpy.run_path(sys.argv[1], run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", script, str(runtime.WORKER)], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_training_backend_does_not_require_generator_checkout(self):
        contract = {"source_id": "test"}
        with patch.object(backend, "manifest", return_value={"contract": contract, "cuda_smoke_test": True}), \
             patch.object(runtime, "expected", return_value=contract), \
             patch.object(runtime, "CHECKOUT", self.root / "missing-generator"), \
             patch.object(backend, "venv_python", return_value=Path(sys.executable)):
            self.assertEqual(backend.load_config(require_generator=False)["python"], sys.executable)
            with self.assertRaisesRegex(backend.BackendError, "checkout is missing"):
                backend.load_config(require_generator=True)

    def test_setup_rejects_unverified_download(self):
        with self.assertRaises(RuntimeError):
            setup.download("https://example.invalid/file", self.root / "file", "")

    def test_generator_refresh_rolls_back_without_reinstalling_dependencies(self):
        target = self.root / "splat_backend"
        checkout = target / "4DAnyone"
        checkout.mkdir(parents=True)
        (checkout / "inference.py").write_text("old generator")
        source = self.root / "source"
        source.mkdir()
        (source / "inference.py").write_text("new generator")
        manifest = target / "manifest.json"
        manifest.write_text('{"old": true}')
        with patch.object(setup.runtime, "BACKEND", target), \
             patch.object(setup.runtime, "CHECKOUT", checkout), \
             patch.object(setup.runtime, "GENERATOR_SOURCE", source), \
             patch.object(setup.runtime, "MANIFEST", manifest), \
             patch.object(setup, "run") as run, \
             patch.object(setup, "verify", side_effect=RuntimeError("smoke failed")):
            with self.assertRaisesRegex(RuntimeError, "smoke failed"):
                setup.refresh_sources({}, "new")
            self.assertEqual((checkout / "inference.py").read_text(), "old generator")
            self.assertEqual(json.loads(manifest.read_text()), {"old": True})
            run.assert_not_called()
        with patch.object(setup.runtime, "BACKEND", target), \
             patch.object(setup.runtime, "CHECKOUT", checkout), \
             patch.object(setup.runtime, "GENERATOR_SOURCE", source), \
             patch.object(setup.runtime, "MANIFEST", manifest), \
             patch.object(setup, "run") as run, patch.object(setup, "verify"):
            setup.refresh_sources({}, "new")
            self.assertEqual((checkout / "inference.py").read_text(), "new generator")
            self.assertEqual(json.loads(manifest.read_text())["generator_source_id"], "new")
            run.assert_not_called()
        self.assertFalse(checkout.with_name("4DAnyone.previous").exists())

    def test_current_environment_refreshes_generator_without_downloads(self):
        target = self.root / "splat_backend"
        target.mkdir()
        manifest = target / "manifest.json"
        contract = runtime.expected()
        manifest.write_text(json.dumps({"contract": contract, "cuda_smoke_test": True,
                                       "generator_source_id": "old"}))
        with patch.object(setup.runtime, "ROOT", self.root), \
             patch.object(setup.runtime, "BACKEND", target), \
             patch.object(setup.runtime, "MANIFEST", manifest), \
             patch.object(setup, "python", return_value=Path(sys.executable)), \
             patch.object(setup, "refresh_sources") as refresh, patch.object(setup, "download") as download, \
             patch.object(setup, "clean_cache") as clean:
            setup.install()
            refresh.assert_called_once_with(contract, runtime.source_id(runtime.GENERATOR_SOURCE), retire_legacy=False)
            download.assert_not_called()
            clean.assert_called_once_with()

    def test_clean_cache_is_best_effort_and_targets_the_uv_cache(self):
        tools = self.root / "cache"
        tools.mkdir()
        # No uv.exe yet: nothing to run, and no crash.
        with patch.object(setup.runtime, "TOOLS", tools), patch.object(setup, "run") as run:
            setup.clean_cache()
            run.assert_not_called()
        # With uv present, it delegates to `uv cache clean` and swallows failures.
        (tools / "uv.exe").write_text("stub")
        with patch.object(setup.runtime, "TOOLS", tools), \
             patch.object(setup, "run", side_effect=subprocess.CalledProcessError(1, "uv")) as run:
            setup.clean_cache()  # must not raise
            self.assertEqual(list(run.call_args.args[0]), [tools / "uv.exe", "cache", "clean"])

    def test_keep_cache_skips_cleanup_on_refresh(self):
        target = self.root / "splat_backend"
        target.mkdir()
        manifest = target / "manifest.json"
        contract = runtime.expected()
        manifest.write_text(json.dumps({"contract": contract, "cuda_smoke_test": True,
                                       "generator_source_id": "old"}))
        with patch.object(setup.runtime, "ROOT", self.root), \
             patch.object(setup.runtime, "BACKEND", target), \
             patch.object(setup.runtime, "MANIFEST", manifest), \
             patch.object(setup, "python", return_value=Path(sys.executable)), \
             patch.object(setup, "refresh_sources"), patch.object(setup, "download"), \
             patch.object(setup, "clean_cache") as clean:
            setup.install(keep_cache=True)
            clean.assert_not_called()

    def test_setup_rejects_archive_traversal(self):
        import zipfile
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../escaped.txt", "no")
        with self.assertRaises(RuntimeError):
            setup.extract(archive, self.root / "extract")
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_setup_deletion_is_contained(self):
        with self.assertRaises(RuntimeError):
            setup.remove_backend(self.root)
        self.assertTrue(self.root.is_dir())

    def test_failed_upgrade_restores_previous_install(self):
        target = self.root / "splat_backend"
        previous = self.root / "splat_backend.previous"
        target.mkdir()
        manifest = target / "manifest.json"
        manifest.write_text('{"old":true}')
        (target / "keep").write_text("previous backend")
        class Space:
            free = 30 * 10**9
        with patch.object(setup.runtime, "ROOT", self.root), \
             patch.object(setup.runtime, "BACKEND", target), \
             patch.object(setup.runtime, "MANIFEST", manifest), \
             patch.object(setup.runtime, "VENV", target / "venv"), \
             patch.object(setup.runtime, "expected", return_value={}), \
             patch.object(setup, "download", return_value=self.root / "archive"), \
             patch.object(setup, "extract"), patch.object(setup.shutil, "disk_usage", return_value=Space()), \
             patch.object(setup, "run", side_effect=RuntimeError("install failed")):
            with self.assertRaisesRegex(RuntimeError, "install failed"):
                setup.install(rebuild=True)
        self.assertEqual((target / "keep").read_text(), "previous backend")
        self.assertFalse(previous.exists())

    def test_cancellation_kills_subprocess(self):
        started = time.monotonic()
        mm = types.ModuleType("comfy.model_management")
        mm.throw_exception_if_processing_interrupted = Mock(side_effect=[None, RuntimeError("cancelled")])
        comfy = types.ModuleType("comfy")
        comfy.model_management = mm
        utils = types.ModuleType("comfy.utils")
        utils.ProgressBar = Mock()
        with patch.dict(sys.modules, {"comfy":comfy, "comfy.model_management":mm, "comfy.utils":utils}), \
             patch.object(runner, "kill_tree", wraps=runner.kill_tree) as kill:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                runner.run([sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(30)"], self.root)
            kill.assert_called_once()
            self.assertIsNotNone(kill.call_args.args[0].poll())
        self.assertLess(time.monotonic() - started, 5)

if __name__ == "__main__":
    unittest.main()
