"""Download recovery checks without network access or model dependencies."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from requests.exceptions import ChunkedEncodingError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendored" / "4danyone"))
from fdanyone import download
from fdanyone.errors import AssetError
from fdanyone import assets

spec = importlib.util.spec_from_file_location("training_weights", Path(__file__).resolve().parents[1] / "core/splatting/training/weights.py")
weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)


class ModelDownloadTests(unittest.TestCase):
    def test_interrupted_model_keeps_partial_file_and_publishes_only_on_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            partial = root / ".cache" / "huggingface" / "download" / "4danyone" / "checkpoint.incomplete"
            destination = root / "4danyone" / "checkpoint.pt"
            attempts = []

            def fetch(**kwargs):
                attempts.append(kwargs)
                self.assertFalse(destination.exists())
                if len(attempts) == 1:
                    partial.parent.mkdir(parents=True)
                    partial.write_bytes(b"first")
                    raise ChunkedEncodingError("truncated response")
                self.assertEqual(partial.read_bytes(), b"first")
                partial.write_bytes(b"first second")
                destination.parent.mkdir(parents=True)
                partial.replace(destination)
                return str(destination)

            with patch.object(download, "MODEL_FILES", ["checkpoint.pt"]), \
                 patch.object(download, "ensure_foreground_model"), \
                 patch("huggingface_hub.hf_hub_download", side_effect=fetch), \
                 patch.object(download.time, "sleep"):
                self.assertEqual(download.ensure_models(root), root)
            self.assertEqual(attempts[0], attempts[1])
            self.assertEqual(attempts[1]["local_dir"], str(root))
            self.assertEqual(destination.read_bytes(), b"first second")

    def test_repeated_interruptions_stop_with_recovery_instructions(self):
        fetch = Mock(side_effect=ChunkedEncodingError("truncated response"))
        with patch.object(download.time, "sleep") as sleep:
            with self.assertRaisesRegex(AssetError, "Run the workflow again") as error:
                download._download_with_retry(fetch)
        self.assertIsInstance(error.exception.__cause__, ChunkedEncodingError)
        self.assertEqual(fetch.call_count, 5)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4, 8])

    def test_local_errors_are_not_retried(self):
        fetch = Mock(side_effect=PermissionError("read-only directory"))
        with patch.object(download.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                download._download_with_retry(fetch)
        fetch.assert_called_once()
        sleep.assert_not_called()

    def test_completed_models_skip_all_network_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in download.MODEL_FILES:
                path = root / "4danyone" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"completed")
            for name in download.BIREFNET_FILES:
                path = root / "birefnet" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"completed")
            with patch("huggingface_hub.hf_hub_download") as fetch, \
                 patch("huggingface_hub.snapshot_download") as snapshot:
                download.ensure_models(root)
            fetch.assert_not_called()
            snapshot.assert_not_called()

    def test_base_with_custom_checkpoint_skips_default_checkpoint_and_lora(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            custom = root / "custom.safetensors"
            custom.write_bytes(b"custom")
            with patch("huggingface_hub.hf_hub_download") as fetch, \
                 patch.object(download, "ensure_foreground_model"):
                download.ensure_models(root, enable_turbo=False, checkpoint_path=custom)
            self.assertEqual([call.kwargs["filename"] for call in fetch.call_args_list],
                             ["4danyone/Wan2.2_VAE.pth", "4danyone/prompt_context.safetensors"])

    def test_birefnet_downloads_only_missing_files_into_its_own_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "birefnet").mkdir()
            (root / "birefnet/model.safetensors").write_bytes(b"completed")
            with patch("huggingface_hub.snapshot_download") as fetch:
                download.ensure_foreground_model(root)
            self.assertEqual(fetch.call_args.kwargs["local_dir"], root / "birefnet")
            self.assertNotIn("model.safetensors", fetch.call_args.kwargs["allow_patterns"])
            self.assertEqual(fetch.call_args.kwargs["max_workers"], 1)

    def test_asset_resolvers_follow_the_organized_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "4danyone").mkdir()
            for name in (assets.CHECKPOINT, assets.WAN_VAE, assets.PROMPT_CONTEXT):
                (root / "4danyone" / name).write_bytes(b"model")
            self.assertEqual(assets.resolve_checkpoint(None, root), root / "4danyone" / assets.CHECKPOINT)
            base = assets.resolve_base_assets(root)
            self.assertEqual(base.vae, root / "4danyone" / assets.WAN_VAE)
            self.assertEqual(base.prompt_context, root / "4danyone" / assets.PROMPT_CONTEXT)

    def test_vgg_download_retries_in_the_same_cache_and_reuses_completed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / weights.FILENAME
            with patch.object(weights, "cache_dir", return_value=root), \
                 patch.object(weights.time, "sleep"), \
                 patch("huggingface_hub.hf_hub_download", side_effect=[ChunkedEncodingError("broken"), str(destination)]) as fetch:
                self.assertEqual(weights.perceptual_weights(), destination)
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(fetch.call_args_list[0], fetch.call_args_list[1])
                destination.write_bytes(b"complete")
                self.assertEqual(weights.perceptual_weights(), destination)
                self.assertEqual(fetch.call_count, 2)


if __name__ == "__main__":
    unittest.main()
