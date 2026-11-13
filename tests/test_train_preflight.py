import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_tu = importlib.import_module("hardonian_foundry.training_utils")
TrainingLockError = _tu.TrainingLockError
acquire_training_lock = _tu.acquire_training_lock
available_ram_bytes = _tu.available_ram_bytes
git_blob_sha1 = _tu.git_blob_sha1
model_registry = _tu.model_registry
resolve_model_spec = _tu.resolve_model_spec
select_device = _tu.select_device
sha_bytes = _tu.sha_bytes
verify_snapshot = _tu.verify_snapshot

SCRIPT = ROOT / "scripts" / "train_synthetic_lora.py"
spec = importlib.util.spec_from_file_location("synthetic_lora", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)


def _make_open_dir(path: Path) -> None:
    if os.name == "nt":
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["icacls", str(path), "/grant", "*S-1-1-0:(OI)(CI)R"],
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        path.mkdir(mode=0o755, parents=True, exist_ok=True)
        path.chmod(0o755)


class TrainingPreflightTest(unittest.TestCase):
    def test_synthetic_holdout_is_disjoint_and_deterministic(self):
        data = module.examples()
        self.assertEqual(data, module.examples())
        self.assertEqual((len(data["train"]), len(data["holdout"])), (12, 4))
        self.assertFalse({x["prompt"] for x in data["train"]} & {x["prompt"] for x in data["holdout"]})
        self.assertTrue(all(x["id"].startswith("synthetic-") for rows in data.values() for x in rows))
        self.assertEqual(len(module.sha_bytes(json.dumps(data).encode())), 64)

    def test_verified_snapshot_is_passed_to_both_loaders(self):
        calls = []

        class Loader:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append((path, kwargs))
                return Loader()

            def to(self, device):
                self.device = device
                return self

        class Torch:
            float32 = "float32"

        snapshot = Path("/verified/local/snapshot")
        _, model = module.load_verified_snapshot(snapshot, Loader, Loader, Torch)
        self.assertEqual([call[0] for call in calls], [str(snapshot), str(snapshot)])
        self.assertTrue(
            all(
                call[1]["local_files_only"] and not call[1]["trust_remote_code"] and not call[1]["token"]
                for call in calls
            )
        )
        self.assertEqual(model.device, "cpu")

    def test_insecure_output_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "open"
            _make_open_dir(root)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--hf-home", tmp, "--output-root", str(root)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("owner-only", result.stderr)
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_model_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--hf-home", tmp, "--output-root", tmp],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no download attempted", result.stderr)
            self.assertEqual(os.listdir(tmp), [])

    def test_steps_out_of_range_rejected(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--hf-home", ".", "--output-root", ".", "--steps", "21"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_verify_snapshot_checks_weights_blobs_and_auto_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "snap"
            snap.mkdir()
            weights = b"fake-safetensors-payload"
            cfg_bytes = b'{"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}\n'
            (snap / "model.safetensors").write_bytes(weights)
            (snap / "config.json").write_bytes(cfg_bytes)
            blobs = {"config.json": git_blob_sha1(cfg_bytes)}

            info = verify_snapshot(snap, model_sha256=sha_bytes(weights), blobs=blobs)
            self.assertEqual(info["weights_sha256"], sha_bytes(weights))

            # Reject auto_map custom remote code even if matching updated blob hash
            bad_cfg = b'{"auto_map": {"AutoModelForCausalLM": "remote.Model"}}\n'
            (snap / "config.json").write_bytes(bad_cfg)
            with self.assertRaisesRegex(ValueError, "auto_map"):
                verify_snapshot(snap, model_sha256=sha_bytes(weights), blobs={"config.json": git_blob_sha1(bad_cfg)})

            # Reject tampered metadata blob
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                verify_snapshot(snap, model_sha256=sha_bytes(weights), blobs=blobs)

            # Reject tampered weights
            (snap / "config.json").write_bytes(cfg_bytes)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_snapshot(snap, model_sha256="f" * 64, blobs=blobs)

    def test_available_ram_bytes_and_select_device(self):
        ram = available_ram_bytes()
        self.assertIsInstance(ram, int)
        self.assertGreater(ram, 0)

        cpu_spec = select_device("cpu")
        self.assertEqual(cpu_spec.device, "cpu")
        self.assertFalse(cpu_spec.allow_gpu)
        self.assertEqual(cpu_spec.torch_dtype_name, "float32")
        with self.assertRaises(ValueError):
            select_device("tpu-unknown")

    def test_model_registry_loads_defaults_and_custom_models(self):
        defaults = model_registry()
        self.assertIn("Qwen/Qwen2.5-Coder-0.5B-Instruct", defaults)

        with tempfile.TemporaryDirectory() as tmp:
            reg_path = Path(tmp) / "registry.json"
            reg_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "org/tiny-test": {
                                "revision": "rev123",
                                "model_sha256": "a" * 64,
                                "lora_target_modules": ["q_proj", "k_proj", "v_proj"],
                                "weight_files": {"model-00001.safetensors": "b" * 64},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            custom = resolve_model_spec("org/tiny-test", registry_path=reg_path)
            self.assertEqual(custom.revision, "rev123")
            self.assertEqual(tuple(custom.lora_target_modules), ("q_proj", "k_proj", "v_proj"))
            self.assertEqual(custom.weight_files, {"model-00001.safetensors": "b" * 64})
            with self.assertRaises(ValueError):
                resolve_model_spec("org/missing", registry_path=reg_path)

    def test_acquire_training_lock_contention_across_processes(self):
        probe_code = (
            "import sys\n"
            f"sys.path.insert(0, {str(SRC)!r})\n"
            "from hardonian_foundry.training_utils import acquire_training_lock, TrainingLockError\n"
            "try:\n"
            "    with acquire_training_lock(sys.argv[1]):\n"
            "        print('ACQUIRED')\n"
            "except TrainingLockError as exc:\n"
            "    print('LOCKED:' + str(exc))\n"
            "    sys.exit(2)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            lock_file = Path(tmp) / ".training.lock"
            with acquire_training_lock(lock_file):
                res = subprocess.run(
                    [sys.executable, "-c", probe_code, str(lock_file)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(res.returncode, 2)
                self.assertIn("LOCKED:", res.stdout)

            res_after = subprocess.run(
                [sys.executable, "-c", probe_code, str(lock_file)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(res_after.returncode, 0)
            self.assertIn("ACQUIRED", res_after.stdout)

        with self.assertRaises(TrainingLockError):
            raise TrainingLockError("smoke")


if __name__ == "__main__":
    unittest.main()
