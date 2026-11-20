import tempfile
import unittest
from pathlib import Path

import torch

from src.models.train import create_run_directory, save_checkpoint_safely


class CheckpointSafetyTests(unittest.TestCase):
    def test_existing_checkpoint_is_preserved_without_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "lstm_b1.pt"
            original_bytes = b"existing checkpoint"
            checkpoint_path.write_bytes(original_bytes)

            with self.assertRaises(FileExistsError):
                save_checkpoint_safely(
                    {"value": torch.tensor([2])},
                    checkpoint_path,
                )

            self.assertEqual(checkpoint_path.read_bytes(), original_bytes)

    def test_explicit_overwrite_replaces_checkpoint_atomically(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "lstm_b1.pt"
            checkpoint_path.write_bytes(b"old checkpoint")

            save_checkpoint_safely(
                {"value": torch.tensor([2])},
                checkpoint_path,
                overwrite=True,
            )

            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(payload["value"].tolist(), [2])
            self.assertFalse(list(Path(temporary_directory).glob("*.tmp")))

    def test_run_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_root = Path(temporary_directory)
            run = create_run_directory(model_root, "Mission1", "run-1")

            self.assertEqual(run, model_root / "mission1" / "runs" / "run-1")
            with self.assertRaises(FileExistsError):
                create_run_directory(model_root, "Mission1", "run-1")


if __name__ == "__main__":
    unittest.main()
