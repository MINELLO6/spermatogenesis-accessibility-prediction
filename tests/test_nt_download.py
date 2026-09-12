"""Offline tests: no model download or GPU is required."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.download_nt_model import FILES, MANIFEST, MODEL_ID, REVISION, prepare_model


def fake_download(**kwargs):
    directory = Path(kwargs["local_dir"])
    for name in FILES:
        (directory / name).write_text("test fixture", encoding="utf-8")
    (directory / "config.json").write_text(
        json.dumps({"hidden_size": 512, "num_hidden_layers": 12}), encoding="utf-8"
    )


class DownloadTests(unittest.TestCase):
    def test_pinned_snapshot_and_offline_check(self):
        with tempfile.TemporaryDirectory() as temp:
            download = Mock(side_effect=fake_download)
            destination = prepare_model(temp, downloader=download)
            args = download.call_args.kwargs
            self.assertEqual(args["revision"], REVISION)
            self.assertEqual(args["repo_id"], MODEL_ID)
            self.assertNotIn("pytorch_model.bin", args["allow_patterns"])
            self.assertEqual(prepare_model(temp, check_only=True), destination)
            self.assertEqual(download.call_count, 1)
            (destination / "vocab.txt").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                prepare_model(temp, check_only=True)

    def test_unknown_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            old = Path(temp) / "config.json"
            old.write_text("existing data", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                prepare_model(temp, downloader=Mock())
            self.assertEqual(old.read_text(encoding="utf-8"), "existing data")

    def test_interrupted_download_can_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(OSError):
                prepare_model(temp, downloader=Mock(side_effect=OSError("interrupted")))
            with self.assertRaisesRegex(ValueError, "did not complete"):
                prepare_model(temp, check_only=True)
            prepare_model(temp, downloader=fake_download)
            self.assertTrue(json.loads((Path(temp) / MANIFEST).read_text())["complete"])

    def test_wrong_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / MANIFEST).write_text(
                json.dumps({"repo_id": MODEL_ID, "revision": "other"})
            )
            with self.assertRaisesRegex(ValueError, "different model"):
                prepare_model(temp, downloader=Mock())


if __name__ == "__main__":
    unittest.main()
