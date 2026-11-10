#!/usr/bin/env python3
"""Tests for supabase_sync.py — fast post-iteration backup and game archival."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import supabase_sync


class TestSupabaseSync(unittest.TestCase):
    def setUp(self):
        self._prev = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._prev)
        self._tmp.cleanup()

    @patch("supabase_sync.get_client")
    def test_backup_pod_does_not_sync_games_by_default(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create dummy state files
        with open("training_log.csv", "w") as f:
            f.write("row1\n")
        with open("config.json", "w") as f:
            f.write("{}")

        # Create dummy archive directory and game files
        os.makedirs("archive", exist_ok=True)
        with open("archive/games_12345.safetensors", "wb") as f:
            f.write(b"dummy safetensors content")

        with patch("supabase_sync.sync_archive_games") as mock_sync_games:
            supabase_sync.backup_pod(sync_games=False)
            mock_sync_games.assert_not_called()

        self.assertTrue(os.path.exists("pod_state.tar.gz"))
        # Verify games file is NOT in pod_state.tar.gz
        import tarfile
        with tarfile.open("pod_state.tar.gz") as tar:
            names = tar.getnames()
            self.assertIn("training_log.csv", names)
            self.assertIn("config.json", names)
            self.assertNotIn("archive/games_12345.safetensors", names)

    @patch("supabase_sync.get_client")
    def test_sync_iteration_uploads_expected_files(self, mock_get_client):
        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_client.storage.from_.return_value = mock_storage
        mock_get_client.return_value = mock_client

        with open("training_log.csv", "w") as f:
            f.write("header\n")
        with open("model.safetensors", "wb") as f:
            f.write(b"model bytes")
        os.makedirs("checkpoints", exist_ok=True)
        cp_path = "checkpoints/model_checkpoint_iter80.safetensors"
        with open(cp_path, "wb") as f:
            f.write(b"cp bytes")

        supabase_sync.sync_iteration(checkpoint=cp_path)

        # Check upload calls on mock_storage
        uploaded_paths = [call.kwargs.get("path") for call in mock_storage.upload.call_args_list]
        self.assertIn("pod_state.tar.gz", uploaded_paths)
        self.assertIn("model.safetensors", uploaded_paths)
        self.assertIn("model_checkpoint_iter80.safetensors", uploaded_paths)
        self.assertIn("training_log.csv", uploaded_paths)


if __name__ == "__main__":
    unittest.main()
