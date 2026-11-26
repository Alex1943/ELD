"""Regression checks for server paths; no GPU/data dependencies."""
import json
import os
import posixpath
import unittest
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import train_raw as raw_paths


class RawPathsTests(unittest.TestCase):
    def test_entrypoint_runs_without_raw_paths_module(self):
        with tempfile.TemporaryDirectory() as root:
            script = os.path.join(root, 'train_raw.py')
            shutil.copyfile(raw_paths.__file__, script)
            for name in ('manifest.csv', 'degradation_metrics/protocol.py', 'isp/model.bin'):
                filename = os.path.join(root, name)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w'):
                    pass
            cfg = {'data_root': root, 'datasetgen_root': root, 'manifest': 'manifest.csv'}
            config = os.path.join(root, 'config.json')
            with open(config, 'w') as handle:
                json.dump(cfg, handle)
            result = subprocess.run([sys.executable, script, '--config', config, '--check-paths'],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('demosaic weights:', result.stdout)

    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), '..', 'options', 'raw_eld.json'), encoding='utf-8') as handle:
            self.cfg = json.load(handle)

    def test_linux_selects_server_roots(self):
        with patch.object(raw_paths.os, 'path', posixpath):
            for key in ('data_root', 'datasetgen_root'):
                root = raw_paths.platform_root(self.cfg, key, 'Linux')
                self.assertEqual(root, self.cfg[key]['linux'])
                self.assertNotIn('D:', root)
            root = raw_paths.platform_root(self.cfg, 'data_root', 'Linux')
            self.assertEqual(raw_paths.resolve_path('manifest.csv', root, 'Linux'), root + '/manifest.csv')

    def test_windows_selects_local_roots(self):
        import ntpath
        with patch.object(raw_paths.os, 'path', ntpath):
            for key in ('data_root', 'datasetgen_root'):
                self.assertEqual(raw_paths.platform_root(self.cfg, key, 'Windows'),
                                 ntpath.normpath(self.cfg[key]['windows']))

    def test_windows_drive_rejected_on_linux(self):
        for path in ('D:/work/data', 'D:\\work\\data'):
            with self.assertRaisesRegex(ValueError, 'Windows path'):
                raw_paths.resolve_path(path, '/server/ELD', 'Linux')

    def test_server_path_rejected_on_windows(self):
        with self.assertRaisesRegex(ValueError, 'Linux server path'):
            raw_paths.resolve_path('/mnt/server/data', 'D:/ELD', 'Windows')

    def test_relative_manifest_paths_are_portable(self):
        with patch.object(raw_paths.os, 'path', posixpath):
            self.assertEqual(raw_paths.resolve_path('rain\\video\\raw.npy', '/dataset', 'Linux'),
                             '/dataset/rain/video/raw.npy')

    def test_missing_weight_fails_before_training(self):
        with patch.object(raw_paths.os.path, 'isfile', side_effect=lambda path: not path.endswith('model.bin')):
            with self.assertRaisesRegex(FileNotFoundError, 'model.bin'):
                raw_paths.check_paths(self.cfg)


if __name__ == '__main__':
    unittest.main()
