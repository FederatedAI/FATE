import unittest
from unittest.mock import patch

import pickle
import hashlib
from pathlib import Path
from collections import deque
from tempfile import TemporaryDirectory

from fate_flow.components import checkpoint


data = {
    'mydata': 'The quick brown fox jumps over the lazy dog',
}
pickled = pickle.dumps(data)
sha1 = hashlib.sha1(pickled).hexdigest()


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.checkpoint = checkpoint.Checkpoint(Path(self.tmpdir.name), 123, 'foobar')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_path(self):
        self.assertEqual(self.checkpoint.directory, Path(self.tmpdir.name) / '123#foobar')
        self.assertEqual(self.checkpoint.database, Path(self.tmpdir.name) / '123#foobar' / 'database.yaml')

    def test_save_checkpoint(self):
        self.assertFalse(self.checkpoint.available)

        self.checkpoint.save(data)
        self.assertTrue(self.checkpoint.available)
        self.assertEqual(pickle.loads(self.checkpoint.filepath.read_bytes()), data)

    def test_read_checkpoint(self):
        self.assertFalse(self.checkpoint.filepath.exists())
        self.assertFalse(self.checkpoint.hashpath.exists())

        self.checkpoint.filepath.write_bytes(pickled)
        self.assertFalse(self.checkpoint.available)

        self.checkpoint.hashpath.write_text(sha1, 'utf8')
        self.assertTrue(self.checkpoint.available)
        self.assertEqual(self.checkpoint.read(), data)

    def test_remove_checkpoint(self):
        self.checkpoint.save(data)
        self.checkpoint.remove()
        self.assertFalse(self.checkpoint.filepath.exists())
        self.assertFalse(self.checkpoint.hashpath.exists())

    def test_read_checkpoint_no_hash_file(self):
        self.checkpoint.filepath.write_bytes(pickled)
        with self.assertRaisesRegex(FileNotFoundError, 'Hash file is not found, checkpoint may be incorrect.'):
            self.checkpoint.read()

    def test_read_checkpoint_hash_not_match(self):
        self.checkpoint.filepath.write_bytes(pickled)
        self.checkpoint.hashpath.write_text('abcdef', 'utf8')
        with self.assertRaisesRegex(ValueError, 'Hash dose not match, checkpoint may be incorrect.'):
            self.checkpoint.read()


class TestCheckpointManager(unittest.TestCase):

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        with patch('fate_flow.components.checkpoint.get_project_base_directory', return_value=self.tmpdir.name):
            self.checkpoint_manager = checkpoint.CheckpointManager('job_id', 'role', 1000, 'model_id', 'model_version')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_directory(self):
        self.assertEqual(self.checkpoint_manager.directory,
                         Path(self.tmpdir.name) / 'model_local_cache' /
                         'model_id' / 'model_version' / 'checkpoint' / 'pipeline')

    def test_load_checkpoints_from_disk(self):
        for x in range(1, 51):
            (self.checkpoint_manager.directory / f'{x}#foobar{x}.pickle').write_bytes(pickled)
            (self.checkpoint_manager.directory / f'{x}#foobar{x}.sha1').write_text(sha1, 'utf8')

        self.checkpoint_manager.load_checkpoints_from_disk()
        self.assertEqual(self.checkpoint_manager.checkpoints_number, 50)
        self.assertEqual(self.checkpoint_manager.latest_step_index, 50)
        self.assertEqual(self.checkpoint_manager.latest_step_name, 'foobar50')
        self.assertEqual(self.checkpoint_manager.latest_checkpoint.read(), data)

    def test_checkpoint_index(self):
        for x in range(1, 101, 2):
            _data = data.copy()
            _data['step_index'] = x
            _data['step_name'] = f'foobar{x}'
            _pickled = pickle.dumps(_data)
            _sha1 = hashlib.sha1(_pickled).hexdigest()
            (self.checkpoint_manager.directory / f'{x}#foobar{x}.pickle').write_bytes(_pickled)
            (self.checkpoint_manager.directory / f'{x}#foobar{x}.sha1').write_text(_sha1, 'utf8')

        self.checkpoint_manager.load_checkpoints_from_disk()
        self.assertEqual(list(self.checkpoint_manager.number_indexed_checkpoints.keys()),
                         list(range(1, 101, 2)))
        self.assertEqual(list(self.checkpoint_manager.name_indexed_checkpoints.keys()),
                         [f'foobar{x}' for x in range(1, 101, 2)])

        for x in range(1, 101, 2):
            _checkpoint = self.checkpoint_manager.get_checkpoint_by_index(x)
            self.assertIs(self.checkpoint_manager.get_checkpoint_by_name(f'foobar{x}'), _checkpoint)
            self.assertEqual(_checkpoint.step_index, x)
            self.assertEqual(_checkpoint.step_name, f'foobar{x}')

            _data = _checkpoint.read()
            self.assertEqual(_data['step_index'], x)
            self.assertEqual(_data['step_name'], f'foobar{x}')

    def test_new_checkpoint(self):
        self.checkpoint_manager.checkpoints = deque(maxlen=10)

        for x in range(1, 31):
            _checkpoint = self.checkpoint_manager.new_checkpoint(x, f'foobar{x}')
            _checkpoint.save(data)
            self.assertEqual(self.checkpoint_manager.latest_step_index, x)
            self.assertEqual(self.checkpoint_manager.latest_step_name, f'foobar{x}')

        self.assertEqual(self.checkpoint_manager.checkpoints_number, 10)
        self.assertEqual(len(list(self.checkpoint_manager.directory.glob('*.pickle'))), 10)
        self.assertEqual(len(list(self.checkpoint_manager.directory.glob('*.sha1'))), 10)
        self.assertEqual(len(list(self.checkpoint_manager.directory.glob('*.lock'))), 30)

    def test_clean(self):
        for x in range(10):
            _checkpoint = self.checkpoint_manager.new_checkpoint(x, f'foobar{x}')
            _checkpoint.save(data)

        self.assertEqual(self.checkpoint_manager.checkpoints_number, 10)
        self.assertEqual(len(list(self.checkpoint_manager.directory.glob('*'))), 30)

        self.checkpoint_manager.clean()
        self.assertEqual(self.checkpoint_manager.checkpoints_number, 0)
        self.assertTrue(self.checkpoint_manager.directory.exists())
        self.assertEqual(len(list(self.checkpoint_manager.directory.glob('*'))), 0)


if __name__ == '__main__':
    unittest.main()
