import unittest
from unittest.mock import patch

import hashlib
from pathlib import Path
from datetime import datetime
from collections import deque
from tempfile import TemporaryDirectory

from ruamel import yaml

from fate_flow.components import checkpoint


model_string = (Path(__file__).parent.parent / 'misc' / 'DataIOMeta.pb').read_bytes()
sha1 = hashlib.sha1(model_string).hexdigest()
buffer_name = 'DataIOMeta'
model_buffers = {
    'my_model': checkpoint.PipelinedModel.parse_proto_object(buffer_name, model_string),
}
data = yaml.dump({
    'step_index': 123,
    'step_name': 'foobar',
    'create_time': '2021-07-08T07:51:01.963423',
    'models': {
        'my_model': {
            'filename': 'my_model.pb',
            'sha1': sha1,
            'buffer_name': buffer_name,
        },
    },
}, Dumper=yaml.RoundTripDumper)


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.checkpoint = checkpoint.Checkpoint(Path(self.tmpdir.name), 123, 'foobar')
        self.filepath = self.checkpoint.directory / 'my_model.pb'

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_path(self):
        directory = Path(self.tmpdir.name) / '123#foobar'
        self.assertEqual(self.checkpoint.directory, directory)
        self.assertEqual(self.checkpoint.database, directory / 'database.yaml')

    def test_save_checkpoint(self):
        self.assertTrue(self.checkpoint.directory.exists())
        self.assertFalse(self.checkpoint.available)
        self.assertFalse(self.filepath.exists())
        self.assertIsNone(self.checkpoint.create_time)

        self.checkpoint.save(model_buffers)
        self.assertTrue(self.checkpoint.available)
        self.assertTrue(self.filepath.exists())
        self.assertIsNotNone(self.checkpoint.create_time)

        self.assertEqual(self.checkpoint.database.read_text('utf8'),
                         data.replace('2021-07-08T07:51:01.963423', self.checkpoint.create_time.isoformat()), 1)
        self.assertEqual(self.filepath.read_bytes(), model_string)

    def test_read_checkpoint(self):
        self.assertTrue(self.checkpoint.directory.exists())
        self.assertFalse(self.checkpoint.available)
        self.assertFalse(self.filepath.exists())

        self.filepath.write_bytes(model_string)
        self.assertFalse(self.checkpoint.available)

        self.checkpoint.database.write_text(data, 'utf8')
        self.assertTrue(self.checkpoint.available)
        self.assertIsNone(self.checkpoint.create_time)

        self.assertEqual(self.checkpoint.read(), model_buffers)
        self.assertEqual(self.checkpoint.step_index, 123)
        self.assertEqual(self.checkpoint.step_name, 'foobar')
        self.assertEqual(self.checkpoint.create_time, datetime.fromisoformat('2021-07-08T07:51:01.963423'))

    def test_remove_checkpoint(self):
        self.checkpoint.save(model_buffers)
        self.checkpoint.database.write_text(data, 'utf8')
        self.checkpoint.remove()

        self.assertTrue(self.checkpoint.directory.exists())
        self.assertFalse(self.filepath.exists())
        self.assertFalse(self.checkpoint.available)
        self.assertIsNone(self.checkpoint.create_time)

    def test_read_checkpoint_step_index_or_step_name_not_match(self):
        self.filepath.write_bytes(model_string)
        self.checkpoint.database.write_text(data.replace('123', '233', 1), 'utf8')
        with self.assertRaisesRegex(ValueError, 'Checkpoint may be incorrect: step_index or step_name dose not match.'):
            self.checkpoint.read()

    def test_read_checkpoint_no_pb_file(self):
        self.checkpoint.database.write_text(data, 'utf8')
        with self.assertRaisesRegex(FileNotFoundError, 'Checkpoint is incorrect: protobuf file not found.'):
            self.checkpoint.read()

    def test_read_checkpoint_hash_not_match(self):
        self.filepath.write_bytes(model_string)
        self.checkpoint.database.write_text(data.replace(sha1, 'abcdef', 1), 'utf8')
        with self.assertRaisesRegex(ValueError, 'Checkpoint may be incorrect: hash dose not match.'):
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
            directory = self.checkpoint_manager.directory / f'{x}#foobar{x}'
            directory.mkdir(0o755)

            (directory / 'my_model.pb').write_bytes(model_string)
            (directory / 'database.yaml').write_text(
                data.replace('123', str(x), 1).replace('foobar', f'foobar{x}', 1), 'utf8')

        self.checkpoint_manager.load_checkpoints_from_disk()
        self.assertEqual(self.checkpoint_manager.checkpoints_number, 50)
        self.assertEqual(self.checkpoint_manager.latest_step_index, 50)
        self.assertEqual(self.checkpoint_manager.latest_step_name, 'foobar50')
        self.assertEqual(self.checkpoint_manager.latest_checkpoint.read(), model_buffers)

    def test_checkpoint_index(self):
        for x in range(1, 101, 2):
            directory = self.checkpoint_manager.directory / f'{x}#foobar{x}'
            directory.mkdir(0o755)

            (directory / 'my_model.pb').write_bytes(model_string)
            (directory / 'database.yaml').write_text(
                data.replace('123', str(x), 1).replace('foobar', f'foobar{x}', 1), 'utf8')

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
            self.assertIsNone(_checkpoint.create_time)

            _model_buffers = _checkpoint.read()
            self.assertEqual(_checkpoint.step_index, x)
            self.assertEqual(_checkpoint.step_name, f'foobar{x}')
            self.assertEqual(_checkpoint.create_time.isoformat(), '2021-07-08T07:51:01.963423')

    def test_new_checkpoint(self):
        self.checkpoint_manager.checkpoints = deque(maxlen=10)

        for x in range(1, 31):
            _checkpoint = self.checkpoint_manager.new_checkpoint(x, f'foobar{x}')
            _checkpoint.save(model_buffers)
            self.assertEqual(self.checkpoint_manager.latest_step_index, x)
            self.assertEqual(self.checkpoint_manager.latest_step_name, f'foobar{x}')

        self.assertEqual(self.checkpoint_manager.checkpoints_number, 10)
        self.assertEqual(len(list(self.checkpoint_manager.directory.rglob('my_model.pb'))), 10)
        self.assertEqual(len(list(self.checkpoint_manager.directory.rglob('database.yaml'))), 10)
        self.assertEqual(len(list(self.checkpoint_manager.directory.rglob('.lock'))), 10)
        self.assertEqual(len(list(self.checkpoint_manager.directory.glob('*'))), 30)

    def test_clean(self):
        for x in range(10):
            _checkpoint = self.checkpoint_manager.new_checkpoint(x, f'foobar{x}')
            _checkpoint.save(model_buffers)

        self.assertEqual(self.checkpoint_manager.checkpoints_number, 10)
        self.assertEqual(len(list(self.checkpoint_manager.directory.glob('*'))), 10)

        self.checkpoint_manager.clean()
        self.assertEqual(self.checkpoint_manager.checkpoints_number, 0)
        self.assertTrue(self.checkpoint_manager.directory.exists())
        self.assertEqual(len(list(self.checkpoint_manager.directory.glob('*'))), 0)


if __name__ == '__main__':
    unittest.main()
