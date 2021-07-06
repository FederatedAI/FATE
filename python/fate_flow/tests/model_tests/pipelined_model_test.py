import unittest
from unittest.mock import patch

import os
import io
import shutil
import hashlib
import concurrent.futures
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile

from ruamel import yaml

from fate_flow.pipelined_model.pipelined_model import PipelinedModel
from fate_flow.settings import TEMP_DIRECTORY


with open(Path(__file__).parent.parent / 'misc' / 'define_meta.yaml', encoding='utf8') as _f:
    data_define_meta = yaml.safe_load(_f)
args_update_component_meta = [
    'dataio_0',
    'DataIO',
    'dataio',
    {
        'DataIOMeta': 'DataIOMeta',
        'DataIOParam': 'DataIOParam',
    },
]


class TestPipelinedModel(unittest.TestCase):

    def setUp(self):
        shutil.rmtree(TEMP_DIRECTORY, True)

        self.pipelined_model = PipelinedModel('foobar', 'v1')
        shutil.rmtree(self.pipelined_model.model_path, True)
        self.pipelined_model.create_pipelined_model()

        with open(self.pipelined_model.define_meta_path, 'w', encoding='utf8') as f:
            yaml.dump(data_define_meta, f)

    def tearDown(self):
        shutil.rmtree(TEMP_DIRECTORY, True)
        shutil.rmtree(self.pipelined_model.model_path, True)

    def test_write_read_file_same_time(self):
        fw = open(self.pipelined_model.define_meta_path, 'r+', encoding='utf8')
        self.assertEqual(yaml.safe_load(fw), data_define_meta)
        fw.seek(0)
        fw.write('foobar')

        with open(self.pipelined_model.define_meta_path, encoding='utf8') as fr:
            self.assertEqual(yaml.safe_load(fr), data_define_meta)

        fw.truncate()

        with open(self.pipelined_model.define_meta_path, encoding='utf8') as fr:
            self.assertEqual(fr.read(), 'foobar')

        fw.seek(0)
        fw.write('abc')
        fw.close()

        with open(self.pipelined_model.define_meta_path, encoding='utf8') as fr:
            self.assertEqual(fr.read(), 'abcbar')

    def test_update_component_meta_with_changes(self):
        with patch('ruamel.yaml.dump', side_effect=yaml.dump) as yaml_dump:
            self.pipelined_model.update_component_meta(
                'dataio_0', 'DataIO_v0', 'dataio', {
                    'DataIOMeta': 'DataIOMeta_v0',
                    'DataIOParam': 'DataIOParam_v0',
                }
            )
        yaml_dump.assert_called_once()

        with open(self.pipelined_model.define_meta_path, encoding='utf8') as tmp:
            define_index = yaml.safe_load(tmp)

        _data = deepcopy(data_define_meta)
        _data['component_define']['dataio_0']['module_name'] = 'DataIO_v0'
        _data['model_proto']['dataio_0']['dataio'] = {
            'DataIOMeta': 'DataIOMeta_v0',
            'DataIOParam': 'DataIOParam_v0',
        }

        self.assertEqual(define_index, _data)

    def test_update_component_meta_without_changes(self):
        with open(self.pipelined_model.define_meta_path, 'w', encoding='utf8') as f:
            yaml.dump(data_define_meta, f, Dumper=yaml.RoundTripDumper)

        with patch('ruamel.yaml.dump', side_effect=yaml.dump) as yaml_dump:
            self.pipelined_model.update_component_meta(*args_update_component_meta)
        yaml_dump.assert_not_called()

        with open(self.pipelined_model.define_meta_path, encoding='utf8') as tmp:
            define_index = yaml.safe_load(tmp)
        self.assertEqual(define_index, data_define_meta)

    def test_update_component_meta_multi_thread(self):
        with patch('ruamel.yaml.safe_load', side_effect=yaml.safe_load) as yaml_load, \
                patch('ruamel.yaml.dump', side_effect=yaml.dump) as yaml_dump, \
                concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            for _ in range(100):
                executor.submit(self.pipelined_model.update_component_meta, *args_update_component_meta)
        self.assertEqual(yaml_load.call_count, 100)
        self.assertEqual(yaml_dump.call_count, 0)

        with open(self.pipelined_model.define_meta_path, encoding='utf8') as tmp:
            define_index = yaml.safe_load(tmp)
        self.assertEqual(define_index, data_define_meta)

    def test_update_component_meta_empty_file(self):
        open(self.pipelined_model.define_meta_path, 'w').close()
        with self.assertRaisesRegex(ValueError, 'Invalid meta file'):
            self.pipelined_model.update_component_meta(*args_update_component_meta)

    def test_packaging_model(self):
        archive_file_path = self.pipelined_model.packaging_model()
        self.assertEqual(archive_file_path, self.pipelined_model.archive_model_file_path)
        self.assertTrue(Path(archive_file_path).is_file())
        self.assertTrue(Path(archive_file_path + '.sha1').is_file())

        with ZipFile(archive_file_path) as z:
            with io.TextIOWrapper(z.open('define/define_meta.yaml'), encoding='utf8') as f:
                define_index = yaml.safe_load(f)
        self.assertEqual(define_index, data_define_meta)

        with open(archive_file_path, 'rb') as f, open(archive_file_path + '.sha1', encoding='utf8') as g:
            sha1 = hashlib.sha1(f.read()).hexdigest()
            sha1_orig = g.read().strip()
        self.assertEqual(sha1, sha1_orig)

    def test_packaging_model_not_exists(self):
        shutil.rmtree(self.pipelined_model.model_path, True)
        with self.assertRaisesRegex(FileNotFoundError, 'Can not found foobar v1 model local cache'):
            self.pipelined_model.packaging_model()

    def test_unpack_model(self):
        archive_file_path = self.pipelined_model.packaging_model()
        self.assertTrue(Path(archive_file_path + '.sha1').is_file())

        shutil.rmtree(self.pipelined_model.model_path, True)
        self.assertFalse(Path(self.pipelined_model.model_path).exists())

        self.pipelined_model.unpack_model(archive_file_path)
        with open(self.pipelined_model.define_meta_path, encoding='utf8') as tmp:
            define_index = yaml.safe_load(tmp)
        self.assertEqual(define_index, data_define_meta)

    def test_unpack_model_local_cache_exists(self):
        archive_file_path = self.pipelined_model.packaging_model()

        with self.assertRaisesRegex(FileExistsError, 'Model foobar v1 local cache already existed'):
            self.pipelined_model.unpack_model(archive_file_path)

    def test_unpack_model_no_hash_file(self):
        archive_file_path = self.pipelined_model.packaging_model()
        Path(archive_file_path + '.sha1').unlink()
        self.assertFalse(Path(archive_file_path + '.sha1').exists())

        shutil.rmtree(self.pipelined_model.model_path, True)
        self.assertFalse(os.path.exists(self.pipelined_model.model_path))

        self.pipelined_model.unpack_model(archive_file_path)
        with open(self.pipelined_model.define_meta_path, encoding='utf8') as tmp:
            define_index = yaml.safe_load(tmp)
        self.assertEqual(define_index, data_define_meta)

    def test_unpack_model_hash_not_match(self):
        archive_file_path = self.pipelined_model.packaging_model()
        self.assertTrue(Path(archive_file_path + '.sha1').is_file())
        with open(archive_file_path + '.sha1', 'w', encoding='utf8') as f:
            f.write('abc123')

        shutil.rmtree(self.pipelined_model.model_path, True)
        self.assertFalse(Path(self.pipelined_model.model_path).exists())

        with self.assertRaisesRegex(ValueError, 'Hash not match.'):
            self.pipelined_model.unpack_model(archive_file_path)


if __name__ == '__main__':
    unittest.main()
