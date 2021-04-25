import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from copy import deepcopy
import concurrent.futures
from ruamel import yaml

from fate_flow.pipelined_model.pipelined_model import PipelinedModel


with open(Path(__file__).parent / 'define_meta.yaml', 'r', encoding='utf8') as _f:
    data = yaml.safe_load(_f)
args = [
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
        with tempfile.NamedTemporaryFile('w', encoding='utf8', delete=False) as self.tmp:
            yaml.dump(data, self.tmp)
        with open(self.tmp.name, 'r', encoding='utf8') as f:
            self.assertEqual(yaml.safe_load(f), data)

        self.pipelined_model = PipelinedModel('foobar', 'v1')
        self.pipelined_model.define_meta_path = self.tmp.name

    def tearDown(self):
        os.remove(self.tmp.name)

    def test_write_read_file_same_time(self):
        fw = open(self.tmp.name, 'r+', encoding='utf8')
        self.assertEqual(yaml.safe_load(fw), data)
        fw.seek(0)
        fw.write('foobar')

        with open(self.tmp.name, 'r', encoding='utf8') as fr:
            self.assertEqual(yaml.safe_load(fr), data)

        fw.truncate()

        with open(self.tmp.name, 'r', encoding='utf8') as fr:
            self.assertEqual(fr.read(), 'foobar')

        fw.seek(0)
        fw.write('abc')
        fw.close()

        with open(self.tmp.name, 'r', encoding='utf8') as fr:
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

        with open(self.tmp.name, 'r', encoding='utf8') as tmp:
            define_index = yaml.safe_load(tmp)

        _data = deepcopy(data)
        _data['component_define']['dataio_0']['module_name'] = 'DataIO_v0'
        _data['model_proto']['dataio_0']['dataio'] = {
            'DataIOMeta': 'DataIOMeta_v0',
            'DataIOParam': 'DataIOParam_v0',
        }

        self.assertEqual(define_index, _data)

    def test_update_component_meta_without_changes(self):
        with open(self.tmp.name, 'w', encoding='utf8') as f:
            yaml.dump(data, f, Dumper=yaml.RoundTripDumper)

        with patch('ruamel.yaml.dump', side_effect=yaml.dump) as yaml_dump:
            self.pipelined_model.update_component_meta(*args)
        yaml_dump.assert_not_called()

        with open(self.tmp.name, 'r', encoding='utf8') as tmp:
            define_index = yaml.safe_load(tmp)
        self.assertEqual(define_index, data)

    def test_update_component_meta_multi_thread(self):
        with patch('ruamel.yaml.safe_load', side_effect=yaml.safe_load) as yaml_load, \
                patch('ruamel.yaml.dump', side_effect=yaml.dump) as yaml_dump, \
                concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            for _ in range(100):
                executor.submit(self.pipelined_model.update_component_meta, *args)
        self.assertEqual(yaml_load.call_count, 100)
        self.assertEqual(yaml_dump.call_count, 0)

        with open(self.tmp.name, 'r', encoding='utf8') as tmp:
            define_index = yaml.safe_load(tmp)
        self.assertEqual(define_index, data)

    def test_update_component_meta_empty_file(self):
        open(self.tmp.name, 'w').close()

        with self.assertRaisesRegex(ValueError, 'Invalid meta file'):
            self.pipelined_model.update_component_meta(*args)


if __name__ == '__main__':
    unittest.main()
