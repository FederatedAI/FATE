#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import hashlib
from pathlib import Path
from shutil import rmtree
from datetime import datetime
from collections import deque, OrderedDict

from ruamel import yaml
from filelock import FileLock

from fate_flow.settings import stat_logger
from fate_flow.entity.types import RunParameters
from fate_flow.pipelined_model.pipelined_model import PipelinedModel
from fate_arch.protobuf.python import default_empty_fill_pb2
from fate_arch.common.file_utils import get_project_base_directory


class Checkpoint:

    def __init__(self, directory: Path, step_index: int, step_name: str):
        self.step_index = step_index
        self.step_name = step_name
        self.create_time = None
        self.directory = directory / f'{step_index}#{step_name}'
        self.directory.mkdir(0o755, True, True)
        self.database = self.directory / 'database.yaml'
        self.lock = self._lock

    @property
    def _lock(self):
        return FileLock(self.directory / '.lock')

    def __deepcopy__(self, memo):
        return self

    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('lock')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = self._lock

    @property
    def available(self):
        return self.database.exists()

    def save(self, model_buffers):
        if not model_buffers:
            raise ValueError('model_buffers is empty.')

        self.create_time = datetime.utcnow()
        data = {
            'step_index': self.step_index,
            'step_name': self.step_name,
            'create_time': self.create_time.isoformat(),
            'models': {},
        }

        model_strings = {}
        for model_name, buffer_object in model_buffers.items():
            # the type is bytes, not str
            buffer_object_serialized_string = buffer_object.SerializeToString()
            if not buffer_object_serialized_string:
                fill_message = default_empty_fill_pb2.DefaultEmptyFillMessage()
                fill_message.flag = 'set'
                buffer_object_serialized_string = fill_message.SerializeToString()
            model_strings[model_name] = buffer_object_serialized_string

            data['models'][model_name] = {
                'filename': f'{model_name}.pb',
                'sha1': hashlib.sha1(buffer_object_serialized_string).hexdigest(),
                'buffer_name': type(buffer_object).__name__,
            }

        with self.lock:
            for model_name, model in data['models'].items():
                (self.directory / model['filename']).write_bytes(model_strings[model_name])
            self.database.write_text(yaml.dump(data, Dumper=yaml.RoundTripDumper), 'utf8')
        stat_logger.info(f'Checkpoint saved. path: {self.directory}')

    def read(self):
        with self.lock:
            data = yaml.safe_load(self.database.read_text('utf8'))
            if data['step_index'] != self.step_index or data['step_name'] != self.step_name:
                raise ValueError('Checkpoint may be incorrect: step_index or step_name dose not match. '
                                 f'filepath: {self.database} '
                                 f'expected step_index: {self.step_index} actual step_index: {data["step_index"]} '
                                 f'expected step_name: {self.step_name} actual step_index: {data["step_name"]}')

            for model_name, model in data['models'].items():
                model['filepath'] = self.directory / model['filename']
                if not model['filepath'].exists():
                    raise FileNotFoundError('Checkpoint is incorrect: protobuf file not found. '
                                            f'filepath: {model["filepath"]}')

            model_strings = {model_name: model['filepath'].read_bytes()
                             for model_name, model in data['models'].items()}

        for model_name, model in data['models'].items():
            sha1 = hashlib.sha1(model_strings[model_name]).hexdigest()
            if sha1 != model['sha1']:
                raise ValueError('Checkpoint may be incorrect: hash dose not match. '
                                 f'filepath: {model["filepath"]} expected: {model["sha1"]} actual: {sha1}')

        self.create_time = datetime.fromisoformat(data['create_time'])
        return {model_name: PipelinedModel.parse_proto_object(model['buffer_name'], model_strings[model_name])
                for model_name, model in data['models'].items()}

    def remove(self):
        self.create_time = None
        rmtree(self.directory)
        self.directory.mkdir(0o755)


class CheckpointManager:

    def __init__(self, job_id: str, role: str, party_id: int,
                 model_id: str = None, model_version: str = None,
                 component_name: str = None, component_module_name: str = None,
                 task_id: str = None, task_version: int = None,
                 job_parameters: RunParameters = None,
                 max_to_keep: int = None
                 ):
        self.job_id = job_id
        self.role = role
        self.party_id = party_id
        self.model_id = model_id
        self.model_version = model_version
        self.component_name = component_name if component_name else 'pipeline'
        self.module_name = component_module_name if component_module_name else 'Pipeline'
        self.task_id = task_id
        self.task_version = task_version
        self.job_parameters = job_parameters

        self.directory = (Path(get_project_base_directory()) / 'model_local_cache' /
                          model_id / model_version / 'checkpoint' / self.component_name)
        self.directory.mkdir(0o755, True, True)

        if isinstance(max_to_keep, int):
            if max_to_keep <= 0:
                raise ValueError('max_to_keep must be positive')
        elif max_to_keep is not None:
            raise TypeError('max_to_keep must be an integer')
        self.checkpoints = deque(maxlen=max_to_keep)

    def load_checkpoints_from_disk(self):
        checkpoints = []
        for directory in self.directory.glob('*'):
            if not directory.is_dir() or '#' not in directory.name:
                continue

            step_index, step_name = directory.name.split('#', 1)
            checkpoint = Checkpoint(self.directory, int(step_index), step_name)

            if not checkpoint.available:
                continue
            checkpoints.append(checkpoint)

        self.checkpoints = deque(sorted(checkpoints, key=lambda i: i.step_index), self.max_checkpoints_number)

    @property
    def checkpoints_number(self):
        return len(self.checkpoints)

    @property
    def max_checkpoints_number(self):
        return self.checkpoints.maxlen

    @property
    def number_indexed_checkpoints(self):
        return OrderedDict((i.step_index, i) for i in self.checkpoints)

    @property
    def name_indexed_checkpoints(self):
        return OrderedDict((i.step_name, i) for i in self.checkpoints)

    def get_checkpoint_by_index(self, step_index):
        return self.number_indexed_checkpoints.get(step_index)

    def get_checkpoint_by_name(self, step_name):
        return self.name_indexed_checkpoints.get(step_name)

    @property
    def latest_checkpoint(self):
        if self.checkpoints:
            return self.checkpoints[-1]

    @property
    def latest_step_index(self):
        if self.latest_checkpoint is not None:
            return self.latest_checkpoint.step_index

    @property
    def latest_step_name(self):
        if self.latest_checkpoint is not None:
            return self.latest_checkpoint.step_name

    def new_checkpoint(self, step_index, step_name):
        popped_checkpoint = None
        if self.max_checkpoints_number and self.checkpoints_number >= self.max_checkpoints_number:
            popped_checkpoint = self.checkpoints[0]

        checkpoint = Checkpoint(self.directory, step_index, step_name)
        self.checkpoints.append(checkpoint)

        if popped_checkpoint is not None:
            popped_checkpoint.remove()

        return checkpoint

    def clean(self):
        self.checkpoints = deque(maxlen=self.max_checkpoints_number)
        rmtree(self.directory)
        self.directory.mkdir(0o755)
