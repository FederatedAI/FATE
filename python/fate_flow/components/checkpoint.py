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
import pickle
import hashlib
from pathlib import Path
from shutil import rmtree
from datetime import datetime, timezone
from collections import deque, OrderedDict

from filelock import FileLock

from fate_flow.settings import stat_logger
from fate_flow.entity.types import RunParameters
from fate_arch.common.file_utils import get_project_base_directory


class Checkpoint:

    def __init__(self, directory: Path, step_index: int, step_name: str):
        self.step_index = step_index
        self.step_name = step_name
        self.create_time = None
        self.filepath = directory / f'{step_index}#{step_name}.pickle'
        self.hashpath = self.filepath.with_suffix('.sha1')
        self.lock = self._lock

    @property
    def _lock(self):
        return FileLock(self.filepath.with_suffix('.lock'))

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
        return self.filepath.exists() and self.hashpath.exists()

    def save(self, data):
        pickled = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
        sha1 = hashlib.sha1(pickled).hexdigest()

        with self.lock:
            self.filepath.write_bytes(pickled)
            self.hashpath.write_text(sha1, 'utf8')
            self.create_time = datetime.utcnow()

        stat_logger.info(f'Checkpoint saved. filepath: {self.filepath} sha1: {sha1}')
        return self.filepath

    def read(self):
        if not self.hashpath.exists():
            raise FileNotFoundError(f'Hash file is not found, checkpoint may be incorrect. '
                                    f'filepath: {self.hashpath}')

        with self.lock:
            sha1_orig = self.hashpath.read_text('utf8')
            pickled = self.filepath.read_bytes()
            self.create_time = datetime.fromtimestamp(self.filepath.stat().st_mtime, tz=timezone.utc)

        sha1 = hashlib.sha1(pickled).hexdigest()
        if sha1 != sha1_orig:
            raise ValueError(f'Hash dose not match, checkpoint may be incorrect. '
                             f'expected: {sha1_orig} actual: {sha1}')

        return pickle.loads(pickled)

    def remove(self):
        with self.lock:
            for i in (self.hashpath, self.filepath):
                try:
                    i.unlink()
                except FileNotFoundError:
                    pass


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
        for filepath in self.directory.glob('*.pickle'):
            if not filepath.with_suffix('.sha1').exists():
                continue

            step_index, step_name = filepath.name.rsplit('.', 1)[0].split('#', 1)
            checkpoints.append(Checkpoint(self.directory, int(step_index), step_name))

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
        rmtree(self.directory, True)
        self.directory.mkdir(0o755)
