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
import os
import pickle
from pathlib import Path
from datetime import datetime
from collections import deque, OrderedDict

from filelock import FileLock

from fate_flow.entity.types import RunParameters
from fate_arch.common.file_utils import get_project_base_directory


class Checkpoint:

    def __init__(self, directory: Path, step_index: int, step_name: str):
        self.step_index = step_index
        self.step_name = step_name
        self.create_time = datetime.utcnow()
        self.filepath = directory / f'{step_index}#{step_name}.pickle'
        self.locker = FileLock(f'{self.filepath}.lock')

    @property
    def available(self):
        return self.filepath.exists()

    def save(self, data):
        with self.locker, open(self.filepath, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return self.filepath

    def read(self):
        with self.locker, open(self.filepath, 'rb') as f:
            return pickle.load(f)

    def remove(self):
        with self.locker:
            os.remove(self.filepath)


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
        os.makedirs(self.directory, exist_ok=True)

        if isinstance(max_to_keep, int):
            if max_to_keep <= 0:
                raise ValueError('max_to_keep must be positive')
        elif max_to_keep is not None:
            raise TypeError('max_to_keep must be an integer')
        self.checkpoints = deque(maxlen=max_to_keep)

    def load_checkpoints_from_disk(self):
        checkpoints = []
        for filepath in self.directory.rglob('*.pickle'):
            step_index, step_name = filepath.name.rsplit('.', 1)[0].split('#', 1)
            checkpoints.append(Checkpoint(self.directory, int(step_index), step_name))

        self.checkpoints = deque(sorted((i.step_index, i) for i in checkpoints), self.max_checkpoints_number)

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
