#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
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
import inspect
import importlib
from pathlib import Path

from filelock import FileLock

from fate_arch.common.file_utils import get_python_base_directory
from fate_arch.protobuf.python.default_empty_fill_pb2 import DefaultEmptyFillMessage
from fate_flow.settings import stat_logger


def serialize_buffer_object(buffer_object):
    # the type is bytes, not str
    serialized_string = buffer_object.SerializeToString()
    if not serialized_string:
        fill_message = DefaultEmptyFillMessage()
        fill_message.flag = 'set'
        serialized_string = fill_message.SerializeToString()
    return serialized_string


def get_proto_buffer_class(buffer_name):
    package_path = Path(get_python_base_directory('federatedml', 'protobuf', 'generated'))
    package_module = 'federatedml.protobuf.generated'

    e = ModuleNotFoundError(f'No module named {buffer_name}')
    for f in package_path.glob('*.py'):
        try:
            proto_module = importlib.import_module('.'.join([package_module, f.stem]))
            for name, obj in inspect.getmembers(proto_module):
                if inspect.isclass(obj) and name == buffer_name:
                    return obj
        except Exception as e:
            stat_logger.warning(e)
    raise e


def parse_proto_object(buffer_name, serialized_string):
    try:
        buffer_object = get_proto_buffer_class(buffer_name)()
    except Exception as e:
        stat_logger.exception('Can not restore proto buffer object', e)
        raise e
    buffer_name = type(buffer_object).__name__

    try:
        buffer_object.ParseFromString(serialized_string)
    except Exception as e1:
        stat_logger.exception(e1)
        try:
            DefaultEmptyFillMessage().ParseFromString(serialized_string)
            buffer_object.ParseFromString(bytes())
        except Exception as e2:
            stat_logger.exception(e2)
            raise e1
        else:
            stat_logger.info(f'parse {buffer_name} proto object with default values')
    else:
        stat_logger.info(f'parse {buffer_name} proto object normal')
    return buffer_object


class Locker:

    def __init__(self, directory):
        if isinstance(directory, str):
            directory = Path(directory)
        self.directory = directory
        self.lock = self._lock

    @property
    def _lock(self):
        return FileLock(self.directory / '.lock')

    def __copy__(self):
        return self

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
