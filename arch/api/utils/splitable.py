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

__FATE_BIG_OBJ_MAX_PART_SIZE = "__fate_big_obj_max_part_size"
__SUFFIX = "_part_"


def get_slice(obj, index):
    _max_size = getattr(obj, __FATE_BIG_OBJ_MAX_PART_SIZE)
    return slice(index * _max_size, (index + 1) * _max_size)


def is_splitable_obj(obj):
    return hasattr(obj, __FATE_BIG_OBJ_MAX_PART_SIZE)


def num_split_parts(obj, bytes_size):
    return (bytes_size - 1) / getattr(obj, __FATE_BIG_OBJ_MAX_PART_SIZE) + 1


def get_split_obj_header(num_split):
    return _FragHead(num_split)


def is_split_obj_header(obj):
    return isinstance(obj, _FragHead)


def _attr_injected_meta_class(**attrs):

    class _AttrInjected(type):

        def __call__(cls, *args, **kwargs):
            instance = type.__call__(cls, *args, **kwargs)
            for k, v in attrs.items():
                setattr(instance, k, v)
            return instance

    return _AttrInjected


def segment_transfer_enabled(max_part_size=0x2000000):
    """
    a metaclass, indicate objects in this class should be transfer in segments
    Args:
        max_part_size: defaults 32MB
    """
    return _attr_injected_meta_class(__FATE_BIG_OBJ_MAX_PART_SIZE=max_part_size)


def get_part_key(prefix, index):
    return f"{prefix}{__SUFFIX}{index}"


class _FragHead(object):
    def __init__(self, num_part):
        self._num_part = num_part

    @property
    def num_part(self):
        return self._num_part
