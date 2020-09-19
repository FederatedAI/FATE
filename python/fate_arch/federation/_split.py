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


import pickle

import typing

__FATE_BIG_OBJ_MAX_PART_SIZE = "__fate_big_obj_max_part_size"


def segment_transfer_enabled(max_part_size=None):
    """
    a metaclass, indicate objects in this class should be transfer in segments
    Args:
        max_part_size: 0xeffe20 for eggroll 1.x and 0x3bff8800 for eggroll 2.x
    """

    if max_part_size is None:
        max_part_size = 0x3bff8800
    return _attr_injected_meta_class(**{__FATE_BIG_OBJ_MAX_PART_SIZE: max_part_size})


def _is_splitable_obj(obj):
    return hasattr(obj, __FATE_BIG_OBJ_MAX_PART_SIZE)


def _attr_injected_meta_class(**attrs):
    class _AttrInjected(type):

        def __call__(cls, *args, **kwargs):
            instance = type.__call__(cls, *args, **kwargs)
            for k, v in attrs.items():
                setattr(instance, k, v)
            return instance

    return _AttrInjected


def _get_splits(obj) -> typing.Tuple[typing.Any, typing.Iterable]:
    obj_bytes = pickle.dumps(obj, protocol=4)
    byte_size = len(obj_bytes)
    num_slice = (byte_size - 1) // getattr(obj, __FATE_BIG_OBJ_MAX_PART_SIZE) + 1
    if num_slice <= 1:
        return obj, ()
    else:
        head = _SplitHead(num_slice)
        _max_size = getattr(obj, __FATE_BIG_OBJ_MAX_PART_SIZE)
        kv = [(i, obj_bytes[slice(i * _max_size, (i + 1) * _max_size)]) for i in range(num_slice)]
        return head, kv


def _is_split_head(obj):
    return isinstance(obj, _SplitHead)


def _split_get(splits):
    obj_bytes = b''.join(splits)
    obj = pickle.loads(obj_bytes)
    return obj


class _SplitHead(object):
    def __init__(self, num_split):
        self._num_split = num_split

    def num_split(self):
        return self._num_split
