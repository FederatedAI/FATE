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

import enum
import struct
from functools import singledispatch


class SerdeObjectTypes(enum.IntEnum):
    INT = 0
    FLOAT = 1
    STRING = 2
    BYTES = 3
    LIST = 4
    DICT = 5
    TUPLE = 6


_deserializer_registry = {}


def _register_deserializer(obj_type_enum):
    def _register(deserializer_func):
        _deserializer_registry[obj_type_enum] = deserializer_func
        return deserializer_func

    return _register


def _dispatch_deserializer(obj_type_enum):
    return _deserializer_registry[obj_type_enum]


class SafeSerdes(object):
    @staticmethod
    def serialize(obj):
        obj_type, obj_bytes = serialize_obj(obj)
        return struct.pack("!h", obj_type) + obj_bytes

    @staticmethod
    def deserialize(raw_bytes):
        (obj_type,) = struct.unpack("!h", raw_bytes[:2])
        return _dispatch_deserializer(obj_type)(raw_bytes[2:])


@singledispatch
def serialize_obj(obj):
    raise NotImplementedError("Unsupported type: {}".format(type(obj)))


@serialize_obj.register(int)
def _(obj):
    return SerdeObjectTypes.INT, struct.pack("!q", obj)


@_register_deserializer(SerdeObjectTypes.INT)
def _(raw_bytes):
    return struct.unpack("!q", raw_bytes)[0]


@serialize_obj.register(float)
def _(obj):
    return SerdeObjectTypes.FLOAT, struct.pack("!d", obj)


@_register_deserializer(SerdeObjectTypes.FLOAT)
def _(raw_bytes):
    return struct.unpack("!d", raw_bytes)[0]


@serialize_obj.register(str)
def _(obj):
    utf8_str = obj.encode("utf-8")
    return SerdeObjectTypes.STRING, struct.pack("!I", len(utf8_str)) + utf8_str


@_register_deserializer(SerdeObjectTypes.STRING)
def _(raw_bytes):
    length = struct.unpack("!I", raw_bytes[:4])[0]
    return raw_bytes[4 : 4 + length].decode("utf-8")


if __name__ == "__main__":
    print(SafeSerdes.deserialize(SafeSerdes.serialize(1)))
    print(SafeSerdes.deserialize(SafeSerdes.serialize(1.0)))
    print(SafeSerdes.deserialize(SafeSerdes.serialize("hello")))
