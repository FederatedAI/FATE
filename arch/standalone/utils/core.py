#
#  Copyright 2019 The Eggroll Authors. All Rights Reserved.
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

#import ujson
import json
import time
import uuid


def get_fate_uuid():
    return uuid.uuid1().hex


def get_commit_id():
    # the model may be larger, SHA1 is not used
    return get_fate_uuid()


def string_to_bytes(string):
    return string if isinstance(string, bytes) else string.encode(encoding="utf-8")


def bytes_to_string(byte):
    return byte.decode(encoding="utf-8")


def json_dumps(src, byte=False):
    if byte:
        return string_to_bytes(json.dumps(src))
    else:
        return json.dumps(src)


def json_loads(src):
    if isinstance(src, bytes):
        return json.loads(bytes_to_string(src))
    else:
        return json.loads(src)


def current_timestamp():
    return int(time.time()*1000)
