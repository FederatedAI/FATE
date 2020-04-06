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
import base64
import json
import os
import pickle
import socket
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
    return int(time.time() * 1000)


def timestamp_to_date(timestamp, format_string="%Y-%m-%d %H:%M:%S"):
    timestamp = int(timestamp) / 1000
    time_array = time.localtime(timestamp)
    str_date = time.strftime(format_string, time_array)
    return str_date


def base64_encode(src):
    return bytes_to_string(base64.b64encode(src.encode("utf-8")))


def base64_decode(src):
    return bytes_to_string(base64.b64decode(src))


def serialize_b64(src, to_str=False):
    dest = base64.b64encode(pickle.dumps(src))
    if not to_str:
        return dest
    else:
        return bytes_to_string(dest)


def deserialize_b64(src):
    return pickle.loads(base64.b64decode(string_to_bytes(src) if isinstance(src, str) else src))


def get_lan_ip():
    if os.name != "nt":
        import fcntl
        import struct

        def get_interface_ip(ifname):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return socket.inet_ntoa(
                fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', string_to_bytes(ifname[:15])))[20:24])

    ip = socket.gethostbyname(socket.getfqdn())
    if ip.startswith("127.") and os.name != "nt":
        interfaces = [
            "bond1",
            "eth0",
            "eth1",
            "eth2",
            "wlan0",
            "wlan1",
            "wifi0",
            "ath0",
            "ath1",
            "ppp0",
        ]
        for ifname in interfaces:
            try:
                ip = get_interface_ip(ifname)
                break
            except IOError as e:
                pass
    return ip or ''
