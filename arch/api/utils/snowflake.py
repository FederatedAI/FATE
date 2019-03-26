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
import datetime
import time
import os
import random
import socket
from arch.api.utils.os import get_lan_ip
from arch.api.utils.os import utc_strtime_to_timestamp

# twitter's snowflake parameters
twepoch = utc_strtime_to_timestamp('1992-09-30 00:00:00.777')
#host_id = int(socket.inet_aton(get_lan_ip()).encode('hex'), 16)
host_id_bits = 32
worker_id_bits = 15
sequence_id_bits = 12
max_host_id = 1 << host_id_bits
max_worker_id = 1 << worker_id_bits
max_sequence_id = 1 << sequence_id_bits
max_timestamp = 1 << (100 - host_id_bits - worker_id_bits - sequence_id_bits)


def get_id(host_id="xx"):
    timestamp_ms = int(time.time() * 1000)
    worker_id = os.getpid()
    sequence_id = random.randrange(0, 2**12)
    return make_snowflake(timestamp_ms, host_id, worker_id, sequence_id)


def make_snowflake(timestamp_ms, host_id, worker_id, sequence_id, twepoch=twepoch):
    """generate a twitter-snowflake id, based on 
    https://github.com/twitter/snowflake/blob/master/src/main/scala/com/twitter/service/snowflake/IdWorker.scala
    :param: timestamp_ms time since UNIX epoch in milliseconds"""

    sid = ((int(timestamp_ms) - twepoch) % max_timestamp) << host_id_bits << worker_id_bits << sequence_id_bits
    sid += (host_id % max_host_id) << worker_id_bits << sequence_id_bits
    sid += (worker_id % max_worker_id) << sequence_id_bits
    sid += sequence_id % max_sequence_id

    return sid


def melt(snowflake_id, twepoch=twepoch):
    """inversely transform a snowflake id back to its parts."""
    sequence_id = snowflake_id & (max_sequence_id - 1)
    worker_id = (snowflake_id >> sequence_id_bits) & (max_worker_id - 1)
    host_id = (snowflake_id >> sequence_id_bits >> worker_id_bits) & (max_host_id - 1)
    timestamp_ms = snowflake_id >> sequence_id_bits >> worker_id_bits >> host_id_bits
    timestamp_ms += twepoch
    return (timestamp_ms, int(host_id), int(worker_id), int(sequence_id))


def local_datetime(timestamp_ms):
    """convert millisecond timestamp to local datetime object."""
    return datetime.datetime.fromtimestamp(timestamp_ms / 1000.)