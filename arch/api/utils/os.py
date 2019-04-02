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
# -*- coding: utf-8 -*-
import os
import socket
from datetime import datetime
import time


def get_lan_ip():
    if os.name != "nt":
        import fcntl
        import struct

        def get_interface_ip(ifname):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s',
                                                                                ifname[:15]))[20:24])

    #ip = socket.gethostbyname(socket.gethostname())
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


def utc_datetime_to_timestamp(utc_datetime):
    utc_timestamp = time.mktime(utc_datetime.timetuple()) * 1000.0 + utc_datetime.microsecond / 1000.0
    return utc_timestamp


def strtime_to_datetime(timestr, format='%Y-%m-%d %H:%M:%S.%f'):
    local_datetime = datetime.strptime(timestr, format)
    return local_datetime


def utc_strtime_to_timestamp(utc_timestr, format='%Y-%m-%d %H:%M:%S.%f'):
    utc_datetime = strtime_to_datetime(utc_timestr.replace(' 24:',' 00:'), format=format)
    timestamp = utc_datetime_to_timestamp(utc_datetime)
    return timestamp


def timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp/1000.0)
