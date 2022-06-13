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

import json
import pika

from fate_arch.common import log
from fate_arch.federation._nretry import nretry

LOGGER = log.getLogger()


class MQChannel(object):

    def __init__(self,
                 host,
                 port,
                 user,
                 password,
                 namespace,
                 vhost,
                 send_queue_name,
                 receive_queue_name,
                 src_party_id,
                 src_role,
                 dst_party_id,
                 dst_role,
                 extra_args: dict):
        self._host = host
        self._port = port
        self._credentials = pika.PlainCredentials(user, password)
        self._namespace = namespace
        self._vhost = vhost
        self._send_queue_name = send_queue_name
        self._receive_queue_name = receive_queue_name
        self._src_party_id = src_party_id
        self._src_role = src_role
        self._dst_party_id = dst_party_id
        self._dst_role = dst_role
        self._conn = None
        self._channel = None
        self._extra_args = extra_args

        if "heartbeat" not in self._extra_args:
            self._extra_args["heartbeat"] = 3600

    def __str__(self):
        return (
            f"MQChannel(host={self._host}, port={self._port}, namespace={self._namespace}, "
            f"src_party_id={self._src_party_id}, src_role={self._src_role}，"
            f"dst_party_id={self._dst_party_id}, dst_role={self._dst_role}，"
            f"send_queue_name={self._send_queue_name}, receive_queue_name={self._receive_queue_name})，"
        )

    def __repr__(self):
        return self.__str__()

    @nretry
    def produce(self, body, properties: dict):
        self._get_channel()
        LOGGER.debug(f"send queue: {self._send_queue_name}")

        if "headers" in properties:
            headers = json.loads(properties["headers"])
        else:
            headers = {}

        properties = pika.BasicProperties(
            content_type=properties["content_type"],
            app_id=properties["app_id"],
            message_id=properties["message_id"],
            correlation_id=properties["correlation_id"],
            headers=headers,
            delivery_mode=1,
        )

        return self._channel.basic_publish(exchange='', routing_key=self._send_queue_name, body=body,
                                           properties=properties)

    @nretry
    def consume(self):
        self._get_channel()
        LOGGER.debug(f"receive queue: {self._receive_queue_name}")
        return self._channel.consume(queue=self._receive_queue_name)

    @nretry
    def ack(self, delivery_tag):
        self._get_channel()
        return self._channel.basic_ack(delivery_tag=delivery_tag)

    @nretry
    def cancel(self):
        self._get_channel()
        return self._channel.cancel()

    def _get_channel(self):
        if self._check_alive():
            return
        else:
            self._clear()

        if not self._conn:
            self._conn = pika.BlockingConnection(pika.ConnectionParameters(host=self._host, port=self._port,
                                                                           virtual_host=self._vhost,
                                                                           credentials=self._credentials,
                                                                           **self._extra_args))

        if not self._channel:
            self._channel = self._conn.channel()
            self._channel.confirm_delivery()

    def _clear(self):
        try:
            if self._conn and self._conn.is_open:
                self._conn.close()
            self._conn = None

            if self._channel and self._channel.is_open:
                self._channel.close()
            self._channel = None
        except Exception as e:
            LOGGER.exception(e)
            self._conn = None
            self._channel = None

    def _check_alive(self):
        return self._channel and self._channel.is_open and self._conn and self._conn.is_open
