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


import pika
import time
from fate_arch.common import log

LOGGER = log.getLogger()


def connection_retry(func):
    """retry connection
    """

    def wrapper(self, *args, **kwargs):
        """wrapper
        """
        res = None
        exception = None
        for ntry in range(60):
            try:
                res = func(self, *args, **kwargs)
                exception = None
                break
            except Exception as e:
                LOGGER.error("function %s error" % func.__name__, exc_info=True)
                exception = e
                time.sleep(0.1)

        if exception is not None:
            LOGGER.exception(
                f"failed",
                exc_info=exception)
            raise exception

        return res

    return wrapper


class MQChannel(object):

    def __init__(
            self,
            host,
            port,
            user,
            password,
            vhost,
            send_queue_name,
            receive_queue_name,
            party_id,
            role,
            extra_args: dict):
        self._host = host
        self._port = port
        self._credentials = pika.PlainCredentials(user, password)
        self._vhost = vhost
        self._send_queue_name = send_queue_name
        self._receive_queue_name = receive_queue_name
        self._conn = None
        self._channel = None
        self._party_id = party_id
        self._role = role
        self._extra_args = extra_args

        if "heartbeat" not in self._extra_args:
            self._extra_args["heartbeat"] = 3600

    @property
    def party_id(self):
        return self._party_id

    @connection_retry
    def basic_publish(self, body, properties):
        self._get_channel()
        LOGGER.debug(f"send queue: {self._send_queue_name}")
        return self._channel.basic_publish(exchange='', routing_key=self._send_queue_name, body=body,
                                           properties=properties)

    @connection_retry
    def consume(self):
        self._get_channel()
        LOGGER.debug(f"receive queue: {self._receive_queue_name}")
        return self._channel.consume(queue=self._receive_queue_name)

    @connection_retry
    def basic_ack(self, delivery_tag):
        self._get_channel()
        return self._channel.basic_ack(delivery_tag=delivery_tag)

    @connection_retry
    def cancel(self):
        self._get_channel()
        return self._channel.cancel()

    @connection_retry
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
