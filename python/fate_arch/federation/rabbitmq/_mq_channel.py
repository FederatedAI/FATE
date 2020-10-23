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
from fate_arch.common import log

LOGGER = log.getLogger()


class MQChannel(object):

    def __init__(self, host, port, user, password, vhost, send_queue_name, receive_queue_name, party_id, role, extra_args: dict):
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

    @property
    def party_id(self):
        return self._party_id

    def basic_publish(self, body, properties):
        try:
            self._get_channel()
            LOGGER.debug(f"send queue: {self._send_queue_name}")
            return self._channel.basic_publish(exchange='', routing_key=self._send_queue_name, body=body,
                                               properties=properties)
        except Exception as e:
            LOGGER.error("Lost connection to rabbitmq service on manager, exception:{}.".format(e))
            self._clear()
            LOGGER.debug("Trying to reconnect...")
            self._get_channel()
            return self._channel.basic_publish(exchange='', routing_key=self._send_queue_name, body=body,
                                               properties=properties)

    def consume(self):
        try:
            self._get_channel()
            LOGGER.debug(f"receive queue: {self._receive_queue_name}")
            return self._channel.consume(queue=self._receive_queue_name)
        except Exception as e:
            LOGGER.error("Lost connection to rabbitmq service on manager, exception:{}.".format(e))
            self._clear()
            LOGGER.debug("Trying to reconnect...")
            self._get_channel()
            return self._channel.consume(queue=self._receive_queue_name)

    def basic_ack(self, delivery_tag):
        try:
            self._get_channel()
            return self._channel.basic_ack(delivery_tag=delivery_tag)
        except Exception as e:
            LOGGER.error("Lost connection to rabbitmq service on manager, exception:{}.".format(e))
            self._clear()
            LOGGER.debug("Trying to reconnect...")
            self._get_channel()
            return self._channel.basic_ack(delivery_tag=delivery_tag)

    def cancel(self):
        try:
            self._get_channel()
            return self._channel.cancel()
        except Exception as e:
            LOGGER.error("Lost connection to rabbitmq service on manager, exception:{}.".format(e))
            self._clear()
            LOGGER.debug("Trying to reconnect...")
            self._get_channel()
            return self._channel.cancel()

    def _get_channel(self):
        try:
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
        except Exception as e:
            LOGGER.error("get channel fail, exception:{}.".format(e))

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