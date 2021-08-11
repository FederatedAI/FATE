########################################################
# Copyright 2019-2021 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

import time

import pulsar
from pulsar import _pulsar
from fate_arch.common import log

LOGGER = log.getLogger()
CHANNEL_TYPE_PRODUCER = 'producer'
CHANNEL_TYPE_CONSUMER = 'consumer'
DEFAULT_TENANT = 'fl-tenant'
DEFAULT_CLUSTER = 'standalone'
TOPIC_PREFIX = '{}/{}/{}'
UNIQUE_PRODUCER_NAME = 'unique_producer'
UNIQUE_CONSUMER_NAME = 'unique_consumer'
DEFAULT_SUBSCRIPTION_NAME = 'unique'


def connection_retry(func):
    """retry connection
    """

    def wrapper(self, *args, **kwargs):
        """wrapper
        """
        res = None
        for ntry in range(60):
            try:
                res = func(self, *args, **kwargs)
                break
            except Exception as e:
                LOGGER.debug(e)
                time.sleep(3)
        return res
    return wrapper

 # A channel cloud only be able to send or receive message.


class MQChannel(object):
    # TODO add credential to secure pulsar cluster
    def __init__(self, host, port, mng_port, pulsar_tenant, pulsar_namespace, pulsar_send_topic, pulsar_receive_topic, party_id, role, credential=None, extra_args: dict = None):
        # "host:port" is used to connect the pulsar broker
        self._host = host
        self._port = port
        self._mng_port = mng_port
        self._tenant = pulsar_tenant
        self._namespace = pulsar_namespace
        self._send_topic = pulsar_send_topic
        self._receive_topic = pulsar_receive_topic
        self._credential = credential
        self._party_id = party_id
        self._role = role
        self._extra_args = extra_args

        # "_channel" is the subscriptor for the topic
        self._producer_send = None
        self._producer_conn = None

        self._consumer_receive = None
        self._consumer_conn = None

        self._sequence_id = None
        self._latest_confirmed = None

        self._producer_config = {}
        if extra_args.get('producer') is not None:
            self._producer_config.update(extra_args['producer'])

        self._consumer_config = {}
        if extra_args.get('consumer') is not None:
            self._consumer_config.update(extra_args['consumer'])

    @property
    def party_id(self):
        return self._party_id

    # splitting the creation of producer and producer to avoid resource wasted
    @connection_retry
    def basic_publish(self, body, properties):
        self._get_or_create_producer()
        LOGGER.debug('send queue: {}'.format(
            self._producer_send.topic()))
        LOGGER.debug('send data size: {}'.format(len(body)))

        message_id = self._producer_send.send(
            content=body, properties=properties)
        if message_id is None:
            raise Exception("publish failed")

        self._sequence_id = message_id

    @connection_retry
    def consume(self):
        self._get_or_create_consumer()

        LOGGER.debug('receive topic: {}'.format(
            self._consumer_receive.topic()))

        message = self._consumer_receive.receive()

        return message

    @connection_retry
    def basic_ack(self, message):
        self._get_or_create_consumer()
        try:
            self._consumer_receive.acknowledge(message)
            self._latest_confirmed = message
        except:
            self._consumer_receive.negative_acknowledge(message)

    def cancel(self):
        if self._consumer_conn is not None:
            try:
                self._consumer_conn.close()
            except Exception as e:
                LOGGER.debug('meet {} when trying to close consumer'.format(e))

        if self._producer_conn is not None:
            try:
                self._producer_conn.close()
            except Exception as e:
                LOGGER.debug('meet {} when trying to close producer'.format(e))

    @connection_retry
    def _get_or_create_producer(self):
        if self._check_producer_alive() != True:
            # if self._producer_conn is None:
            try:
                self._producer_conn = pulsar.Client(
                    service_url='pulsar://{}:{}'.format(
                        self._host, self._port),
                    operation_timeout_seconds=30)
            except Exception:
                self._producer_conn = None

            # alway used current client to fetch producer
            try:
                self._producer_send = self._producer_conn.create_producer(TOPIC_PREFIX.format(self._tenant, self._namespace, self._send_topic),
                                                                          producer_name=UNIQUE_PRODUCER_NAME,
                                                                          send_timeout_millis=500,
                                                                          max_pending_messages=500,
                                                                          compression_type=pulsar.CompressionType.LZ4,
                                                                          # initial_sequence_id=self._sequence_id,
                                                                          **self._producer_config)
            except Exception:
                self._producer_conn = None

    @connection_retry
    def _get_or_create_consumer(self):
        if self._check_consumer_alive() != True:
            # if self._consumer_conn is None:
            try:
                self._consumer_conn = pulsar.Client(
                    service_url='pulsar://{}:{}'.format(
                        self._host, self._port),
                    operation_timeout_seconds=30)
            except Exception:
                self._consumer_conn = None

            try:
                self._consumer_receive = self._consumer_conn.subscribe(TOPIC_PREFIX.format(self._tenant, self._namespace, self._receive_topic),
                                                                       subscription_name=DEFAULT_SUBSCRIPTION_NAME,
                                                                       consumer_name=UNIQUE_CONSUMER_NAME,
                                                                       initial_position=pulsar.InitialPosition.Earliest,
                                                                       replicate_subscription_state_enabled=True,
                                                                       **self._consumer_config)
            except Exception:
                self._consumer_conn = None

    def _check_producer_alive(self):
        if self._producer_conn is None or self._producer_send is None:
            return False

        try:
            self._producer_conn.get_topic_partitions("test-alive")
            self._producer_send.flush()
            return True
        except Exception as e:
            LOGGER.debug('catch {}, closing producer client'.format(e))
            if self._producer_conn is not None:
                try:
                    self._producer_conn.close()
                except Exception:
                    pass

            self._producer_conn = None
            self._producer_send = None
            return False

    def _check_consumer_alive(self):
        if self._consumer_conn is None or self._consumer_receive is None:
            return False

        try:
            self._consumer_conn.get_topic_partitions("test-alive")
            #message = self._consumer_receive.receive(timeout_millis=3000)
            self._consumer_receive.acknowledge(
                self._latest_confirmed)
            return True
        except Exception as e:
            LOGGER.debug('catch {}, closing consumer client'.format(e))
            if self._consumer_conn is not None:
                try:
                    self._consumer_conn.close()
                except Exception:
                    pass
            self._consumer_conn = None
            self._consumer_receive = None
            return False
