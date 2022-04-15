########################################################
# Copyright 2019-2021 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

import time

import pulsar
from fate_arch.common import log

LOGGER = log.getLogger()
CHANNEL_TYPE_PRODUCER = "producer"
CHANNEL_TYPE_CONSUMER = "consumer"
DEFAULT_TENANT = "fl-tenant"
DEFAULT_CLUSTER = "standalone"
TOPIC_PREFIX = "{}/{}/{}"
UNIQUE_PRODUCER_NAME = "unique_producer"
UNIQUE_CONSUMER_NAME = "unique_consumer"
DEFAULT_SUBSCRIPTION_NAME = "unique"


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


# A channel cloud only be able to send or receive message.


class MQChannel(object):
    # TODO add credential to secure pulsar cluster
    def __init__(
        self,
        host,
        port,
        mng_port,
        pulsar_tenant,
        pulsar_namespace,
        pulsar_send_topic,
        pulsar_receive_topic,
        party_id,
        role,
        credential=None,
        extra_args: dict = None,
    ):
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

        # these are pulsar message id
        self._latest_confirmed = None
        self._first_confirmed = None

        self._subscription_config = {}
        if self._extra_args.get("subscription") is not None:
            self._subscription_config.update(self._extra_args["subscription"])

        self._producer_config = {}
        if self._extra_args.get("producer") is not None:
            self._producer_config.update(self._extra_args["producer"])

        self._consumer_config = {}
        if self._extra_args.get("consumer") is not None:
            self._consumer_config.update(self._extra_args["consumer"])

    @property
    def party_id(self):
        return self._party_id

    # splitting the creation of producer and producer to avoid resource wasted
    @connection_retry
    def basic_publish(self, body, properties):
        self._get_or_create_producer()
        LOGGER.debug("send queue: {}".format(self._producer_send.topic()))
        LOGGER.debug("send data size: {}".format(len(body)))

        message_id = self._producer_send.send(
            content=body, properties=properties)
        if message_id is None:
            raise Exception("publish failed")

        self._sequence_id = message_id

    @connection_retry
    def consume(self):
        self._get_or_create_consumer()

        try:
            LOGGER.debug("receive topic: {}".format(
                self._consumer_receive.topic()))
            receive_timeout = self._consumer_config.get(
                'receive_timeout_millis', None)
            if receive_timeout is not None:
                LOGGER.debug(
                    f"receive timeout millis {receive_timeout}")
            message = self._consumer_receive.receive(
                timeout_millis=receive_timeout)
            return message
        except Exception:
            self._consumer_receive.seek(pulsar.MessageId.earliest)
            raise TimeoutError("meet receive timeout, try to reset the cursor")

    @connection_retry
    def basic_ack(self, message):
        # assume consumer is alive
        try:
            self._consumer_receive.acknowledge(message)
            self._latest_confirmed = message

            if self._first_confirmed is None:
                self._first_confirmed = message
        except Exception as e:
            LOGGER.debug("meet {} when trying to ack message".format(e))
            self._get_or_create_consumer()
            self._consumer_receive.negative_acknowledge(message)

    @connection_retry
    def unack_all(self):
        self._get_or_create_consumer()
        self._consumer_receive.seek(pulsar.MessageId.earliest)

    def cancel(self):
        if self._consumer_conn is not None:
            try:
                self._consumer_receive.close()
                self._consumer_conn.close()
            except Exception as e:
                LOGGER.debug("meet {} when trying to close consumer".format(e))

            self._consumer_receive = None
            self._consumer_conn = None

        if self._producer_conn is not None:
            try:
                self._producer_send.close()
                self._producer_conn.close()
            except Exception as e:
                LOGGER.debug("meet {} when trying to close producer".format(e))

            self._producer_send = None
            self._producer_conn = None

    @connection_retry
    def _get_or_create_producer(self):
        if self._check_producer_alive() != True:
            # if self._producer_conn is None:
            try:
                self._producer_conn = pulsar.Client(
                    service_url="pulsar://{}:{}".format(
                        self._host, self._port),
                    operation_timeout_seconds=30,
                )
            except Exception as e:
                self._producer_conn = None

            # alway used current client to fetch producer
            try:
                self._producer_send = self._producer_conn.create_producer(
                    TOPIC_PREFIX.format(
                        self._tenant, self._namespace, self._send_topic
                    ),
                    producer_name=UNIQUE_PRODUCER_NAME,
                    send_timeout_millis=60000,
                    max_pending_messages=500,
                    compression_type=pulsar.CompressionType.LZ4,
                    **self._producer_config,
                )
            except Exception as e:
                LOGGER.debug(
                    f"catch exception {e} in creating pulsar producer")
                self._producer_conn = None

    @connection_retry
    def _get_or_create_consumer(self):
        if self._check_consumer_alive() != True:
            try:
                self._consumer_conn = pulsar.Client(
                    service_url="pulsar://{}:{}".format(
                        self._host, self._port),
                    operation_timeout_seconds=30,
                )
            except Exception:
                self._consumer_conn = None

            try:
                self._consumer_receive = self._consumer_conn.subscribe(
                    TOPIC_PREFIX.format(
                        self._tenant, self._namespace, self._receive_topic
                    ),
                    subscription_name=DEFAULT_SUBSCRIPTION_NAME,
                    consumer_name=UNIQUE_CONSUMER_NAME,
                    initial_position=pulsar.InitialPosition.Earliest,
                    replicate_subscription_state_enabled=True,
                    **self._subscription_config,
                )

                # set cursor to latest confirmed
                if self._latest_confirmed is not None:
                    self._consumer_receive.seek(self._latest_confirmed)

            except Exception as e:
                LOGGER.debug(
                    f"catch exception {e} in creating pulsar consumer")
                self._consumer_conn.close()
                self._consumer_conn = None

    def _check_producer_alive(self):
        if self._producer_conn is None or self._producer_send is None:
            return False

        try:
            self._producer_conn.get_topic_partitions("test-alive")
            self._producer_send.flush()
            return True
        except Exception as e:
            LOGGER.debug("catch {}, closing producer client".format(e))
            if self._producer_conn is not None:
                try:
                    self._producer_conn.close()
                except Exception:
                    pass

            self._producer_conn = None
            self._producer_send = None
            return False

    def _check_consumer_alive(self):
        try:
            if self._latest_confirmed is not None:
                self._consumer_receive.acknowledge(self._latest_confirmed)
                return True
            else:
                return False
        except Exception as e:
            self._consumer_conn = None
            self._consumer_receive = None
            return False
