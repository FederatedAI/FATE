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

from fate_arch.common import Party
from fate_arch.common import file_utils
from fate_arch.common.log import getLogger
from fate_arch.federation._federation import FederationBase
from fate_arch.federation.rabbitmq._mq_channel import MQChannel
from fate_arch.federation.rabbitmq._rabbit_manager import RabbitManager

LOGGER = getLogger()

# default message max size in bytes = 1MB
DEFAULT_MESSAGE_MAX_SIZE = 1048576


class MQ(object):
    def __init__(self, host, port, union_name, policy_id, route_table):
        self.host = host
        self.port = port
        self.union_name = union_name
        self.policy_id = policy_id
        self.route_table = route_table

    def __str__(self):
        return (
            f"MQ(host={self.host}, port={self.port}, union_name={self.union_name}, "
            f"policy_id={self.policy_id}, route_table={self.route_table})"
        )

    def __repr__(self):
        return self.__str__()


class _TopicPair(object):
    def __init__(self, tenant=None, namespace=None, vhost=None, send=None, receive=None):
        self.tenant = tenant
        self.namespace = namespace
        self.vhost = vhost
        self.send = send
        self.receive = receive


class Federation(FederationBase):
    @staticmethod
    def from_conf(
            federation_session_id: str,
            party: Party,
            runtime_conf: dict,
            **kwargs
    ):
        rabbitmq_config = kwargs["rabbitmq_config"]
        LOGGER.debug(f"rabbitmq_config: {rabbitmq_config}")
        host = rabbitmq_config.get("host")
        port = rabbitmq_config.get("port")
        mng_port = rabbitmq_config.get("mng_port")
        base_user = rabbitmq_config.get("user")
        base_password = rabbitmq_config.get("password")
        mode = rabbitmq_config.get("mode", "replication")
        # max_message_size；
        max_message_size = int(rabbitmq_config.get("max_message_size", DEFAULT_MESSAGE_MAX_SIZE))

        union_name = federation_session_id
        policy_id = federation_session_id

        rabbitmq_run = runtime_conf.get("job_parameters", {}).get("rabbitmq_run", {})
        LOGGER.debug(f"rabbitmq_run: {rabbitmq_run}")

        max_message_size = int(rabbitmq_run.get(
            "max_message_size", max_message_size))

        LOGGER.debug(f"set max message size to {max_message_size} Bytes")

        rabbit_manager = RabbitManager(
            base_user, base_password, f"{host}:{mng_port}", rabbitmq_run
        )
        rabbit_manager.create_user(union_name, policy_id)
        route_table_path = rabbitmq_config.get("route_table")
        if route_table_path is None:
            route_table_path = "conf/rabbitmq_route_table.yaml"
        route_table = file_utils.load_yaml_conf(conf_path=route_table_path)
        mq = MQ(host, port, union_name, policy_id, route_table)
        conf = rabbit_manager.runtime_config.get(
            "connection", {}
        )

        return Federation(
            federation_session_id, party, mq, rabbit_manager, max_message_size, conf, mode
        )

    def __init__(self, session_id, party: Party, mq: MQ, rabbit_manager: RabbitManager, max_message_size, conf, mode):
        super().__init__(session_id=session_id, party=party, mq=mq, max_message_size=max_message_size, conf=conf)
        self._rabbit_manager = rabbit_manager
        self._vhost_set = set()
        self._mode = mode

    def __getstate__(self):
        pass

    def destroy(self, parties):
        LOGGER.debug("[rabbitmq.cleanup]start to cleanup...")
        for party in parties:
            if self._party == party:
                continue
            vhost = self._get_vhost(party)
            LOGGER.debug(f"[rabbitmq.cleanup]start to cleanup vhost {vhost}...")
            self._rabbit_manager.clean(vhost)
            LOGGER.debug(f"[rabbitmq.cleanup]cleanup vhost {vhost} done")
        if self._mq.union_name:
            LOGGER.debug(f"[rabbitmq.cleanup]clean user {self._mq.union_name}.")
            self._rabbit_manager.delete_user(user=self._mq.union_name)

    def _get_vhost(self, party):
        low, high = (
            (self._party, party) if self._party < party else (party, self._party)
        )
        vhost = (
            f"{self._session_id}-{low.role}-{low.party_id}-{high.role}-{high.party_id}"
        )
        return vhost

    def _maybe_create_topic_and_replication(self, party, topic_suffix):
        if self._mode == "replication":
            return self._create_topic_by_replication_mode(party, topic_suffix)

        if self._mode == "client":
            return self._create_topic_by_client_mode(party, topic_suffix)

        raise ValueError("mode={self._mode} is not support!")

    def _create_topic_by_client_mode(self, party, topic_suffix):
        # gen names
        vhost_name = self._get_vhost(party)
        send_queue_name = f"{self._session_id}-{self._party.role}-{self._party.party_id}-{party.role}-{party.party_id}-{topic_suffix}"
        receive_queue_name = f"{self._session_id}-{party.role}-{party.party_id}-{self._party.role}-{self._party.party_id}-{topic_suffix}"

        topic_pair = _TopicPair(
            namespace=self._session_id,
            vhost=vhost_name,
            send=send_queue_name,
            receive=receive_queue_name
        )

        # initial vhost
        if topic_pair.vhost not in self._vhost_set:
            self._rabbit_manager.create_vhost(topic_pair.vhost)
            self._rabbit_manager.add_user_to_vhost(
                self._mq.union_name, topic_pair.vhost
            )
            self._vhost_set.add(topic_pair.vhost)

        # initial send queue, the name is send-${vhost}
        self._rabbit_manager.create_queue(topic_pair.vhost, topic_pair.send)
        # initial receive queue, the name is receive-${vhost}
        self._rabbit_manager.create_queue(topic_pair.vhost, topic_pair.receive)

        return topic_pair

    def _create_topic_by_replication_mode(self, party, topic_suffix):
        # gen names
        vhost_name = self._get_vhost(party)
        send_queue_name = f"send-{self._session_id}-{self._party.role}-{self._party.party_id}-{party.role}-{party.party_id}-{topic_suffix}"
        receive_queue_name = f"receive-{self._session_id}-{party.role}-{party.party_id}-{self._party.role}-{self._party.party_id}-{topic_suffix}"

        topic_pair = _TopicPair(
            namespace=self._session_id,
            vhost=vhost_name,
            send=send_queue_name,
            receive=receive_queue_name
        )

        # initial vhost
        if topic_pair.vhost not in self._vhost_set:
            self._rabbit_manager.create_vhost(topic_pair.vhost)
            self._rabbit_manager.add_user_to_vhost(
                self._mq.union_name, topic_pair.vhost
            )
            self._vhost_set.add(topic_pair.vhost)

        # initial send queue, the name is send-${vhost}
        self._rabbit_manager.create_queue(topic_pair.vhost, topic_pair.send)

        # initial receive queue, the name is receive-${vhost}
        self._rabbit_manager.create_queue(
            topic_pair.vhost, topic_pair.receive
        )

        upstream_uri = self._upstream_uri(party_id=party.party_id)
        self._rabbit_manager.federate_queue(
            upstream_host=upstream_uri,
            vhost=topic_pair.vhost,
            send_queue_name=topic_pair.send,
            receive_queue_name=topic_pair.receive,
        )

        return topic_pair

    def _upstream_uri(self, party_id):
        host = self._mq.route_table.get(int(party_id)).get("host")
        port = self._mq.route_table.get(int(party_id)).get("port")
        upstream_uri = (
            f"amqp://{self._mq.union_name}:{self._mq.policy_id}@{host}:{port}"
        )
        return upstream_uri

    def _get_channel(
            self, topic_pair, src_party_id, src_role, dst_party_id, dst_role, mq=None, conf: dict = None):
        LOGGER.debug(f"rabbitmq federation _get_channel, src_party_id={src_party_id}, src_role={src_role},"
                     f"dst_party_id={dst_party_id}, dst_role={dst_role}")
        return MQChannel(
            host=mq.host,
            port=mq.port,
            user=mq.union_name,
            password=mq.policy_id,
            namespace=topic_pair.namespace,
            vhost=topic_pair.vhost,
            send_queue_name=topic_pair.send,
            receive_queue_name=topic_pair.receive,
            src_party_id=src_party_id,
            src_role=src_role,
            dst_party_id=dst_party_id,
            dst_role=dst_role,
            extra_args=conf,
        )

    def _get_consume_message(self, channel_info):
        for method, properties, body in channel_info.consume():
            LOGGER.debug(
                f"[rabbitmq._get_consume_message] method: {method}, properties: {properties}"
            )

            properties = {
                "message_id": properties.message_id,
                "correlation_id": properties.correlation_id,
                "content_type": properties.content_type,
                "headers": json.dumps(properties.headers)
            }

            yield method.delivery_tag, properties, body

    def _consume_ack(self, channel_info, id):
        channel_info.ack(delivery_tag=id)
