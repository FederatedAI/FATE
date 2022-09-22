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

from fate_arch.common import Party
from fate_arch.common import file_utils
from fate_arch.common.log import getLogger
from fate_arch.federation._federation import FederationBase
from fate_arch.federation.pulsar._mq_channel import (
    MQChannel,
    DEFAULT_TENANT,
    DEFAULT_CLUSTER,
    DEFAULT_SUBSCRIPTION_NAME,
)
from fate_arch.federation.pulsar._pulsar_manager import PulsarManager

LOGGER = getLogger()
# default message max size in bytes = 1MB
DEFAULT_MESSAGE_MAX_SIZE = 104857


class MQ(object):
    def __init__(self, host, port, route_table):
        self.host = host
        self.port = port
        self.route_table = route_table

    def __str__(self):
        return (
            f"MQ(host={self.host}, port={self.port} "
            f"route_table={self.route_table}), "
            f"type=pulsar"
        )

    def __repr__(self):
        return self.__str__()


class _TopicPair(object):
    def __init__(self, tenant, namespace, send, receive):
        self.tenant = tenant
        self.namespace = namespace
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
        pulsar_config = kwargs["pulsar_config"]
        LOGGER.debug(f"pulsar_config: {pulsar_config}")
        host = pulsar_config.get("host", "localhost")
        port = pulsar_config.get("port", "6650")
        mng_port = pulsar_config.get("mng_port", "8080")
        topic_ttl = int(pulsar_config.get("topic_ttl", 0))
        cluster = pulsar_config.get("cluster", DEFAULT_CLUSTER)
        # tenant name should be unified between parties
        tenant = pulsar_config.get("tenant", DEFAULT_TENANT)

        # max_message_size；
        max_message_size = int(pulsar_config.get("max_message_size", DEFAULT_MESSAGE_MAX_SIZE))

        pulsar_run = runtime_conf.get(
            "job_parameters", {}).get("pulsar_run", {})
        LOGGER.debug(f"pulsar_run: {pulsar_run}")

        max_message_size = int(pulsar_run.get(
            "max_message_size", max_message_size))

        LOGGER.debug(f"set max message size to {max_message_size} Bytes")

        # topic ttl could be overwritten by run time config
        topic_ttl = int(pulsar_run.get("topic_ttl", topic_ttl))

        # pulsar not use user and password so far
        # TODO add credential to connections
        base_user = pulsar_config.get("user")
        base_password = pulsar_config.get("password")
        mode = pulsar_config.get("mode", "replication")

        pulsar_manager = PulsarManager(
            host=host, port=mng_port, runtime_config=pulsar_run
        )

        # init tenant
        tenant_info = pulsar_manager.get_tenant(tenant=tenant).json()
        if tenant_info.get("allowedClusters") is None:
            pulsar_manager.create_tenant(
                tenant=tenant, admins=[], clusters=[cluster])

        route_table_path = pulsar_config.get("route_table")
        if route_table_path is None:
            route_table_path = "conf/pulsar_route_table.yaml"
        route_table = file_utils.load_yaml_conf(conf_path=route_table_path)
        mq = MQ(host, port, route_table)

        conf = pulsar_manager.runtime_config.get(
            "connection", {}
        )

        LOGGER.debug(f"federation mode={mode}")

        return Federation(
            federation_session_id,
            party,
            mq,
            pulsar_manager,
            max_message_size,
            topic_ttl,
            cluster,
            tenant,
            conf,
            mode
        )

    def __init__(self, session_id, party: Party, mq: MQ, pulsar_manager: PulsarManager, max_message_size, topic_ttl,
                 cluster, tenant, conf, mode):
        super().__init__(session_id=session_id, party=party, mq=mq, max_message_size=max_message_size, conf=conf)

        self._pulsar_manager = pulsar_manager
        self._topic_ttl = topic_ttl
        self._cluster = cluster
        self._tenant = tenant
        self._mode = mode

    def __getstate__(self):
        pass

    def destroy(self, parties):
        # The idea cleanup strategy is to consume all message in topics,
        # and let pulsar cluster to collect the used topics.

        LOGGER.debug("[pulsar.cleanup]start to cleanup...")

        # 1. remove subscription
        response = self._pulsar_manager.unsubscribe_namespace_all_topics(
            tenant=self._tenant,
            namespace=self._session_id,
            subscription_name=DEFAULT_SUBSCRIPTION_NAME,
        )
        if response.ok:
            LOGGER.debug("successfully unsubscribe all topics")
        else:
            LOGGER.error(response.text)

        # 2. reset retention policy
        response = self._pulsar_manager.set_retention(
            self._tenant,
            self._session_id,
            retention_time_in_minutes=0,
            retention_size_in_MB=0,
        )
        if response.ok:
            LOGGER.debug("successfully reset all retention policy")
        else:
            LOGGER.error(response.text)

        # 3. remove cluster from namespace
        response = self._pulsar_manager.set_clusters_to_namespace(
            self._tenant, self._session_id, [self._cluster]
        )
        if response.ok:
            LOGGER.debug("successfully reset all replicated cluster")
        else:
            LOGGER.error(response.text)

        # # 4. remove namespace
        # response = self._pulsar_manager.delete_namespace(
        #     self._tenant, self._session_id
        # )
        # if response.ok:
        #     LOGGER.debug(f"successfully delete namespace={self._session_id}")
        # else:
        #     LOGGER.error(response.text)

    def _maybe_create_topic_and_replication(self, party, topic_suffix):
        if self._mode == "replication":
            return self._create_topic_by_replication_mode(party, topic_suffix)

        if self._mode == "client":
            return self._create_topic_by_client_mode(party, topic_suffix)

        raise ValueError("mode={self._mode} is not support!")

    def _create_topic_by_client_mode(self, party, topic_suffix):
        send_topic_name = f"{self._party.role}-{self._party.party_id}-{party.role}-{party.party_id}-{topic_suffix}"
        receive_topic_name = f"{party.role}-{party.party_id}-{self._party.role}-{self._party.party_id}-{topic_suffix}"

        # topic_pair is a pair of topic for sending and receiving message respectively
        topic_pair = _TopicPair(
            tenant=self._tenant,
            namespace=self._session_id,
            send=send_topic_name,
            receive=receive_topic_name,
        )

        # init pulsar namespace
        namespaces = self._pulsar_manager.get_namespace(
            self._tenant).json()
        # create namespace
        if f"{self._tenant}/{self._session_id}" not in namespaces:
            # append target cluster to the pulsar namespace
            code = self._pulsar_manager.create_namespace(
                self._tenant, self._session_id
            ).status_code

            # according to https://pulsar.apache.org/admin-rest-api/?version=2.7.0&apiversion=v2#operation/getPolicies
            # return 409 if existed
            # return 204 if ok
            if code == 204 or code == 409:
                LOGGER.debug(
                    "successfully create pulsar namespace: %s", self._session_id
                )
            else:
                raise Exception(
                    "unable to create pulsar namespace with status code: {}".format(
                        code
                    )
                )

            # set message ttl for the namespace
            response = self._pulsar_manager.set_retention(
                self._tenant,
                self._session_id,
                retention_time_in_minutes=int(self._topic_ttl),
                retention_size_in_MB=-1,
            )

            LOGGER.debug(response.text)
            if response.ok:
                LOGGER.debug(
                    "successfully set message ttl to namespace: {} about {} mintues".format(
                        self._session_id, self._topic_ttl
                    )
                )
            else:
                LOGGER.debug("failed to set message ttl to namespace")

        return topic_pair

    def _create_topic_by_replication_mode(self, party, topic_suffix):
        send_topic_name = f"{self._party.role}-{self._party.party_id}-{party.role}-{party.party_id}-{topic_suffix}"
        receive_topic_name = f"{party.role}-{party.party_id}-{self._party.role}-{self._party.party_id}-{topic_suffix}"

        # topic_pair is a pair of topic for sending and receiving message respectively
        topic_pair = _TopicPair(
            tenant=self._tenant,
            namespace=self._session_id,
            send=send_topic_name,
            receive=receive_topic_name,
        )

        if party.party_id == self._party.party_id:
            LOGGER.debug(
                "connecting to local broker, skipping cluster creation"
            )
        else:
            # init pulsar cluster
            cluster = self._pulsar_manager.get_cluster(
                party.party_id).json()
            if (
                    cluster.get("brokerServiceUrl", "") == ""
                    and cluster.get("brokerServiceUrlTls", "") == ""
            ):
                LOGGER.debug(
                    "pulsar cluster with name %s does not exist or broker url is empty, creating...",
                    party.party_id,
                )

                remote_party = self._mq.route_table.get(
                    int(party.party_id), None
                )

                # handle party does not exist in route table first
                if remote_party is None:
                    domain = self._mq.route_table.get(
                        "default").get("domain")
                    host = f"{party.party_id}.{domain}"
                    port = self._mq.route_table.get("default").get(
                        "brokerPort", "6650"
                    )
                    sslPort = self._mq.route_table.get("default").get(
                        "brokerSslPort", "6651"
                    )
                    proxy = self._mq.route_table.get(
                        "default").get("proxy", "")
                # fetch party info from the route table
                else:
                    host = self._mq.route_table.get(int(party.party_id)).get(
                        "host"
                    )
                    port = self._mq.route_table.get(int(party.party_id)).get(
                        "port", "6650"
                    )
                    sslPort = self._mq.route_table.get(int(party.party_id)).get(
                        "sslPort", "6651"
                    )
                    proxy = self._mq.route_table.get(int(party.party_id)).get(
                        "proxy", ""
                    )

                broker_url = f"pulsar://{host}:{port}"
                broker_url_tls = f"pulsar+ssl://{host}:{sslPort}"
                if proxy != "":
                    proxy = f"pulsar+ssl://{proxy}"

                if self._pulsar_manager.create_cluster(
                        cluster_name=party.party_id,
                        broker_url=broker_url,
                        broker_url_tls=broker_url_tls,
                        proxy_url=proxy,
                ).ok:
                    LOGGER.debug(
                        "pulsar cluster with name: %s, broker_url: %s created",
                        party.party_id,
                        broker_url,
                    )
                elif self._pulsar_manager.update_cluster(
                        cluster_name=party.party_id,
                        broker_url=broker_url,
                        broker_url_tls=broker_url_tls,
                        proxy_url=proxy,
                ).ok:
                    LOGGER.debug(
                        "pulsar cluster with name: %s, broker_url: %s updated",
                        party.party_id,
                        broker_url,
                    )
                else:
                    error_message = (
                        "unable to create pulsar cluster: %s".format(
                            party.party_id
                        )
                    )
                    LOGGER.error(error_message)
                    # just leave this alone.
                    raise Exception(error_message)

            # update tenant
            tenant_info = self._pulsar_manager.get_tenant(
                self._tenant).json()
            if party.party_id not in tenant_info["allowedClusters"]:
                tenant_info["allowedClusters"].append(party.party_id)
                if self._pulsar_manager.update_tenant(
                        self._tenant,
                        tenant_info.get("admins", []),
                        tenant_info.get(
                            "allowedClusters",
                        ),
                ).ok:
                    LOGGER.debug(
                        "successfully update tenant with cluster: %s",
                        party.party_id,
                    )
                else:
                    raise Exception("unable to update tenant")

        # TODO: remove this for the loop
        # init pulsar namespace
        namespaces = self._pulsar_manager.get_namespace(
            self._tenant).json()
        # create namespace
        if f"{self._tenant}/{self._session_id}" not in namespaces:
            # append target cluster to the pulsar namespace
            clusters = [self._cluster]
            if (
                    party.party_id != self._party.party_id
                    and party.party_id not in clusters
            ):
                clusters.append(party.party_id)

            policy = {"replication_clusters": clusters}

            code = self._pulsar_manager.create_namespace(
                self._tenant, self._session_id, policies=policy
            ).status_code
            # according to https://pulsar.apache.org/admin-rest-api/?version=2.7.0&apiversion=v2#operation/getPolicies
            # return 409 if existed
            # return 204 if ok
            if code == 204 or code == 409:
                LOGGER.debug(
                    "successfully create pulsar namespace: %s", self._session_id
                )
            else:
                raise Exception(
                    "unable to create pulsar namespace with status code: {}".format(
                        code
                    )
                )

            # set message ttl for the namespace
            response = self._pulsar_manager.set_retention(
                self._tenant,
                self._session_id,
                retention_time_in_minutes=int(self._topic_ttl),
                retention_size_in_MB=-1,
            )

            LOGGER.debug(response.text)
            if response.ok:
                LOGGER.debug(
                    "successfully set message ttl to namespace: {} about {} mintues".format(
                        self._session_id, self._topic_ttl
                    )
                )
            else:
                LOGGER.debug("failed to set message ttl to namespace")
        # update party to namespace
        else:
            if party.party_id != self._party.party_id:
                clusters = self._pulsar_manager.get_cluster_from_namespace(
                    self._tenant, self._session_id
                ).json()
                if party.party_id not in clusters:
                    clusters.append(party.party_id)
                    if self._pulsar_manager.set_clusters_to_namespace(
                            self._tenant, self._session_id, clusters
                    ).ok:
                        LOGGER.debug(
                            "successfully set clusters: {}  to pulsar namespace: {}".format(
                                clusters, self._session_id
                            )
                        )
                    else:
                        raise Exception(
                            "unable to update clusters: {} to pulsar namespaces: {}".format(
                                clusters, self._session_id
                            )
                        )

        return topic_pair

    def _get_channel(self, topic_pair: _TopicPair, src_party_id, src_role, dst_party_id, dst_role, mq=None,
                     conf: dict = None):
        return MQChannel(
            host=mq.host,
            port=mq.port,
            tenant=topic_pair.tenant,
            namespace=topic_pair.namespace,
            send_topic=topic_pair.send,
            receive_topic=topic_pair.receive,
            src_party_id=src_party_id,
            src_role=src_role,
            dst_party_id=dst_party_id,
            dst_role=dst_role,
            credential=None,
            extra_args=conf,
        )

    def _get_consume_message(self, channel_info):
        while True:
            message = channel_info.consume()
            body = message.data()
            properties = message.properties()
            message_id = message.message_id()
            yield message_id, properties, body

    def _consume_ack(self, channel_info, id):
        channel_info.ack(message=id)
