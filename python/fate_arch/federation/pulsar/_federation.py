########################################################
# Copyright 2019-2021 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

import io
import json
import sys
import time
import typing

from pickle import dumps as p_dumps, loads as p_loads

# noinspection PyPackageRequirements
from pyspark import SparkContext

from fate_arch.common import file_utils
from fate_arch.abc import FederationABC, GarbageCollectionABC
from fate_arch.common import Party
from fate_arch.common.log import getLogger
from fate_arch.computing.spark import Table
from fate_arch.computing.spark._materialize import materialize
from fate_arch.federation.pulsar._mq_channel import (
    MQChannel,
    DEFAULT_TENANT,
    DEFAULT_CLUSTER,
    DEFAULT_SUBSCRIPTION_NAME,
)
from fate_arch.federation.pulsar._pulsar_manager import PulsarManager
from fate_arch.federation._datastream import Datastream


LOGGER = getLogger()
# default message max size in bytes = 50MB
DEFAULT_MESSAGE_MAX_SIZE = 104857 * 50
NAME_DTYPE_TAG = "<dtype>"
_SPLIT_ = "^"


class FederationDataType(object):
    OBJECT = "obj"
    TABLE = "Table"


# to create pulsar client


class MQ(object):
    def __init__(self, host, port, mng_port, route_table):
        self.host = host
        self.port = port
        self.mng_port = mng_port
        self.route_table = route_table

    def __str__(self):
        return (
            f"MQ(host={self.host}, port={self.port}, union_name={self.union_name}, "
            f"policy_id={self.policy_id}, route_table={self.route_table}), "
            f"type=pulsar"
        )

    def __repr__(self):
        return self.__str__()


# to locate pulsar topic


class _TopicPair(object):
    def __init__(self, tenant, namespace, send, receive):
        self.tenant = tenant
        self.namespace = namespace
        self.send = send
        self.receive = receive


class Federation(FederationABC):
    @staticmethod
    def from_conf(
        federation_session_id: str,
        party: Party,
        runtime_conf: dict,
        pulsar_config: dict,
    ):
        LOGGER.debug(f"pulsar_config: {pulsar_config}")
        host = pulsar_config.get("host", "localhost")
        port = pulsar_config.get("port", "6650")
        mng_port = pulsar_config.get("mng_port", "8080")
        topic_ttl = int(pulsar_config.get("topic_ttl", 0))
        cluster = pulsar_config.get("cluster", DEFAULT_CLUSTER)
        # tenant name should be unified between parties
        tenant = pulsar_config.get("tenant", DEFAULT_TENANT)

        # pulsaar runtime config
        pulsar_run = runtime_conf.get(
            "job_parameters", {}).get("pulsar_run", {})
        LOGGER.debug(f"pulsar_run: {pulsar_run}")

        max_message_size = pulsar_run.get(
            "max_message_size", DEFAULT_MESSAGE_MAX_SIZE)
        LOGGER.debug(f"set max message size to {max_message_size} Bytes")

        # topic ttl could be overwritten by run time config
        topic_ttl = int(pulsar_run.get("topic_ttl", topic_ttl))

        # pulsar not use user and password so far
        # TODO add credential to connections
        base_user = pulsar_config.get("user")
        base_password = pulsar_config.get("password")

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
        mq = MQ(host, port, mng_port, route_table)
        return Federation(
            federation_session_id,
            party,
            mq,
            pulsar_manager,
            max_message_size,
            topic_ttl,
            cluster,
            tenant,
        )

    def __init__(
        self,
        session_id,
        party: Party,
        mq: MQ,
        pulsar_manager: PulsarManager,
        max_message_size,
        topic_ttl,
        cluster,
        tenant,
    ):
        self._session_id = session_id
        self._party = party
        self._mq = mq
        self._pulsar_manager = pulsar_manager

        self._topic_map: typing.MutableMapping[_TopicKey, _TopicPair] = {}
        self._channels_map: typing.MutableMapping[_TopicKey, MQChannel] = {}

        self._name_dtype_map = {}
        self._message_cache = {}
        self._max_message_size = max_message_size
        self._topic_ttl = topic_ttl
        self._cluster = cluster
        self._tenant = tenant

    def __getstate__(self):
        pass

    def get(
        self, name: str, tag: str, parties: typing.List[Party], gc: GarbageCollectionABC
    ) -> typing.List:
        log_str = f"[pulsar.get](name={name}, tag={tag}, parties={parties})"
        LOGGER.debug(f"[{log_str}]start to get")

        _name_dtype_keys = [
            _SPLIT_.join([party.role, party.party_id, name, tag, "get"])
            for party in parties
        ]

        if _name_dtype_keys[0] not in self._name_dtype_map:
            party_topic_infos = self._get_party_topic_infos(
                parties, dtype=NAME_DTYPE_TAG
            )
            channel_infos = self._get_channels(
                party_topic_infos=party_topic_infos)
            rtn_dtype = []
            for i, info in enumerate(channel_infos):
                obj = self._receive_obj(
                    info, name, tag=_SPLIT_.join([tag, NAME_DTYPE_TAG])
                )
                rtn_dtype.append(obj)
                LOGGER.debug(
                    f"[pulsar.get] _name_dtype_keys: {_name_dtype_keys}, dtype: {obj}"
                )

            for k in _name_dtype_keys:
                if k not in self._name_dtype_map:
                    self._name_dtype_map[k] = rtn_dtype[0]

        rtn_dtype = self._name_dtype_map[_name_dtype_keys[0]]

        rtn = []
        dtype = rtn_dtype.get("dtype", None)
        partitions = rtn_dtype.get("partitions", None)

        if dtype == FederationDataType.TABLE:
            party_topic_infos = self._get_party_topic_infos(
                parties, name, partitions=partitions
            )
            for i in range(len(party_topic_infos)):
                party = parties[i]
                role = party.role
                party_id = party.party_id
                topic_infos = party_topic_infos[i]
                receive_func = self._get_partition_receive_func(
                    name,
                    tag,
                    party_id,
                    role,
                    topic_infos,
                    mq=self._mq,
                    conf=self._pulsar_manager.runtime_config,
                )

                sc = SparkContext.getOrCreate()
                rdd = sc.parallelize(range(partitions), partitions)
                rdd = rdd.mapPartitionsWithIndex(receive_func)
                rdd = materialize(rdd)
                table = Table(rdd)
                rtn.append(table)

                # add gc
                # 1. spark
                gc.add_gc_action(tag, table, "__del__", {})
                LOGGER.debug(
                    f"[{log_str}]received rdd({i + 1}/{len(parties)}), party: {parties[i]}"
                )
        else:
            party_topic_infos = self._get_party_topic_infos(parties, name)
            channel_infos = self._get_channels(
                party_topic_infos=party_topic_infos)
            for i, info in enumerate(channel_infos):
                obj = self._receive_obj(info, name, tag)
                LOGGER.debug(
                    f"[{log_str}]received obj({i + 1}/{len(parties)}), party: {parties[i]} "
                )
                rtn.append(obj)

        LOGGER.debug(f"[{log_str}]finish to get")
        return rtn

    def remote(
        self,
        v,
        name: str,
        tag: str,
        parties: typing.List[Party],
        gc: GarbageCollectionABC,
    ) -> typing.NoReturn:
        log_str = f"[pulsar.remote](name={name}, tag={tag}, parties={parties})"

        _name_dtype_keys = [
            _SPLIT_.join([party.role, party.party_id, name, tag, "remote"])
            for party in parties
        ]

        # tell the receiver what sender is going to send.

        if _name_dtype_keys[0] not in self._name_dtype_map:
            party_topic_infos = self._get_party_topic_infos(
                parties, dtype=NAME_DTYPE_TAG
            )
            channel_infos = self._get_channels(
                party_topic_infos=party_topic_infos)
            if isinstance(v, Table):
                body = {"dtype": FederationDataType.TABLE,
                        "partitions": v.partitions}
            else:
                body = {"dtype": FederationDataType.OBJECT}

            LOGGER.debug(
                f"[pulsar.remote] _name_dtype_keys: {_name_dtype_keys}, dtype: {body}"
            )
            self._send_obj(
                name=name,
                tag=_SPLIT_.join([tag, NAME_DTYPE_TAG]),
                data=p_dumps(body),
                channel_infos=channel_infos,
            )

            for k in _name_dtype_keys:
                if k not in self._name_dtype_map:
                    self._name_dtype_map[k] = body

        if isinstance(v, Table):
            total_size = v.count()
            partitions = v.partitions
            LOGGER.debug(
                f"[{log_str}]start to remote RDD, total_size={total_size}, partitions={partitions}"
            )

            party_topic_infos = self._get_party_topic_infos(
                parties, name, partitions=partitions
            )
            # add gc
            gc.add_gc_action(tag, v, "__del__", {})

            send_func = self._get_partition_send_func(
                name,
                tag,
                partitions,
                party_topic_infos,
                mq=self._mq,
                maximun_message_size=self._max_message_size,
                conf=self._pulsar_manager.runtime_config,
            )
            # noinspection PyProtectedMember
            v._rdd.mapPartitionsWithIndex(send_func).count()
        else:
            LOGGER.debug(f"[{log_str}]start to remote obj")
            party_topic_infos = self._get_party_topic_infos(parties, name)
            channel_infos = self._get_channels(
                party_topic_infos=party_topic_infos)
            self._send_obj(
                name=name, tag=tag, data=p_dumps(v), channel_infos=channel_infos
            )

        LOGGER.debug(f"[{log_str}]finish to remote")

    def cleanup(self, parties):
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

        # 4. clear all backlog ?

    def _get_party_topic_infos(
        self, parties: typing.List[Party], name=None, partitions=None, dtype=None
    ) -> typing.List:
        topic_infos = [
            self._get_or_create_topic(party, name, partitions, dtype)
            for party in parties
        ]

        # the return is formed like this: [[(topic_key1, topic_info1),
        # (topic_key2, topic_info2)...],[(topic_key1, topic_info1), (topic_key2,
        # topic_info2]...]
        return topic_infos

    def _get_or_create_topic(
        self, party: Party, name=None, partitions=None, dtype=None, client_type=None
    ) -> typing.Tuple:
        topic_key_list = []
        topic_infos = []

        if dtype is not None:
            topic_key = _SPLIT_.join(
                [party.role, party.party_id, dtype, dtype])
            topic_key_list.append(topic_key)
        else:
            if partitions is not None:
                for i in range(partitions):
                    topic_key = _SPLIT_.join(
                        [party.role, party.party_id, name, str(i)])
                    topic_key_list.append(topic_key)
            elif name is not None:
                topic_key = _SPLIT_.join([party.role, party.party_id, name])
                topic_key_list.append(topic_key)
            else:
                topic_key = _SPLIT_.join([party.role, party.party_id])
                topic_key_list.append(topic_key)

        for topic_key in topic_key_list:
            if topic_key not in self._topic_map:
                LOGGER.debug(
                    f"[pulsar.get_or_create_topic]topic: {topic_key} for party:{party} not found, start to create"
                )
                # gen names

                topic_key_splits = topic_key.split(_SPLIT_)
                queue_suffix = "-".join(topic_key_splits[2:])
                send_topic_name = f"{self._party.role}-{self._party.party_id}-{party.role}-{party.party_id}-{queue_suffix}"
                receive_topic_name = f"{party.role}-{party.party_id}-{self._party.role}-{self._party.party_id}-{queue_suffix}"

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

                self._topic_map[topic_key] = topic_pair
                # TODO: check federated queue status
                LOGGER.debug(
                    f"[pulsar.get_or_create_topic]topic for topic_key: {topic_key}, party:{party} created"
                )

            topic_pair = self._topic_map[topic_key]
            topic_infos.append((topic_key, topic_pair))

        return topic_infos

    def _get_channel(self, mq, topic_pair: _TopicPair, party_id, role, conf: dict):
        return MQChannel(
            host=mq.host,
            port=mq.port,
            mng_port=mq.mng_port,
            pulsar_tenant=topic_pair.tenant,
            pulsar_namespace=topic_pair.namespace,
            pulsar_send_topic=topic_pair.send,
            pulsar_receive_topic=topic_pair.receive,
            party_id=party_id,
            role=role,
            credential=None,
            extra_args=conf,
        )

    def _get_channels(self, party_topic_infos):
        channel_infos = []
        for e in party_topic_infos:
            for topic_key, topic_pair in e:
                topic_key_splits = topic_key.split(_SPLIT_)
                role = topic_key_splits[0]
                party_id = topic_key_splits[1]
                info = self._channels_map.get(topic_key)
                if info is None:
                    info = self._get_channel(
                        self._mq,
                        topic_pair,
                        party_id=party_id,
                        role=role,
                        conf=self._pulsar_manager.runtime_config,
                    )
                    self._channels_map[topic_key] = info
                channel_infos.append(info)
        return channel_infos

    # can't pickle _thread.lock objects
    def _get_channels_index(self, index, party_topic_infos, mq, conf: dict):
        channel_infos = []
        for e in party_topic_infos:
            # select specified topic_info for a party
            topic_key, topic_pair = e[index]
            topic_key_splits = topic_key.split(_SPLIT_)
            role = topic_key_splits[0]
            party_id = topic_key_splits[1]
            info = self._get_channel(
                mq, topic_pair, party_id=party_id, role=role, conf=conf
            )
            channel_infos.append(info)
        return channel_infos

    def _send_obj(self, name, tag, data, channel_infos):
        for info in channel_infos:
            # selfmade properties
            properties = {
                "content_type": "text/plain",
                "app_id": info.party_id,
                "message_id": name,
                "correlation_id": tag,
            }
            LOGGER.debug(f"[pulsar._send_obj]properties:{properties}.")
            info.basic_publish(body=data, properties=properties)

    def _get_message_cache_key(self, name, tag, party_id, role):
        cache_key = _SPLIT_.join([name, tag, str(party_id), role])
        return cache_key

    def _receive_obj(self, channel_info, name, tag):
        party_id = channel_info._party_id
        role = channel_info._role
        wish_cache_key = self._get_message_cache_key(name, tag, party_id, role)

        if wish_cache_key in self._message_cache:
            return self._message_cache[wish_cache_key]

        while True:
            message = channel_info.consume()
            # return None indicates the client is closed
            body = message.data()
            properties = message.properties()
            LOGGER.debug(f"[pulsar._receive_obj] properties: {properties}.")

            if properties["message_id"] != name or properties["correlation_id"] != tag:
                LOGGER.warning(
                    f"[pulsar._receive_obj] require {name}.{tag}, got {properties['message_id']}.{properties['correlation_id']}"
                )
                # just ack and continue
                # channel_info.basic_ack(message.message_id())
                # continue

            cache_key = self._get_message_cache_key(
                properties["message_id"], properties["correlation_id"], party_id, role
            )
            # object
            if properties["content_type"] == "text/plain":
                self._message_cache[cache_key] = p_loads(body)
                # TODO: handle ack failure
                channel_info.basic_ack(message.message_id())
                if cache_key == wish_cache_key:
                    # keep connection open for receiving object
                    # channel_info.cancel()
                    LOGGER.debug(
                        f"[pulsar._receive_obj] cache_key: {cache_key}, obj: {self._message_cache[cache_key]}"
                    )
                    return self._message_cache[cache_key]
            else:
                raise ValueError(
                    f"[pulsar._receive_obj] properties.content_type is {properties.content_type}, but must be text/plain"
                )

    def _send_kv(
        self, name, tag, data, channel_infos, partition_size, partitions, message_key
    ):
        headers = json.dumps(
            {
                "partition_size": partition_size,
                "partitions": partitions,
                "message_key": message_key,
            }
        )

        for info in channel_infos:
            properties = {
                "content_type": "application/json",
                "app_id": info.party_id,
                "message_id": name,
                "correlation_id": tag,
                "headers": headers,
            }
            LOGGER.debug(
                f"[pulsar._send_kv]info: {info}, properties: {properties}.")
            info.basic_publish(body=data, properties=properties)

    def _get_partition_send_func(
        self,
        name,
        tag,
        partitions,
        party_topic_infos,
        mq,
        maximun_message_size,
        conf: dict,
    ):
        def _fn(index, kvs):
            return self._partition_send(
                index,
                kvs,
                name,
                tag,
                partitions,
                party_topic_infos,
                mq,
                maximun_message_size,
                conf,
            )

        return _fn

    def _partition_send(
        self,
        index,
        kvs,
        name,
        tag,
        partitions,
        party_topic_infos,
        mq,
        maximun_message_size,
        conf: dict,
    ):
        channel_infos = self._get_channels_index(
            index=index, party_topic_infos=party_topic_infos, mq=mq, conf=conf
        )
        # reuse datastream here incase message size has limitation in pulsar
        datastream = Datastream()
        base_message_key = str(index)
        message_key_idx = 0
        count = 0
        internal_count = 0
        for k, v in kvs:
            count += 1
            internal_count += 1
            el = {"k": p_dumps(k).hex(), "v": p_dumps(v).hex()}
            # roughly caculate the size of package to avoid serialization ;)
            if (
                datastream.get_size() +
                sys.getsizeof(el["k"]) + sys.getsizeof(el["v"])
                >= maximun_message_size
            ):
                LOGGER.debug(
                    f"[pulsar._partition_send]The count of message is: {internal_count}"
                )
                LOGGER.debug(
                    f"[pulsar._partition_send]The total count of message is: {count}"
                )
                internal_count = 0
                message_key_idx += 1
                message_key = _SPLIT_.join(
                    [base_message_key, str(message_key_idx)])
                self._send_kv(
                    name=name,
                    tag=tag,
                    data=datastream.get_data().encode(),
                    channel_infos=channel_infos,
                    partition_size=-1,
                    partitions=partitions,
                    message_key=message_key,
                )
                datastream.clear()
            datastream.append(el)

        message_key_idx += 1
        message_key = _SPLIT_.join([base_message_key, str(message_key_idx)])

        self._send_kv(
            name=name,
            tag=tag,
            data=datastream.get_data().encode(),
            channel_infos=channel_infos,
            partition_size=count,
            partitions=partitions,
            message_key=message_key,
        )

        return [1]

    def _get_partition_receive_func(
        self, name, tag, party_id, role, topic_infos, mq, conf: dict
    ):
        def _fn(index, kvs):
            return self._partition_receive(
                index, kvs, name, tag, party_id, role, topic_infos, mq, conf
            )

        return _fn

    def _partition_receive(
        self, index, kvs, name, tag, party_id, role, topic_infos, mq, conf: dict
    ):
        topic_pair = topic_infos[index][1]
        channel_info = self._get_channel(mq, topic_pair, party_id, role, conf)

        message_key_cache = set()
        count = 0
        partition_size = -1
        all_data = []
        while True:
            try:
                message = channel_info.consume()
                properties = message.properties()
                # must get bytes
                body = message.data().decode()
                LOGGER.debug(
                    f"[pulsar._partition_receive] properties: {properties}.")
                if (
                    properties["message_id"] != name
                    or properties["correlation_id"] != tag
                ):
                    # leave this code to handle unexpected situation
                    channel_info.basic_ack(message.message_id())
                    LOGGER.debug(
                        f"[pulsar._partition_receive]: require {name}.{tag}, got {properties['message_id']}.{properties['correlation_id']}"
                    )
                    continue

                if properties["content_type"] == "application/json":
                    # headers here is json bytes string
                    header = json.loads(properties["headers"])
                    message_key = header.get("message_key")
                    if message_key in message_key_cache:
                        LOGGER.debug(
                            f"[pulsar._partition_receive] message_key : {message_key} is duplicated"
                        )
                        channel_info.basic_ack(message.message_id())
                        continue

                    message_key_cache.add(message_key)

                    if header.get("partition_size") >= 0:
                        partition_size = header.get("partition_size")

                    data = json.loads(body)
                    data_iter = (
                        (
                            p_loads(bytes.fromhex(el["k"])),
                            p_loads(bytes.fromhex(el["v"])),
                        )
                        for el in data
                    )
                    count += len(data)
                    LOGGER.debug(
                        f"[pulsar._partition_receive] count: {len(data)}")
                    LOGGER.debug(
                        f"[pulsar._partition_receive]total count: {count}")
                    all_data.extend(data_iter)
                    channel_info.basic_ack(message.message_id())
                    if partition_size != -1:
                        if count == partition_size:
                            channel_info.cancel()
                            return all_data
                        else:
                            raise Exception(
                                f"[pulsar._partition_receive] want {partition_size} data in {name}.{tag} but got {count}"
                            )
                else:
                    raise ValueError(
                        f"[pulsar._partition_receive]properties.content_type is {properties.content_type}, but must be application/json"
                    )
            except Exception as e:
                LOGGER.error(
                    f"[pulsar._partition_receive]catch exception {e}, while receiving {name}.{tag}"
                )
                # avoid hang on consume()
                if count == partition_size:
                    channel_info.cancel()
                    return all_data
                else:
                    raise e
