########################################################
# Copyright 2019-2021 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

import logging
import json
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fate_arch.common.log import getLogger
from fate_arch.federation.pulsar._mq_channel import DEFAULT_SUBSCRIPTION_NAME

logger = getLogger()

MAX_RETRIES = 10
MAX_REDIRECT = 5
BACKOFF_FACTOR = 1

# sleep time equips to {BACKOFF_FACTOR} * (2 ** ({NUMBER_OF_TOTALRETRIES} - 1))

CLUSTER = 'clusters/{}'
TENANT = 'tenants/{}'

# APIs are refer to https://pulsar.apache.org/admin-rest-api/?version=2.7.0&apiversion=v2


class PulsarManager():
    def __init__(self, host: str, port: str, runtime_config: dict = {}):
        self.service_url = "http://{}:{}/admin/v2/".format(host, port)
        self.runtime_config = runtime_config

    # create session is used to construct url and request parameters
    def _create_session(self):
        # retry mechanism refers to https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#urllib3.util.Retry
        retry = Retry(total=MAX_RETRIES, redirect=MAX_REDIRECT,
                      backoff_factor=BACKOFF_FACTOR)
        s = requests.Session()
        # initialize headers
        s.headers.update({'Content-Type': 'application/json'})

        http_adapter = HTTPAdapter(max_retries=retry)
        s.mount('http://', http_adapter)
        s.mount('https://', http_adapter)
        return s

    # allocator
    def get_allocator(self, allocator: str = 'default'):
        session = self._create_session()
        response = session.get(
            self.service_url + 'broker-stats/allocator-stats/{}'.format(allocator))
        return response

    # cluster
    def get_cluster(self, cluster_name: str = ''):
        session = self._create_session()
        response = session.get(
            self.service_url + CLUSTER.format(cluster_name))
        return response

    def delete_cluster(self, cluster_name: str = ''):
        session = self._create_session()

        response = session.delete(
            self.service_url + CLUSTER.format(cluster_name))
        return response

    # service_url need to provide "http://" prefix
    def create_cluster(self, cluster_name: str,  broker_url: str, service_url: str = '',
                       service_url_tls: str = '', broker_url_tls: str = '',
                       proxy_url: str = '', proxy_protocol: str = "SNI", peer_cluster_names: list = [],
                       ):
        # initialize data
        data = {
            'serviceUrl': service_url,
            'serviceUrlTls': service_url_tls,
            'brokerServiceUrl': broker_url,
            'brokerServiceUrlTls': broker_url_tls,
            'peerClusterNames': peer_cluster_names,
            'proxyServiceUrl': proxy_url,
            'proxyProtocol': proxy_protocol
        }

        session = self._create_session()

        response = session.put(
            self.service_url + CLUSTER.format(cluster_name), data=json.dumps(data))
        return response

    def update_cluster(self, cluster_name: str,  broker_url: str, service_url: str = '',
                       service_url_tls: str = '', broker_url_tls: str = '',
                       proxy_url: str = '', proxy_protocol: str = "SNI", peer_cluster_names: list = [],
                       ):
        # initialize data
        data = {
            'serviceUrl': service_url,
            'serviceUrlTls': service_url_tls,
            'brokerServiceUrl': broker_url,
            'brokerServiceUrlTls': broker_url_tls,
            'peerClusterNames': peer_cluster_names,
            'proxyServiceUrl': proxy_url,
            'proxyProtocol': proxy_protocol
        }

        session = self._create_session()

        response = session.post(
            self.service_url + CLUSTER.format(cluster_name), data=json.dumps(data))
        return response

    # tenants
    def get_tenant(self, tenant: str = ''):
        session = self._create_session()
        response = session.get(self.service_url + TENANT.format(tenant))
        return response

    def create_tenant(self, tenant: str, admins: list, clusters: list):
        session = self._create_session()

        data = {'adminRoles': admins,
                'allowedClusters': clusters}

        response = session.put(
            self.service_url + TENANT.format(tenant), data=json.dumps(data))

        return response

    def delete_tenant(self, tenant: str):
        session = self._create_session()
        response = session.delete(
            self.service_url + TENANT.format(tenant))
        return response

    def update_tenant(self, tenant: str, admins: list, clusters: list):
        session = self._create_session()

        data = {'adminRoles': admins,
                'allowedClusters': clusters}

        response = session.post(
            self.service_url + TENANT.format(tenant), data=json.dumps(data))
        return response

    # namespace

    def get_namespace(self, tenant: str):
        session = self._create_session()
        response = session.get(
            self.service_url + 'namespaces/{}'.format(tenant))
        return response

     # 'replication_clusters' is always required
    def create_namespace(self, tenant: str, namespace: str, policies: dict = {}):
        session = self._create_session()
        response = session.put(
            self.service_url + 'namespaces/{}/{}'.format(tenant, namespace),
            data=json.dumps(policies)
        )
        return response

    def delete_namespace(self, tenant: str, namespace: str, force: bool = False):
        session = self._create_session()
        response = session.delete(
            self.service_url +
            'namespace/{}/{}?force={}'.format(tenant,
                                              namespace, str(force).lower())
        )
        return response

    def set_clusters_to_namespace(self, tenant: str, namespace: str, clusters: list):
        session = self._create_session()
        response = session.post(
            self.service_url + 'namespaces/{}/{}/replication'.format(tenant, namespace), json=clusters
        )

        return response

    def get_cluster_from_namespace(self, tenant: str, namespace: str):
        session = self._create_session()
        response = session.get(
            self.service_url +
            'namespaces/{}/{}/replication'.format(tenant, namespace)
        )

        return response

    def set_subscription_expiration_time(self, tenant: str, namespace: str, mintues: int = 0):
        session = self._create_session()
        response = session.post(
            self.service_url + 'namespaces/{}/{}/subscriptionExpirationTime'.format(tenant, namespace), json=mintues
        )

        return response

    def set_message_ttl(self, tenant: str, namespace: str, mintues: int = 0):
        session = self._create_session()
        response = session.post(
            # the API accepts data as seconds
            self.service_url + 'namespaces/{}/{}/messageTTL'.format(tenant, namespace), json=mintues*60
        )

        return response

    def unsubscribe_namespace_all_topics(self, tenant: str, namespace: str, subscription_name: str):
        session = self._create_session()
        response = session.post(
            self.service_url +
            'namespaces/{}/{}/unsubscribe/{}'.format(
                tenant, namespace, subscription_name)
        )
        return response

    def set_retention(self, tenant: str, namespace: str,
                      retention_time_in_minutes: int = 0, retention_size_in_MB: int = 0):
        session = self._create_session()

        data = {'retentionTimeInMinutes': retention_time_in_minutes,
                'retentionSizeInMB': retention_size_in_MB}

        response = session.post(
            self.service_url +
            'namespaces/{}/{}/retention'.format(tenant, namespace), data=json.dumps(data)
        )
        return response

    def remove_retention(self, tenant: str, namespace: str):
        session = self._create_session()
        response = session.delete(
            self.service_url +
            'namespaces/{}/{}/retention'.format(tenant, namespace),
        )

        return response

    # topic
    def unsubscribe_topic(self, tenant: str, namespace: str, topic: str, subscription_name: str):
        session = self._create_session()
        response = session.delete(
            self.service_url +
            'persistent/{}/{}/{}/subscription/{}'.format(
                tenant, namespace, topic, subscription_name)
        )
        return response
