########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

import requests
import time

from fate_arch.common import log

LOGGER = log.getLogger()

C_HTTP_TEMPLATE = "http://{}/api/{}"
C_COMMON_HTTP_HEADER = {'Content-Type': 'application/json'}

"""
APIs are refered to https://rawcdn.githack.com/rabbitmq/rabbitmq-management/v3.8.3/priv/www/api/index.html
"""


def connection_retry(func):
    """retry connection
    """

    def wrapper(self, *args, **kwargs):
        """wrapper
        """
        res = False
        for ntry in range(60):
            try:
                res = func(self, *args, **kwargs)
                if res is True:
                    break
            except Exception as e:
                LOGGER.error("[rabbitmanager]function %s error" % func.__name__, exc_info=True)
                time.sleep(1)
        return res
    return wrapper


class RabbitManager:
    def __init__(self, user, password, endpoint, runtime_config=None):
        self.user = user
        self.password = password
        self.endpoint = endpoint
        # The runtime_config defines the parameters to create queue, exchange .etc
        self.runtime_config = runtime_config if runtime_config is not None else {}

    # return a requests.Response object in case someone need more info about the Response
    @connection_retry
    def create_user(self, user, password):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "users/" + user)
        body = {
            "password": password,
            "tags": ""
        }
        result = requests.put(url, headers=C_COMMON_HTTP_HEADER,
                              json=body, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.create_user] {result}")
        if result.status_code == 201 or result.status_code == 204:
            return True
        else:
            return False

    def delete_user(self, user):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "users/" + user)
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.delete_user] {result}")
        return result

    def create_vhost(self, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "vhosts/" + vhost)
        result = requests.put(
            url, headers=C_COMMON_HTTP_HEADER, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.create_vhost] {result}")
        self.add_user_to_vhost(self.user, vhost)
        return True

    def delete_vhost(self, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "vhosts/" + vhost)
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.delete_vhost] {result}")
        return result

    def delete_vhosts(self):
        result = self.get_vhosts()
        names = None
        try:
            if result.status_code == 200:
                names = [e["name"] for e in result.json()]
        except BaseException:
            names = None
        LOGGER.debug(f"[rabbitmanager.delete_vhosts] {names}")
        if names is not None:
            LOGGER.debug("[rabbitmanager.delete_vhosts]start to delete_vhosts")
            for name in names:
                self.delete_vhost(name)

    def get_vhosts(self):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "vhosts")
        result = requests.get(url, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.get_vhosts] {result}")
        return result

    def add_user_to_vhost(self, user, vhost):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "{}/{}/{}".format("permissions", vhost, user))
        body = {
            "configure": ".*",
            "write": ".*",
            "read": ".*"
        }

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER,
                              json=body, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.add_user_to_vhost] {result}")

        if result.status_code == 201 or result.status_code == 204:
            return True
        else:
            return False

    def remove_user_from_vhost(self, user, vhost):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "{}/{}/{}".format("permissions", vhost, user))
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.remove_user_from_vhost] {result}")
        return result

    def get_exchanges(self, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}".format("exchanges", vhost))
        result = requests.get(url, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.get_exchanges] {result}")
        try:
            if result.status_code == 200:
                exchange_names = [e["name"] for e in result.json()]
                LOGGER.debug(f"[rabbitmanager.get_exchanges] exchange_names={exchange_names}")
                return exchange_names
            else:
                return None
        except BaseException:
            return None

    def create_exchange(self, vhost, exchange_name):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "{}/{}/{}".format("exchanges", vhost, exchange_name))

        basic_config = {
            "type": "direct",
            "auto_delete": False,
            "durable": True,
            "internal": False,
            "arguments": {}
        }

        exchange_runtime_config = self.runtime_config.get("exchange", {})
        basic_config.update(exchange_runtime_config)

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER,
                              json=basic_config, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def delete_exchange(self, vhost, exchange_name):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "{}/{}/{}".format("exchanges", vhost, exchange_name))
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.delete_exchange] vhost={vhost}, exchange_name={exchange_name}, {result}")
        return result

    def get_policies(self, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}".format("policies", vhost))
        result = requests.get(url, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.get_policies] {result}")
        try:
            if result.status_code == 200:
                policies_names = [e["name"] for e in result.json()]
                LOGGER.debug(f"[rabbitmanager.get_policies] policies_names={policies_names}")
                return policies_names
            else:
                return None
        except BaseException:
            return None

    def delete_policy(self, vhost, policy_name):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "{}/{}/{}".format("policies", vhost, policy_name))
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.delete_policy] vhost={vhost}, policy_name={policy_name}, {result}")
        return result

    @connection_retry
    def create_queue(self, vhost, queue_name):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "{}/{}/{}".format("queues", vhost, queue_name))

        basic_config = {
            "auto_delete": False,
            "durable": True,
            "arguments": {}
        }

        queue_runtime_config = self.runtime_config.get("queue", {})
        basic_config.update(queue_runtime_config)
        LOGGER.debug(basic_config)

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER,
                              json=basic_config, auth=(self.user, self.password))

        LOGGER.debug(f"[rabbitmanager.create_queue] {result}")
        if result.status_code == 201 or result.status_code == 204:
            return True
        else:
            return False

    def get_queue(self, vhost, queue_name):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "{}/{}/{}".format("queues", vhost, queue_name))

        result = requests.get(url, headers=C_COMMON_HTTP_HEADER, auth=(self.user, self.password))
        return result

    def get_queues(self, vhost):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "{}/{}".format("queues", vhost))
        result = requests.get(url, headers=C_COMMON_HTTP_HEADER, auth=(self.user, self.password))
        try:
            if result.status_code == 200:
                queue_names = [e["name"] for e in result.json()]
                LOGGER.debug(f"[rabbitmanager.get_all_queue] queue_names={queue_names}")
                return queue_names
            else:
                return None
        except BaseException:
            return None

    def delete_queue(self, vhost, queue_name):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "{}/{}/{}".format("queues", vhost, queue_name))
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.delete_queue] vhost={vhost}, queue_name={queue_name}, {result}")
        return result

    def get_connections(self):
        url = C_HTTP_TEMPLATE.format(
            self.endpoint, "connections")
        result = requests.get(url, headers=C_COMMON_HTTP_HEADER, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager.get_connections] {result}")
        return result

    def delete_connections(self, vhost=None):
        time.sleep(2)
        result = self.get_connections()
        names = None
        try:
            if result.status_code == 200:
                if vhost is None:
                    names = [e["name"] for e in result.json()]
                else:
                    names = [e["name"] for e in result.json() if e["vhost"] == vhost]
        except BaseException:
            names = None
        LOGGER.debug(f"[rabbitmanager.delete_connections] {names}")
        if names is not None:
            LOGGER.debug("[rabbitmanager.delete_connections] start....")
            for name in names:
                url = C_HTTP_TEMPLATE.format(
                    self.endpoint, "{}/{}".format("connections", name))
                result = requests.delete(url, auth=(self.user, self.password))
                LOGGER.debug(result)

    def bind_exchange_to_queue(self, vhost, exchange_name, queue_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/e/{}/q/{}".format("bindings",
                                                                             vhost,
                                                                             exchange_name,
                                                                             queue_name))

        body = {
            "routing_key": queue_name,
            "arguments": {}
        }

        result = requests.post(
            url, headers=C_COMMON_HTTP_HEADER, json=body, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def unbind_exchange_to_queue(self, vhost, exchange_name, queue_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/e/{}/q/{}/{}".format("bindings",
                                                                                vhost,
                                                                                exchange_name,
                                                                                queue_name,
                                                                                queue_name))

        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    @connection_retry
    def _set_federated_upstream(self, upstream_host, vhost, receive_queue_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}/{}".format("parameters",
                                                                         "federation-upstream",
                                                                         vhost,
                                                                         receive_queue_name))
        upstream_runtime_config = self.runtime_config.get("upstream", {})

        upstream_runtime_config['uri'] = upstream_host
        upstream_runtime_config['queue'] = receive_queue_name.replace(
            "receive", "send", 1)

        body = {
            "value": upstream_runtime_config
        }
        LOGGER.debug(f"[rabbitmanager._set_federated_upstream]set_federated_upstream, url: {url} body: {body}")

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER,
                              json=body, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager._set_federated_upstream] {result}")
        if result.status_code != 201 and result.status_code != 204:
            LOGGER.debug(f"[rabbitmanager._set_federated_upstream] _set_federated_upstream fail. {result}")
            return False

        return True

    def _unset_federated_upstream(self, upstream_name, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}/{}".format("parameters",
                                                                         "federation-upstream",
                                                                         vhost,
                                                                         upstream_name))

        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    @connection_retry
    def _set_federated_queue_policy(self, vhost, receive_queue_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("policies",
                                                                      vhost,
                                                                      receive_queue_name))
        body = {
            "pattern": '^' + receive_queue_name + '$',
            "apply-to": "queues",
            "definition":
                {
                    "federation-upstream": receive_queue_name
                }
        }
        LOGGER.debug(f"[rabbitmanager._set_federated_queue_policy]set_federated_queue_policy, url: {url} body: {body}")

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER,
                              json=body, auth=(self.user, self.password))
        LOGGER.debug(f"[rabbitmanager._set_federated_queue_policy] {result}")
        if result.status_code != 201 and result.status_code != 204:
            LOGGER.debug(f"[rabbitmanager._set_federated_queue_policy] _set_federated_queue_policy fail. {result}")
            return False

        return True

    def _unset_federated_queue_policy(self, policy_name, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("policies",
                                                                      vhost,
                                                                      policy_name))

        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    # Create federate queue with upstream
    def federate_queue(self, upstream_host, vhost, send_queue_name, receive_queue_name):
        time.sleep(1)
        LOGGER.debug(f"[rabbitmanager.federate_queue] create federate_queue {receive_queue_name}")

        result = self._set_federated_upstream(
            upstream_host, vhost, receive_queue_name)

        if result is False:
            # should be logged
            LOGGER.debug(f"[rabbitmanager.federate_queue] result_set_upstream fail.")
            return False

        result = self._set_federated_queue_policy(
            vhost, receive_queue_name)

        if result is False:
            LOGGER.debug(f"[rabbitmanager.federate_queue] result_set_policy fail.")
            return False

        return True

    def de_federate_queue(self, vhost, receive_queue_name):
        result = self._unset_federated_queue_policy(receive_queue_name, vhost)
        LOGGER.debug(
            f"delete federate queue policy status code: {result.status_code}")

        result = self._unset_federated_upstream(receive_queue_name, vhost)
        LOGGER.debug(
            f"delete federate queue upstream status code: {result.status_code}")

        return True

    def clean(self, vhost):
        time.sleep(1)
        queue_names = self.get_queues(vhost)
        if queue_names is not None:
            for name in queue_names:
                self.delete_queue(vhost, name)

        exchange_names = self.get_exchanges(vhost)
        if exchange_names is not None:
            for name in exchange_names:
                self.delete_exchange(vhost, name)

        policy_names = self.get_policies(vhost)
        if policy_names is not None:
            for name in policy_names:
                self.delete_policy(vhost, name)

        self.delete_vhost(vhost=vhost)
        self.delete_connections(vhost=vhost)
