########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

import requests

from fate_arch.common import log

LOGGER = log.getLogger()

C_HTTP_TEMPLATE = "http://{}/api/{}"
C_COMMON_HTTP_HEADER = {'Content-Type': 'application/json'}

"""
APIs are refered to https://rawcdn.githack.com/rabbitmq/rabbitmq-management/v3.8.3/priv/www/api/index.html
"""


class RabbitManager:
    def __init__(self, user, password, endpoint):
        self.user = user
        self.password = password
        self.endpoint = endpoint

    # return a requests.Response object in case someone need more info about the Response
    def create_user(self, user, password):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "users/" + user)
        body = {
            "password": password,
            "tags": ""
        }
        result = requests.put(url, headers=C_COMMON_HTTP_HEADER, json=body, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def delete_user(self, user):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "users/" + user)
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def create_vhost(self, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "vhosts/" + vhost)
        result = requests.put(url, headers=C_COMMON_HTTP_HEADER, auth=(self.user, self.password))
        LOGGER.debug(result)
        self.add_user_to_vhost(self.user, vhost)
        return result

    def delete_vhost(self, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "vhosts/" + vhost)
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def get_vhosts(self):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "vhosts")
        result = requests.get(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def add_user_to_vhost(self, user, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("permissions", vhost, user))
        body = {
            "configure": ".*",
            "write": ".*",
            "read": ".*"
        }

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER, json=body, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def remove_user_from_vhost(self, user, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("permissions", vhost, user))
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def get_exchanges(self):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "exchanges/federated")
        result = requests.get(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def create_exchange(self, vhost, exchange_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("exchanges", vhost, exchange_name))

        body = {
            "type": "direct",
            "auto_delete": False,
            "durable": True,
            "internal": False,
            "arguments": {}
        }

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER, json=body, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def delete_exchange(self, vhost, exchange_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("exchanges", vhost, exchange_name))
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def create_queue(self, vhost, queue_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("queues", vhost, queue_name))

        body = {
            "auto_delete": False,
            "durable": True,
            "arguments": {}
        }

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER, json=body, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def delete_queue(self, vhost, queue_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("queues", vhost, queue_name))
        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def bind_exchange_to_queue(self, vhost, exchange_name, queue_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/e/{}/q/{}".format("bindings",
                                                                             vhost,
                                                                             exchange_name,
                                                                             queue_name))

        body = {
            "routing_key": queue_name,
            "arguments": {}
        }

        result = requests.post(url, headers=C_COMMON_HTTP_HEADER, json=body, auth=(self.user, self.password))
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

    def _set_federated_upstream(self, upstream_host, vhost, receive_queue_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}/{}".format("parameters",
                                                                         "federation-upstream",
                                                                         vhost,
                                                                         receive_queue_name))
        body = {
            "value":
                {
                    "uri": upstream_host,
                    "queue": receive_queue_name.replace("receive", "send")
                }
        }
        LOGGER.debug(f"set_federated_upstream, url: {url} body: {body}")

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER, json=body, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def _unset_federated_upstream(self, upstream_name, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}/{}".format("parameters",
                                                                         "federation-upstream",
                                                                         vhost,
                                                                         upstream_name))

        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def _set_federated_queue_policy(self, vhost, receive_queue_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("policies",
                                                                      vhost,
                                                                      receive_queue_name))
        body = {
            "pattern": receive_queue_name,
            "apply-to": "queues",
            "definition":
                {
                    "federation-upstream": receive_queue_name
                }
        }
        LOGGER.debug(f"set_federated_queue_policy, url: {url} body: {body}")

        result = requests.put(url, headers=C_COMMON_HTTP_HEADER, json=body, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    def _unset_federated_queue_policy(self, policy_name, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("policies",
                                                                      vhost,
                                                                      policy_name))

        result = requests.delete(url, auth=(self.user, self.password))
        LOGGER.debug(result)
        return result

    # Create federate queue with upstream
    def federate_queue(self, upstream_host, vhost, send_queue_name, receive_queue_name):
        import time
        time.sleep(1)
        LOGGER.debug(f"create federate_queue {send_queue_name} {receive_queue_name}")

        result_set_upstream = self._set_federated_upstream(upstream_host, vhost, receive_queue_name)

        result_set_policy = self._set_federated_queue_policy(vhost, receive_queue_name)

        if result_set_upstream.status_code != requests.codes.created:
            # should be loogged
            print("result_set_upstream fail.")
            print(result_set_upstream.text)
            # caller need to check None
            # return None 
        elif result_set_policy.status_code != requests.codes.created:
            print("result_set_policy fail.")
            print(result_set_policy.text)
            # return None

    def de_federate_queue(self, vhost, receive_queue_name):
        result = self._unset_federated_queue_policy(receive_queue_name, vhost)
        LOGGER.debug(f"delete federate queue policy status code: {result.status_code}")

        result = self._unset_federated_upstream(receive_queue_name, vhost)
        LOGGER.debug(f"delete federate queue upstream status code: {result.status_code}")

        return True
