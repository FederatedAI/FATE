########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

import requests

from arch.api.utils import log_utils
from arch.api.utils import string_utils

LOGGER = log_utils.getLogger()

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

    def _set_federated_upstream(self, upstream_host, upstream_name, vhost):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}/{}".format("parameters",
                                                                         "federation-upstream",
                                                                         vhost,
                                                                         upstream_name))
        body = {
            "value":
                {
                    "uri": upstream_host,
                    "queue": "{}-{}".format("send", vhost)
                }
        }

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

    def _set_federated_queue_policy(self, upstream_name, vhost, policy_name):
        url = C_HTTP_TEMPLATE.format(self.endpoint, "{}/{}/{}".format("policies",
                                                                      vhost,
                                                                      policy_name))
        body = {
            "pattern": "^receive",
            "apply-to": "queues",
            "definition":
                {
                    "federation-upstream": upstream_name
                }
        }

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
    def federate_queue(self, upstream_host, vhost, union_name=""):
        import time
        time.sleep(5)
        # union name is used for both upstream name and policy name
        # give a random string if not union_name was provided
        if union_name == "":
            union_name = string_utils.RandomString()

        result_set_upstream = self._set_federated_upstream(upstream_host, union_name, vhost)

        result_set_policy = self._set_federated_queue_policy(union_name, vhost, union_name)

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
        # else:
        # return union_name for later operation
        return union_name

    def de_federate_queue(self, union_name, vhost):
        result = self._unset_federated_queue_policy(union_name, vhost)
        print(result.status_code)

        result = self._unset_federated_upstream(union_name, vhost)
        print(result.status_code)

        return True


'''
if __name__ == "__main__":

    import json, time, string, random
    # the below are some test cases
    rabbit_manager = RabbitManager("guest", "guest", "localhost:15672")

    # 2 queue
    # "send-{job_id}" for sending message
    # "receive-{job_id}" for receiving message
    vhost = rabbit_manager.GetVhosts().json()
    print(len(vhost))

    exchanges = rabbit_manager.GetExchanges().json()
    for exchange in exchanges:
        print exchange["name"]

    # create user
    print("Creating user")
    result = rabbit_manager.CreateUser("happy", "happy")
    print(result.status_code)
    print("Finish create user")
    print("")

    # create vhost
    print("Creating vhost")
    result = rabbit_manager.CreateVhost("testing1")
    print(result.status_code)
    print("Finish create vhost")
    print("")

    # add user to vhost
    print("Adding user to vhost")
    result = rabbit_manager.AddUserToVhost("happy", "testing1")
    print(result.status_code)
    print("Finish add user to vhost")
    print("")

    # create exchange
    print("Creating exchange")
    result = rabbit_manager.CreateExchange("testing1", "testing1-exchange")
    print(result.status_code)
    print("Finish create exchange")
    print("")

    # create queue
    print("Creating queue")
    result = rabbit_manager.CreateQueue("testing1", "testing1-queue")
    print(result.status_code)
    print("Finish creating queue")
    print("")

    # bind queue to exchange
    print("Binding queue to exchange")
    result = rabbit_manager.BindExchangeToQueue("testing1", "testing1-exchange", "testing1-queue")
    print(result.status_code)
    print("Finish binding queue to exchange")
    print("")

    # create Federate queue
    ## create new queue for federation
    print("Creating federated queue ")
    queue_name = "receive-testing1"
    rabbit_manager.CreateQueue("testing1", queue_name)

    ## construct host uri with username, password and vhost
    #host_uri = "amqp://{}:{}@192.168.1.1:5673/{}".format("happy", "happy", "testing1") 
    host_uri = "amqp://{}:{}@192.168.1.1:5672".format("happy", "happy") 
    print(host_uri)
    result = rabbit_manager.FederateQueue(host_uri, "testing1", queue_name)
    print(result)
    print("Finish create federated queue")
    print("")

    time.sleep(30)

    print("Now start the clean up")

    # defederate queue
    print("Start defederate queue")
    rabbit_manager.DeFederateQueue(result, "testing1")

    ## delete federated queue
    rabbit_manager.DeleteQueue("testing1", queue_name)
    print("Finish defederated queue")
    print("")

    # unbind queue to exchange
    print("Unbinding exchange and queue")
    result = rabbit_manager.UnbindExchangeToQueue("testing1", "testing1-exchange", "testing1-queue")
    print(result.status_code)
    print("Finish unbind exchange and queue")
    print("")

    # delete queue
    print("Deleting queue")
    rabbit_manager.DeleteQueue("testing1", "testing1-queue")
    print("Finish delete queue")
    print("")

    # delete exchange
    print("Deleting exchange")
    rabbit_manager.DeleteExchange("testing1", "testing1-exchange")
    print("Finish delete exchange")
    print("")

    # remove user from vhost
    print("Removing user from vhost")
    result = rabbit_manager.RemoveUserFromVhost("happy", "testing1")
    print(result.status_code)
    print("Finish Remove user from vhost")

    print("Deleting vhost")
    result = rabbit_manager.DeleteVhost("testing1")
    print(result.status_code)
    print("Finish delete vhost")
    print("")

    print("Deleting user")
    result = rabbit_manager.DeleteUser("happy")
    print(result.status_code)
    print("Finish delete user")
    print("")
'''
