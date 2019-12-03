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
import hashlib

from arch.api.utils import log_utils
from federatedml.secureprotol.encode import Encode
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.raw_intersect_transfer_variable import RawIntersectTransferVariable

LOGGER = log_utils.getLogger()


class Intersect(object):
    def __init__(self, intersect_params):
        self.transfer_variable = None
        self.only_output_key = intersect_params.only_output_key
        self.sync_intersect_ids = intersect_params.sync_intersect_ids

        self._guest_id = None
        self._host_id = None
        self._host_id_list = None

    @property
    def guest_party_id(self):
        return self._guest_id

    @guest_party_id.setter
    def guest_party_id(self, guest_id):
        if not isinstance(guest_id, int):
            raise ValueError("party id should be integer, but get {}".format(guest_id))
        self._guest_id = guest_id

    @property
    def host_party_id(self):
        return self._host_id

    @host_party_id.setter
    def host_party_id(self, host_id):
        if not isinstance(host_id, int):
            raise ValueError("party id should be integer, but get {}".format(host_id))
        self._host_id = host_id

    @property
    def host_party_id_list(self):
        return self._host_id_list

    @host_party_id_list.setter
    def host_party_id_list(self, host_id_list):
        if not isinstance(host_id_list, list):
            raise ValueError(
                "type host_party_id should be list, but get {} with {}".format(type(host_id_list), host_id_list))
        self._host_id_list = host_id_list

    def run(self, data_instances):
        raise NotImplementedError("method init must be define")

    def set_flowid(self, flowid=0):
        if self.transfer_variable is not None:
            self.transfer_variable.set_flowid(flowid)

    def _set_schema(self, schema):
        self.schema = schema

    def _get_schema(self):
        return self.schema

    def _get_value_from_data(self, intersect_ids, data_instances):
        intersect_ids = intersect_ids.join(data_instances, lambda i, d: d)
        intersect_ids.schema = data_instances.schema
        LOGGER.info("get intersect data_instances!")
        return intersect_ids

    def get_common_intersection(self, intersect_ids_list: list):
        if len(intersect_ids_list) == 1:
            return intersect_ids_list[0]

        intersect_ids = None
        for i, value in enumerate(intersect_ids_list):
            if intersect_ids is None:
                intersect_ids = value
                continue
            intersect_ids = intersect_ids.join(value, lambda id, v: "id")

        return intersect_ids


class RsaIntersect(Intersect):
    def __init__(self, intersect_params):
        super().__init__(intersect_params)
        self.intersect_cache_param = intersect_params.intersect_cache_param

    @staticmethod
    def hash(value):
        return hashlib.sha256(bytes(str(value), encoding='utf-8')).hexdigest()


class RawIntersect(Intersect):
    def __init__(self, intersect_params):
        super().__init__(intersect_params)
        self.role = None
        self.with_encode = intersect_params.with_encode
        self.transfer_variable = RawIntersectTransferVariable()
        self.encode_params = intersect_params.encode_params

        self.task_id = None

    def intersect_send_id(self, data_instances):
        sid_encode_pair = None
        if self.with_encode and self.encode_params.encode_method != "none":
            if Encode.is_support(self.encode_params.encode_method):
                encode_operator = Encode(self.encode_params.encode_method, self.encode_params.base64)
                sid_encode_pair = data_instances.map(
                    lambda k, v: (encode_operator.compute(k, postfit_salt=self.encode_params.salt), k))
                data_sid = sid_encode_pair.mapValues(lambda v: 1)
            else:
                raise ValueError("Unknown encode_method, please check the configure of encode_param")
        else:
            data_sid = data_instances.mapValues(lambda v: 1)

        LOGGER.info("Send id role is {}".format(self.role))

        if self.role == consts.GUEST:
            send_ids_federation = self.transfer_variable.send_ids_guest
            recv_role = consts.HOST
        elif self.role == consts.HOST:
            send_ids_federation = self.transfer_variable.send_ids_host
            recv_role = consts.GUEST
        else:
            raise ValueError("Unknown intersect role, please check the code")

        send_ids_federation.remote(data_sid,
                                   role=recv_role,
                                   idx=-1)

        LOGGER.info("Remote data_sid to role-join")
        intersect_ids = None
        if self.sync_intersect_ids:
            if self.role == consts.HOST:
                intersect_ids_federation = self.transfer_variable.intersect_ids_guest
            elif self.role == consts.GUEST:
                intersect_ids_federation = self.transfer_variable.intersect_ids_host
            else:
                raise ValueError("Unknown intersect role, please check the code")

            recv_intersect_ids_list = intersect_ids_federation.get(idx=-1)
            LOGGER.info("Get intersect ids from role-join!")

            ids_list_size = len(recv_intersect_ids_list)
            LOGGER.info("recv_intersect_ids_list's size is {}".format(ids_list_size))

            recv_intersect_ids = self.get_common_intersection(recv_intersect_ids_list)

            if self.role == consts.GUEST and len(self.host_party_id_list) > 1:
                LOGGER.info(
                    "raw intersect send role is guest, and has {} hosts, remote the final intersect_ids to hosts".format(
                        len(self.host_party_id_list)))
                self.transfer_variable.sync_intersect_ids_multi_hosts.remote(recv_intersect_ids,
                                                                             role=consts.HOST,
                                                                             idx=-1)

            if sid_encode_pair and recv_intersect_ids is not None:
                encode_intersect_ids_map = recv_intersect_ids.join(sid_encode_pair, lambda r, s: s)
                intersect_ids = encode_intersect_ids_map.map(lambda k, v: (v, 'intersect_id'))
            else:
                intersect_ids = recv_intersect_ids
        else:
            LOGGER.info("Not Get intersect ids from role-join!")

        if not self.only_output_key:
            intersect_ids = self._get_value_from_data(intersect_ids, data_instances)

        return intersect_ids

    def intersect_join_id(self, data_instances):
        LOGGER.info("Join id role is {}".format(self.role))

        sid_encode_pair = None
        if self.with_encode and self.encode_params.encode_method != "none":
            if Encode.is_support(self.encode_params.encode_method):
                encode_operator = Encode(self.encode_params.encode_method, self.encode_params.base64)
                sid_encode_pair = data_instances.map(
                    lambda k, v: (encode_operator.compute(k, postfit_salt=self.encode_params.salt), k))
                data_sid = sid_encode_pair.mapValues(lambda v: 1)
            else:
                raise ValueError("Unknown encode_method, please check the configure of encode_param")
        else:
            data_sid = data_instances.mapValues(lambda v: 1)

        if self.role == consts.HOST:
            send_ids_federation = self.transfer_variable.send_ids_guest
        elif self.role == consts.GUEST:
            send_ids_federation = self.transfer_variable.send_ids_host
        else:
            raise ValueError("Unknown intersect role, please check the code")

        recv_ids_list = send_ids_federation.get(idx=-1)

        ids_list_size = len(recv_ids_list)
        LOGGER.info("Get ids_list from role-send, ids_list size is {}".format(len(recv_ids_list)))

        if ids_list_size == 1:
            encode_intersect_ids = recv_ids_list[0].join(data_sid, lambda i, d: "intersect_id")
        elif ids_list_size > 1:
            encode_intersect_ids_list = []
            for ids in recv_ids_list:
                encode_intersect_ids_list.append(ids.join(data_sid, lambda i, d: "intersect_id"))
            encode_intersect_ids = self.get_common_intersection(encode_intersect_ids_list)
        else:
            encode_intersect_ids = None
        LOGGER.info("Finish intersect_ids computing")

        if self.sync_intersect_ids:
            if self.role == consts.GUEST:
                intersect_ids_federation = self.transfer_variable.intersect_ids_guest
                send_role = consts.HOST
            elif self.role == consts.HOST:
                intersect_ids_federation = self.transfer_variable.intersect_ids_host
                send_role = consts.GUEST
            else:
                raise ValueError("Unknown intersect role, please check the code")

            intersect_ids_federation.remote(encode_intersect_ids,
                                            role=send_role,
                                            idx=-1)
            LOGGER.info("Remote intersect ids to role-send")

            if self.role == consts.HOST and len(self.host_party_id_list) > 1:
                LOGGER.info(
                    "raw intersect join role is host, and has {} hosts, get the final intersect_ids from guest".format(
                        len(self.host_party_id_list)))
                encode_intersect_ids = self.transfer_variable.sync_intersect_ids_multi_hosts.get(idx=0)

        if sid_encode_pair:
            encode_intersect_ids_map = encode_intersect_ids.join(sid_encode_pair, lambda r, s: s)
            intersect_ids = encode_intersect_ids_map.map(lambda k, v: (v, 'intersect_id'))
        else:
            intersect_ids = encode_intersect_ids

        if not self.only_output_key:
            intersect_ids = self._get_value_from_data(intersect_ids, data_instances)

        if self.task_id is not None:
            namespace = "#".join([str(self.guest_party_id), str(self.host_party_id), "mountain"])
            for k, v in enumerate(recv_ids_list):
                table_name = '_'.join([self.task_id, str(k)])
                v.save_as(table_name, namespace)
                LOGGER.info("save guest_{}'s id in name:{}, namespace:{}".format(k, table_name, namespace))

        return intersect_ids
