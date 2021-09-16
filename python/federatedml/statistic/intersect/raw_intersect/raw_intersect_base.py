#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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

from federatedml.secureprotol.hash.hash_factory import Hash
from federatedml.statistic.intersect import Intersect
from federatedml.transfer_variable.transfer_class.raw_intersect_transfer_variable import RawIntersectTransferVariable
from federatedml.util import consts, LOGGER


class RawIntersect(Intersect):
    def __init__(self):
        super().__init__()
        self.role = None
        self.transfer_variable = RawIntersectTransferVariable()
        self.task_version_id = None
        self.tracker = None

    def load_params(self, param):
        # self.only_output_key = param.only_output_key
        # self.sync_intersect_ids = param.sync_intersect_ids
        super().load_params(param=param)
        self.raw_params = param.raw_params
        self.use_hash = self.raw_params.use_hash
        self.hash_method = self.raw_params.hash_method
        self.base64 = self.raw_params.base64
        self.salt = self.raw_params.salt

        self.join_role = self.raw_params.join_role
        self.hash_operator = Hash(self.hash_method, self.base64)

    def intersect_send_id(self, data_instances):
        sid_hash_pair = None
        if self.use_hash and self.hash_method != "none":
            sid_hash_pair = data_instances.map(
                lambda k, v: (Intersect.hash(k, self.hash_operator, self.salt), k))
            data_sid = sid_hash_pair.mapValues(lambda v: 1)
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
                LOGGER.info(f"raw intersect send role is guest, "
                            f"and has {self.host_party_id_list} hosts, remote the final intersect_ids to hosts")
                self.transfer_variable.sync_intersect_ids_multi_hosts.remote(recv_intersect_ids,
                                                                             role=consts.HOST,
                                                                             idx=-1)

            if sid_hash_pair and recv_intersect_ids is not None:
                hash_intersect_ids_map = recv_intersect_ids.join(sid_hash_pair, lambda r, s: s)
                intersect_ids = hash_intersect_ids_map.map(lambda k, v: (v, 'intersect_id'))
            else:
                intersect_ids = recv_intersect_ids
        else:
            LOGGER.info("Not Get intersect ids from role-join!")

        return intersect_ids

    def intersect_join_id(self, data_instances):
        LOGGER.info("Join id role is {}".format(self.role))

        sid_hash_pair = None
        if self.use_hash and self.hash_method != "none":
            sid_hash_pair = data_instances.map(
                lambda k, v: (Intersect.hash(k, self.hash_operator, self.salt), k))
            data_sid = sid_hash_pair.mapValues(lambda v: 1)
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
            hash_intersect_ids = recv_ids_list[0].join(data_sid, lambda i, d: "intersect_id")
        elif ids_list_size > 1:
            hash_intersect_ids_list = []
            for ids in recv_ids_list:
                hash_intersect_ids_list.append(ids.join(data_sid, lambda i, d: "intersect_id"))
            hash_intersect_ids = self.get_common_intersection(hash_intersect_ids_list)
        else:
            hash_intersect_ids = None
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

            intersect_ids_federation.remote(hash_intersect_ids,
                                            role=send_role,
                                            idx=-1)
            LOGGER.info("Remote intersect ids to role-send")

            if self.role == consts.HOST and len(self.host_party_id_list) > 1:
                LOGGER.info(f"raw intersect join role is host,"
                            f"and has {self.host_party_id_list} hosts, get the final intersect_ids from guest")
                hash_intersect_ids = self.transfer_variable.sync_intersect_ids_multi_hosts.get(idx=0)

        if sid_hash_pair:
            hash_intersect_ids_map = hash_intersect_ids.join(sid_hash_pair, lambda r, s: s)
            intersect_ids = hash_intersect_ids_map.map(lambda k, v: (v, 'intersect_id'))
        else:
            intersect_ids = hash_intersect_ids

        """
        if self.task_version_id is not None:
            namespace = "#".join([str(self.guest_party_id), str(self.host_party_id), "mountain"])
            for k, v in enumerate(recv_ids_list):
                table_name = '_'.join([self.task_version_id, str(k)])
                self.tracker.job_tracker.save_as_table(v, table_name, namespace)
                LOGGER.info("save guest_{}'s id in name:{}, namespace:{}".format(k, table_name, namespace))
        """
        return intersect_ids
