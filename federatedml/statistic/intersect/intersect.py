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

from arch.api.federation import remote, get
from arch.api.utils import log_utils
from federatedml.secureprotol.encode import Encode
from federatedml.util import consts
from federatedml.util import IntersectParamChecker
from federatedml.util.transfer_variable import RawIntersectTransferVariable

LOGGER = log_utils.getLogger()


class Intersect(object):
    def __init__(self, intersect_params):
        self.transfer_variable = None
        self.only_output_key = intersect_params.only_output_key
        IntersectParamChecker.check_param(intersect_params)

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
        LOGGER.info("get intersect data_instances!")
        intersect_ids.schema['header'] = data_instances.schema.get("header")
        return intersect_ids


class RsaIntersect(Intersect):
    def __init__(self, intersect_params):
        super().__init__(intersect_params)


class RawIntersect(Intersect):
    def __init__(self, intersect_params):
        super().__init__(intersect_params)
        self.role = None
        self.send_intersect_id_flag = intersect_params.is_send_intersect_ids
        self.with_encode = intersect_params.with_encode
        self.transfer_variable = RawIntersectTransferVariable()
        self.encode_params = intersect_params.encode_params

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
            send_ids_name = self.transfer_variable.send_ids_guest.name
            send_ids_tag = self.transfer_variable.generate_transferid(self.transfer_variable.send_ids_guest)
            recv_role = consts.HOST
        elif self.role == consts.HOST:
            send_ids_name = self.transfer_variable.send_ids_host.name
            send_ids_tag = self.transfer_variable.generate_transferid(self.transfer_variable.send_ids_host)
            recv_role = consts.GUEST
        else:
            raise ValueError("Unknown intersect role, please check the code")

        remote(data_sid,
               name=send_ids_name,
               tag=send_ids_tag,
               role=recv_role,
               idx=0)

        LOGGER.info("Remote data_sid to role-join")
        intersect_ids = None
        if self.send_intersect_id_flag:
            if self.role == consts.HOST:
                intersect_ids_name = self.transfer_variable.intersect_ids_guest.name
                intersect_ids_tag = self.transfer_variable.generate_transferid(
                    self.transfer_variable.intersect_ids_guest)
            elif self.role == consts.GUEST:
                intersect_ids_name = self.transfer_variable.intersect_ids_host.name
                intersect_ids_tag = self.transfer_variable.generate_transferid(
                    self.transfer_variable.intersect_ids_host)
            else:
                raise ValueError("Unknown intersect role, please check the code")

            recv_intersect_ids = get(name=intersect_ids_name,
                                     tag=intersect_ids_tag,
                                     idx=0)

            if sid_encode_pair:
                encode_intersect_ids = recv_intersect_ids.join(sid_encode_pair, lambda r, s: s)
                intersect_ids = encode_intersect_ids.map(lambda k, v: (v, 'intersect_id'))
            else:
                intersect_ids = recv_intersect_ids

            LOGGER.info("Get intersect ids from role-join!")
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
            send_ids_name = self.transfer_variable.send_ids_guest.name
            send_ids_tag = self.transfer_variable.generate_transferid(self.transfer_variable.send_ids_guest)
        elif self.role == consts.GUEST:
            send_ids_name = self.transfer_variable.send_ids_host.name
            send_ids_tag = self.transfer_variable.generate_transferid(self.transfer_variable.send_ids_host)
        else:
            raise ValueError("Unknown intersect role, please check the code")

        recv_ids = get(name=send_ids_name,
                       tag=send_ids_tag,
                       idx=0)

        LOGGER.info("Get intersect_host_ids from role-send")
        send_intersect_ids = recv_ids.join(data_sid, lambda i, d: "intersect_id")
        LOGGER.info("Finish intersect_ids computing")

        if self.send_intersect_id_flag:
            if self.role == consts.GUEST:
                intersect_ids_name = self.transfer_variable.intersect_ids_guest.name
                intersect_ids_tag = self.transfer_variable.generate_transferid(
                    self.transfer_variable.intersect_ids_guest)
                recv_role = consts.HOST
            elif self.role == consts.HOST:
                intersect_ids_name = self.transfer_variable.intersect_ids_host.name
                intersect_ids_tag = self.transfer_variable.generate_transferid(
                    self.transfer_variable.intersect_ids_host)
                recv_role = consts.GUEST
            else:
                raise ValueError("Unknown intersect role, please check the code")

            remote(send_intersect_ids,
                   name=intersect_ids_name,
                   tag=intersect_ids_tag,
                   role=recv_role,
                   idx=0)
            LOGGER.info("Remote intersect ids to role-send")

        if sid_encode_pair:
            encode_intersect_ids = send_intersect_ids.join(sid_encode_pair, lambda r, s: s)
            intersect_ids = encode_intersect_ids.map(lambda k, v: (v, 'intersect_id'))
        else:
            intersect_ids = send_intersect_ids

        if not self.only_output_key:
            intersect_ids = self._get_value_from_data(intersect_ids, data_instances)

        return intersect_ids
