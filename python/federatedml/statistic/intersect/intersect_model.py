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

from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.feature.instance import Instance
from federatedml.model_base import ModelBase
from federatedml.param.intersect_param import IntersectParam
from federatedml.statistic.intersect import RawIntersectionHost, RawIntersectionGuest, RsaIntersectionHost, \
    RsaIntersectionGuest
from federatedml.statistic.intersect.repeat_id_process import RepeatedIDIntersect
from federatedml.transfer_variable.transfer_class.intersection_func_transfer_variable import \
    IntersectionFuncTransferVariable
from federatedml.util import consts, LOGGER


class IntersectModelBase(ModelBase):
    def __init__(self):
        super().__init__()
        self.intersection_obj = None
        self.intersect_num = -1
        self.intersect_rate = -1
        self.intersect_ids = None
        self.metric_name = "intersection"
        self.metric_namespace = "train"
        self.metric_type = "INTERSECTION"
        self.model_param = IntersectParam()
        self.role = None

        self.guest_party_id = None
        self.host_party_id = None
        self.host_party_id_list = None

        self.transfer_variable = IntersectionFuncTransferVariable()

    def init_intersect_method(self):
        LOGGER.info("Using {} intersection, role is {}".format(self.model_param.intersect_method, self.role))
        self.host_party_id_list = self.component_properties.host_party_idlist
        self.guest_party_id = self.component_properties.guest_partyid

        if self.role not in [consts.HOST, consts.GUEST]:
            raise ValueError("role {} is not support".format(self.role))

    def get_model_summary(self):
        return {"intersect_num": self.intersect_num, "intersect_rate": self.intersect_rate}

    def __share_info(self, data):
        LOGGER.info("Start to share information with another role")
        info_share = self.transfer_variable.info_share_from_guest if self.model_param.info_owner == consts.GUEST else \
            self.transfer_variable.info_share_from_host
        party_role = consts.GUEST if self.model_param.info_owner == consts.HOST else consts.HOST

        if self.role == self.model_param.info_owner:
            if data.schema.get('header') is not None:
                try:
                    share_info_col_idx = data.schema.get('header').index(consts.SHARE_INFO_COL_NAME)

                    one_data = data.first()
                    if isinstance(one_data[1], Instance):
                        share_data = data.join(self.intersect_ids, lambda d, i: [d.features[share_info_col_idx]])
                    else:
                        share_data = data.join(self.intersect_ids, lambda d, i: [d[share_info_col_idx]])

                    info_share.remote(share_data,
                                      role=party_role,
                                      idx=-1)
                    LOGGER.info("Remote share information to {}".format(party_role))

                except Exception as e:
                    LOGGER.warning("Something unexpected:{}, share a empty information to {}".format(e, party_role))
                    share_data = self.intersect_ids.mapValues(lambda v: ['null'])
                    info_share.remote(share_data,
                                      role=party_role,
                                      idx=-1)
            else:
                raise ValueError(
                    "'allow_info_share' is true, and 'info_owner' is {}, but can not header in data, not to do information sharing".format(
                        self.model_param.info_owner))
        else:
            self.intersect_ids = info_share.get(idx=0)
            self.intersect_ids.schema['header'] = [consts.SHARE_INFO_COL_NAME]
            LOGGER.info(
                "Get share information from {}, header:{}".format(self.model_param.info_owner, self.intersect_ids))

        return self.intersect_ids

    def fit(self, data):
        self.init_intersect_method()

        if self.model_param.repeated_id_process:
            if self.model_param.intersect_cache_param.use_cache is True and self.model_param.intersect_method == consts.RSA:
                raise ValueError("Not support cache module while repeated id process.")

            if len(self.host_party_id_list) > 1 and self.model_param.repeated_id_owner != consts.GUEST:
                raise ValueError("While multi-host, repeated_id_owner should be guest.")

            proc_obj = RepeatedIDIntersect(repeated_id_owner=self.model_param.repeated_id_owner, role=self.role)
            data = proc_obj.run_intersect(data=data)

        if self.model_param.allow_info_share:
            if self.model_param.intersect_method == consts.RSA and self.model_param.info_owner == consts.GUEST \
                    or self.model_param.intersect_method == consts.RAW and self.model_param.join_role == self.model_param.info_owner:
                self.model_param.sync_intersect_ids = False

        self.intersect_ids = self.intersection_obj.run_intersect(data)

        if self.model_param.allow_info_share:
            self.intersect_ids = self.__share_info(data)

        LOGGER.info("Finish intersection")

        if self.intersect_ids:
            self.intersect_num = self.intersect_ids.count()
            self.intersect_rate = self.intersect_num * 1.0 / data.count()

        self.set_summary(self.get_model_summary())

        self.callback_metric(metric_name=self.metric_name,
                             metric_namespace=self.metric_namespace,
                             metric_data=[Metric("intersect_count", self.intersect_num),
                                          Metric("intersect_rate", self.intersect_rate)])
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=self.metric_name,
                                     metric_meta=MetricMeta(name=self.metric_name, metric_type=self.metric_type))

    def save_data(self):
        if self.intersect_ids is not None:
            LOGGER.info("intersect_ids count:{}".format(self.intersect_ids.count()))
            LOGGER.info("intersect_ids header schema:{}".format(self.intersect_ids.schema))
        return self.intersect_ids

    def check_consistency(self):
        pass


class IntersectHost(IntersectModelBase):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST

    def init_intersect_method(self):
        super().init_intersect_method()
        self.host_party_id = self.component_properties.local_partyid

        if self.model_param.intersect_method == "rsa":
            self.intersection_obj = RsaIntersectionHost()

        elif self.model_param.intersect_method == "raw":
            self.intersection_obj = RawIntersectionHost()
            self.intersection_obj.tracker = self.tracker
            self.intersection_obj.task_version_id = self.task_version_id
        else:
            raise ValueError("intersect_method {} is not support yet".format(self.model_param.intersect_method))

        self.intersection_obj.host_party_id = self.host_party_id
        self.intersection_obj.guest_party_id = self.guest_party_id
        self.intersection_obj.host_party_id_list = self.host_party_id_list
        self.intersection_obj.load_params(self.model_param)


class IntersectGuest(IntersectModelBase):
    def __init__(self):
        super().__init__()
        self.role = consts.GUEST

    def init_intersect_method(self):
        super().init_intersect_method()

        if self.model_param.intersect_method == "rsa":
            self.intersection_obj = RsaIntersectionGuest()
            self.intersection_obj.guest_party_id = self.guest_party_id

        elif self.model_param.intersect_method == "raw":
            self.intersection_obj = RawIntersectionGuest()
            self.intersection_obj.tracker = self.tracker
            self.intersection_obj.task_version_id = self.task_version_id
        else:
            raise ValueError("intersect_method {} is not support yet".format(self.model_param.intersect_method))

        self.intersection_obj.guest_party_id = self.guest_party_id
        self.intersection_obj.host_party_id_list = self.host_party_id_list
        self.intersection_obj.load_params(self.model_param)
