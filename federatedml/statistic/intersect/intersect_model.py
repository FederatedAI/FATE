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
from arch.api.utils import log_utils
from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.model_base import ModelBase
from federatedml.param.intersect_param import IntersectParam
from federatedml.statistic.intersect import RawIntersectionHost, RawIntersectionGuest, RsaIntersectionHost, \
    RsaIntersectionGuest
from federatedml.statistic.intersect.repeat_id_process import RepeatedIDIntersect
from federatedml.util import consts

LOGGER = log_utils.getLogger()


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

    def __init_intersect_method(self):
        LOGGER.info("Using {} intersection, role is {}".format(self.model_param.intersect_method, self.role))
        self.host_party_id_list = self.component_properties.host_party_idlist
        self.guest_party_id = self.component_properties.guest_partyid
        if self.role == consts.HOST:
            self.host_party_id = self.component_properties.local_partyid

        if self.model_param.intersect_method == "rsa":
            if self.role == consts.HOST:
                self.intersection_obj = RsaIntersectionHost(self.model_param)
                self.intersection_obj.host_party_id = self.host_party_id
            elif self.role == consts.GUEST:
                self.intersection_obj = RsaIntersectionGuest(self.model_param)
            else:
                raise ValueError("role {} is not support".format(self.role))

            self.intersection_obj.guest_party_id = self.guest_party_id

        elif self.model_param.intersect_method == "raw":
            if self.role == consts.HOST:
                self.intersection_obj = RawIntersectionHost(self.model_param)
            elif self.role == consts.GUEST:
                self.intersection_obj = RawIntersectionGuest(self.model_param)
            else:
                raise ValueError("role {} is not support".format(self.role))
            self.intersection_obj.task_id = self.taskid
        else:
            raise ValueError("intersect_method {} is not support yet".format(self.model_param.intersect_method))

        if self.role == consts.HOST:
            self.intersection_obj.host_party_id = self.host_party_id
        self.intersection_obj.guest_party_id = self.guest_party_id
        self.intersection_obj.host_party_id_list = self.host_party_id_list

    def fit(self, data):
        self.__init_intersect_method()

        if self.model_param.repeated_id_process:
            if self.model_param.intersect_cache_param.use_cache is True and self.model_param.intersect_method == consts.RSA:
                raise ValueError("Not support cache module while repeated id process.")

            if len(self.host_party_id_list) > 1 and self.model_param.repeated_id_owner != consts.GUEST:
                raise ValueError("While multi-host, repeated_id_owner should be guest.")

            proc_obj = RepeatedIDIntersect(repeated_id_owner=self.model_param.repeated_id_owner, role=self.role)
            data = proc_obj.run(data=data)

        self.intersect_ids = self.intersection_obj.run(data)
        LOGGER.info("Finish intersection")

        if self.intersect_ids:
            self.intersect_num = self.intersect_ids.count()
            self.intersect_rate = self.intersect_num * 1.0 / data.count()

        self.callback_metric(metric_name=self.metric_name,
                             metric_namespace=self.metric_namespace,
                             metric_data=[Metric("intersect_count", self.intersect_num),
                                          Metric("intersect_rate", self.intersect_rate)])
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=self.metric_name,
                                     metric_meta=MetricMeta(name=self.metric_name, metric_type=self.metric_type))

    def save_data(self):
        if self.intersect_ids is not None:
            LOGGER.info("intersect_ids:{}".format(self.intersect_ids.count()))
        return self.intersect_ids


class IntersectHost(IntersectModelBase):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST


class IntersectGuest(IntersectModelBase):
    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
