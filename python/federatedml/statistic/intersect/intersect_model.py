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


import numpy as np

from fate_arch.common.base_utils import fate_uuid
from federatedml.feature.instance import Instance
from federatedml.model_base import Metric, MetricMeta
from federatedml.model_base import ModelBase
from federatedml.param.intersect_param import IntersectParam
from federatedml.secureprotol.hash.hash_factory import Hash
from federatedml.statistic import data_overview
from federatedml.statistic.intersect import RawIntersectionHost, RawIntersectionGuest, RsaIntersectionHost, \
    RsaIntersectionGuest, DhIntersectionGuest, DhIntersectionHost, EcdhIntersectionHost, EcdhIntersectionGuest
from federatedml.statistic.intersect.match_id_process import MatchIDIntersect
from federatedml.transfer_variable.transfer_class.intersection_func_transfer_variable import \
    IntersectionFuncTransferVariable
from federatedml.util import consts, LOGGER, data_format_preprocess


class IntersectModelBase(ModelBase):
    def __init__(self):
        super().__init__()
        self.intersection_obj = None
        self.proc_obj = None
        # self.intersect_num = -1
        self.intersect_rate = -1
        self.unmatched_num = -1
        self.unmatched_rate = -1
        self.intersect_ids = None
        self.metric_name = "intersection"
        self.metric_namespace = "train"
        self.metric_type = "INTERSECTION"
        self.model_param_name = "IntersectModelParam"
        self.model_meta_name = "IntersectModelMeta"
        self.model_param = IntersectParam()
        self.use_match_id_process = False
        self.role = None
        self.intersect_method = None
        self.match_id_num = None
        self.match_id_intersect_num = -1
        self.recovered_num = -1

        self.guest_party_id = None
        self.host_party_id = None
        self.host_party_id_list = None

        self.transfer_variable = IntersectionFuncTransferVariable()

    def _init_model(self, params):
        self.model_param = params
        self.intersect_preprocess_params = params.intersect_preprocess_params

    def init_intersect_method(self):
        if self.model_param.cardinality_only:
            self.intersect_method = self.model_param.cardinality_method
        else:
            self.intersect_method = self.model_param.intersect_method
        LOGGER.info("Using {} intersection, role is {}".format(self.intersect_method, self.role))

        self.host_party_id_list = self.component_properties.host_party_idlist
        self.guest_party_id = self.component_properties.guest_partyid

        if self.role not in [consts.HOST, consts.GUEST]:
            raise ValueError("role {} is not support".format(self.role))

    def get_model_summary(self):
        return {"intersect_num": self.match_id_intersect_num, "intersect_rate": self.intersect_rate,
                "cardinality_only": self.intersection_obj.cardinality_only,
                "unique_id_num": self.match_id_num}

    def sync_use_match_id(self):
        raise NotImplementedError(f"Should not be called here.")

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
                    "'allow_info_share' is true, and 'info_owner' is {}, but can not get header in data, information sharing not done".format(
                        self.model_param.info_owner))
        else:
            self.intersect_ids = info_share.get(idx=0)
            self.intersect_ids.schema['header'] = [consts.SHARE_INFO_COL_NAME]
            LOGGER.info(
                "Get share information from {}, header:{}".format(self.model_param.info_owner, self.intersect_ids))

        return self.intersect_ids

    def __sync_join_id(self, data, intersect_data):
        LOGGER.debug(f"data count: {data.count()}")
        LOGGER.debug(f"intersect_data count: {intersect_data.count()}")

        if self.model_param.sample_id_generator == consts.GUEST:
            sync_join_id = self.transfer_variable.join_id_from_guest
        else:
            sync_join_id = self.transfer_variable.join_id_from_host
        if self.role == self.model_param.sample_id_generator:
            join_data = data.subtractByKey(intersect_data)
            # LOGGER.debug(f"join_data count: {join_data.count()}")
            if self.model_param.new_sample_id:
                if self.model_param.only_output_key:
                    join_data = join_data.map(lambda k, v: (fate_uuid(), None))
                    join_id = join_data
                else:
                    join_data = join_data.map(lambda k, v: (fate_uuid(), v))
                    join_id = join_data.mapValues(lambda v: None)
                sync_join_id.remote(join_id)

                result_data = intersect_data.union(join_data)
            else:
                join_id = join_data.map(lambda k, v: (k, None))
                result_data = data
                if self.model_param.only_output_key:
                    result_data = data.mapValues(lambda v: None)
                sync_join_id.remote(join_id)
        else:
            join_id = sync_join_id.get(idx=0)
            # LOGGER.debug(f"received join_id count: {join_id.count()}")
            join_data = join_id
            if not self.model_param.only_output_key:
                feature_shape = data.first()[1].features.shape[0]

                def _generate_nan_instance():
                    filler = np.empty((feature_shape,))
                    filler.fill(np.nan)
                    return filler

                join_data = join_id.mapValues(lambda v: Instance(features=_generate_nan_instance()))
            result_data = intersect_data.union(join_data)
        LOGGER.debug(f"result data count: {result_data.count()}")
        return result_data

    def callback(self):
        meta_info = {"intersect_method": self.intersect_method,
                     "join_method": self.model_param.join_method}
        if self.use_match_id_process:
            self.callback_metric(metric_name=self.metric_name,
                                 metric_namespace=self.metric_namespace,
                                 metric_data=[Metric("intersect_count", self.match_id_intersect_num),
                                              Metric("input_match_id_count", self.match_id_num),
                                              Metric("intersect_rate", self.intersect_rate),
                                              Metric("unmatched_count", self.unmatched_num),
                                              Metric("unmatched_rate", self.unmatched_rate),
                                              Metric("intersect_sample_id_count", self.recovered_num)])
        else:
            self.callback_metric(metric_name=self.metric_name,
                                 metric_namespace=self.metric_namespace,
                                 metric_data=[Metric("intersect_count", self.match_id_intersect_num),
                                              Metric("input_match_id_count", self.match_id_num),
                                              Metric("intersect_rate", self.intersect_rate),
                                              Metric("unmatched_count", self.unmatched_num),
                                              Metric("unmatched_rate", self.unmatched_rate)])
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=self.metric_name,
                                     metric_meta=MetricMeta(name=self.metric_name,
                                                            metric_type=self.metric_type,
                                                            extra_metas=meta_info)
                                     )

    def callback_cache_meta(self, intersect_meta):
        metric_name = f"{self.metric_name}_cache_meta"
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=metric_name,
                                     metric_meta=MetricMeta(name=f"{self.metric_name}_cache_meta",
                                                            metric_type=self.metric_type,
                                                            extra_metas=intersect_meta)
                                     )

    def fit(self, data):
        if self.component_properties.caches:
            LOGGER.info(f"Cache provided, will enter intersect online process.")
            return self.intersect_online_process(data, self.component_properties.caches)
        self.init_intersect_method()
        if data_overview.check_with_inst_id(data):
            self.use_match_id_process = True
            LOGGER.info(f"use match_id_process")
        self.sync_use_match_id()

        if self.use_match_id_process:
            if len(self.host_party_id_list) > 1 and self.model_param.sample_id_generator != consts.GUEST:
                raise ValueError("While multi-host, sample_id_generator should be guest.")
            if self.intersect_method == consts.RAW:
                if self.model_param.sample_id_generator != self.intersection_obj.join_role:
                    raise ValueError(f"When using raw intersect with match id process,"
                                     f"'join_role' should be same role as 'sample_id_generator'")
            else:
                if not self.model_param.sync_intersect_ids:
                    if self.model_param.sample_id_generator != consts.GUEST:
                        self.model_param.sample_id_generator = consts.GUEST
                        LOGGER.warning(f"when not sync_intersect_ids with match id process,"
                                       f"sample_id_generator is set to Guest")

            self.proc_obj = MatchIDIntersect(sample_id_generator=self.model_param.sample_id_generator, role=self.role)
            self.proc_obj.new_sample_id = self.model_param.new_sample_id
            if data_overview.check_with_inst_id(data) or self.model_param.with_sample_id:
                self.proc_obj.use_sample_id()
            match_data = self.proc_obj.recover(data=data)
            self.match_id_num = match_data.count()
            if self.intersection_obj.run_cache:
                self.cache_output = self.intersection_obj.generate_cache(match_data)
                intersect_meta = self.intersection_obj.get_intersect_method_meta()
                self.callback_cache_meta(intersect_meta)
                return
            if self.intersection_obj.cardinality_only:
                self.intersection_obj.run_cardinality(match_data)
            else:
                intersect_data = match_data
                if self.model_param.run_preprocess:
                    intersect_data = self.run_preprocess(match_data)
                self.intersect_ids = self.intersection_obj.run_intersect(intersect_data)
                if self.intersect_ids:
                    self.match_id_intersect_num = self.intersect_ids.count()
        else:
            if self.model_param.join_method == consts.LEFT_JOIN:
                raise ValueError(f"Only data with match_id may apply left_join method. Please check input data format")
            self.match_id_num = data.count()
            if self.intersection_obj.run_cache:
                self.cache_output = self.intersection_obj.generate_cache(data)
                intersect_meta = self.intersection_obj.get_intersect_method_meta()
                # LOGGER.debug(f"callback intersect meta is: {intersect_meta}")
                self.callback_cache_meta(intersect_meta)
                return
            if self.intersection_obj.cardinality_only:
                self.intersection_obj.run_cardinality(data)
            else:
                intersect_data = data
                if self.model_param.run_preprocess:
                    intersect_data = self.run_preprocess(data)
                self.intersect_ids = self.intersection_obj.run_intersect(intersect_data)
                if self.intersect_ids:
                    self.match_id_intersect_num = self.intersect_ids.count()

        if self.intersection_obj.cardinality_only:
            if self.intersection_obj.intersect_num is not None:
                # data_count = data.count()
                self.match_id_intersect_num = self.intersection_obj.intersect_num
                self.intersect_rate = self.match_id_intersect_num / self.match_id_num
                self.unmatched_num = self.match_id_num - self.match_id_intersect_num
                self.unmatched_rate = 1 - self.intersect_rate
            self.set_summary(self.get_model_summary())
            self.callback()
            return None

        if self.use_match_id_process:
            if self.model_param.sync_intersect_ids:
                self.intersect_ids = self.proc_obj.expand(self.intersect_ids, match_data=match_data)
            else:
                # self.intersect_ids = match_data
                self.intersect_ids = self.proc_obj.expand(self.intersect_ids,
                                                          match_data=match_data,
                                                          owner_only=True)
            if self.intersect_ids:
                self.recovered_num = self.intersect_ids.count()
            if self.model_param.only_output_key and self.intersect_ids:
                self.intersect_ids = self.intersect_ids.mapValues(lambda v: Instance(inst_id=v.inst_id))
                # self.intersect_ids.schema = {"match_id_name": data.schema["match_id_name"],
                #                             "sid": data.schema.get("sid")}
                self.intersect_ids.schema = data_format_preprocess.DataFormatPreProcess.clean_header(data.schema)

        LOGGER.info("Finish intersection")

        if self.intersect_ids:
            self.intersect_rate = self.match_id_intersect_num / self.match_id_num
            self.unmatched_num = self.match_id_num - self.match_id_intersect_num
            self.unmatched_rate = 1 - self.intersect_rate

        self.set_summary(self.get_model_summary())
        self.callback()

        result_data = self.intersect_ids
        if not self.use_match_id_process and result_data:
            if self.intersection_obj.only_output_key:
                # result_data.schema = {"sid": data.schema.get("sid")}
                result_data.schema = data_format_preprocess.DataFormatPreProcess.clean_header(data.schema)
                LOGGER.debug(f"non-match-id & only_output_key, add sid to schema")
            else:
                result_data = self.intersection_obj.get_value_from_data(result_data, data)
                LOGGER.debug(f"not only_output_key, restore instance value")

        if self.model_param.join_method == consts.LEFT_JOIN:
            result_data = self.__sync_join_id(data, self.intersect_ids)
            result_data.schema = self.intersect_ids.schema

        return result_data

    def check_consistency(self):
        pass

    def load_intersect_meta(self, intersect_meta):
        if self.model_param.intersect_method != intersect_meta.get("intersect_method"):
            raise ValueError(f"Current intersect method must match to cache record.")
        if self.model_param.intersect_method == consts.RSA:
            self.model_param.rsa_params.hash_method = intersect_meta["hash_method"]
            self.model_param.rsa_params.final_hash_method = intersect_meta["final_hash_method"]
            self.model_param.rsa_params.salt = intersect_meta["salt"]
            self.model_param.rsa_params.random_bit = intersect_meta["random_bit"]
        elif self.model_param.intersect_method == consts.DH:
            self.model_param.dh_params.hash_method = intersect_meta["hash_method"]
            self.model_param.dh_params.salt = intersect_meta["salt"]
        elif self.model_param.intersect_method == consts.ECDH:
            self.model_param.ecdh_params.hash_method = intersect_meta["hash_method"]
            self.model_param.ecdh_params.salt = intersect_meta["salt"]
            self.model_param.ecdh_params.curve = intersect_meta["curve"]
        else:
            raise ValueError(f"{self.model_param.intersect_method} does not support cache.")

    def make_filter_process(self, data_instances, hash_operator):
        raise NotImplementedError("This method should not be called here")

    def get_filter_process(self, data_instances, hash_operator):
        raise NotImplementedError("This method should not be called here")

    def run_preprocess(self, data_instances):
        preprocess_hash_operator = Hash(self.model_param.intersect_preprocess_params.preprocess_method, False)
        if self.role == self.model_param.intersect_preprocess_params.filter_owner:
            data = self.make_filter_process(data_instances, preprocess_hash_operator)
        else:
            LOGGER.debug(f"before preprocess, data count: {data_instances.count()}")
            data = self.get_filter_process(data_instances, preprocess_hash_operator)
            LOGGER.debug(f"after preprocess, data count: {data.count()}")
        return data

    def intersect_online_process(self, data_inst, caches):
        # LOGGER.debug(f"caches is: {caches}")
        cache_data, cache_meta = list(caches.values())[0]
        intersect_meta = list(cache_meta.values())[0]["intersect_meta"]
        # LOGGER.debug(f"intersect_meta is: {intersect_meta}")
        self.callback_cache_meta(intersect_meta)
        self.load_intersect_meta(intersect_meta)
        self.init_intersect_method()
        self.intersection_obj.load_intersect_key(cache_meta)

        if data_overview.check_with_inst_id(data_inst):
            self.use_match_id_process = True
            LOGGER.info(f"use match_id_process")
        self.sync_use_match_id()

        intersect_data = data_inst
        self.match_id_num = data_inst.count()
        if self.use_match_id_process:
            if len(self.host_party_id_list) > 1 and self.model_param.sample_id_generator != consts.GUEST:
                raise ValueError("While multi-host, sample_id_generator should be guest.")
            if self.intersect_method == consts.RAW:
                if self.model_param.sample_id_generator != self.intersection_obj.join_role:
                    raise ValueError(f"When using raw intersect with match id process,"
                                     f"'join_role' should be same role as 'sample_id_generator'")
            else:
                if not self.model_param.sync_intersect_ids:
                    if self.model_param.sample_id_generator != consts.GUEST:
                        self.model_param.sample_id_generator = consts.GUEST
                        LOGGER.warning(f"when not sync_intersect_ids with match id process,"
                                       f"sample_id_generator is set to Guest")

            proc_obj = MatchIDIntersect(sample_id_generator=self.model_param.sample_id_generator, role=self.role)
            proc_obj.new_sample_id = self.model_param.new_sample_id
            if data_overview.check_with_inst_id(data_inst) or self.model_param.with_sample_id:
                proc_obj.use_sample_id()
            match_data = proc_obj.recover(data=data_inst)
            intersect_data = match_data
            self.match_id_num = match_data.count()

        if self.role == consts.HOST:
            cache_id = cache_meta[str(self.guest_party_id)].get("cache_id")
            self.transfer_variable.cache_id.remote(cache_id, role=consts.GUEST, idx=0)
            guest_cache_id = self.transfer_variable.cache_id.get(role=consts.GUEST, idx=0)
            self.match_id_num = list(cache_data.values())[0].count()
            if guest_cache_id != cache_id:
                raise ValueError(f"cache_id check failed. cache_id from host & guest must match.")
        elif self.role == consts.GUEST:
            for i, party_id in enumerate(self.host_party_id_list):
                cache_id = cache_meta[str(party_id)].get("cache_id")
                self.transfer_variable.cache_id.remote(cache_id,
                                                       role=consts.HOST,
                                                       idx=i)
                host_cache_id = self.transfer_variable.cache_id.get(role=consts.HOST, idx=i)
                if host_cache_id != cache_id:
                    raise ValueError(f"cache_id check failed. cache_id from host & guest must match.")
        else:
            raise ValueError(f"Role {self.role} cannot run intersection transform.")

        self.intersect_ids = self.intersection_obj.run_cache_intersect(intersect_data, cache_data)
        self.match_id_intersect_num = self.intersect_ids.count()
        if self.use_match_id_process:
            if not self.model_param.sync_intersect_ids:
                self.intersect_ids = proc_obj.expand(self.intersect_ids,
                                                     match_data=match_data,
                                                     owner_only=True)
            else:
                self.intersect_ids = proc_obj.expand(self.intersect_ids, match_data=match_data)
            if self.intersect_ids:
                self.recovered_num = self.intersect_ids.count()
            if self.intersect_ids and self.model_param.only_output_key:
                self.intersect_ids = self.intersect_ids.mapValues(lambda v: Instance(inst_id=v.inst_id))
                # self.intersect_ids.schema = {"match_id_name": data_inst.schema["match_id_name"],
                #                             "sid": data_inst.schema.get("sid")}
                self.intersect_ids.schema = data_format_preprocess.DataFormatPreProcess.clean_header(data_inst.schema)

        LOGGER.info("Finish intersection")

        if self.intersect_ids:
            self.intersect_rate = self.match_id_intersect_num / self.match_id_num
            self.unmatched_num = self.match_id_num - self.match_id_intersect_num
            self.unmatched_rate = 1 - self.intersect_rate

        self.set_summary(self.get_model_summary())
        self.callback()

        result_data = self.intersect_ids
        if not self.use_match_id_process:
            if not self.intersection_obj.only_output_key and result_data:
                result_data = self.intersection_obj.get_value_from_data(result_data, data_inst)
                self.intersect_ids.schema = result_data.schema
                LOGGER.debug(f"not only_output_key, restore value called")
            if self.intersection_obj.only_output_key and result_data:
                # schema = {"sid": data_inst.schema.get("sid")}
                schema = data_format_preprocess.DataFormatPreProcess.clean_header(data_inst.schema)
                result_data = result_data.mapValues(lambda v: None)
                result_data.schema = schema
                self.intersect_ids.schema = schema

        if self.model_param.join_method == consts.LEFT_JOIN:
            result_data = self.__sync_join_id(data_inst, self.intersect_ids)
            result_data.schema = self.intersect_ids.schema

        return result_data


class IntersectHost(IntersectModelBase):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST

    def init_intersect_method(self):
        super().init_intersect_method()
        self.host_party_id = self.component_properties.local_partyid

        if self.intersect_method == consts.RSA:
            self.intersection_obj = RsaIntersectionHost()

        elif self.intersect_method == consts.RAW:
            self.intersection_obj = RawIntersectionHost()
            self.intersection_obj.tracker = self.tracker
            self.intersection_obj.task_version_id = self.task_version_id

        elif self.intersect_method == consts.DH:
            self.intersection_obj = DhIntersectionHost()

        elif self.intersect_method == consts.ECDH:
            self.intersection_obj = EcdhIntersectionHost()

        else:
            raise ValueError("intersect_method {} is not support yet".format(self.intersect_method))

        self.intersection_obj.host_party_id = self.host_party_id
        self.intersection_obj.guest_party_id = self.guest_party_id
        self.intersection_obj.host_party_id_list = self.host_party_id_list
        self.intersection_obj.load_params(self.model_param)
        self.model_param = self.intersection_obj.model_param

    def sync_use_match_id(self):
        self.transfer_variable.use_match_id.remote(self.use_match_id_process, role=consts.GUEST, idx=-1)
        LOGGER.info(f"sync use_match_id flag: {self.use_match_id_process} with Guest")

    def make_filter_process(self, data_instances, hash_operator):
        filter = self.intersection_obj.construct_filter(data_instances,
                                                        self.intersect_preprocess_params.false_positive_rate,
                                                        self.intersect_preprocess_params.hash_method,
                                                        self.intersect_preprocess_params.random_state,
                                                        hash_operator,
                                                        self.intersect_preprocess_params.preprocess_salt)
        self.transfer_variable.intersect_filter_from_host.remote(filter, role=consts.GUEST, idx=0)
        LOGGER.debug(f"filter sent to guest")
        return data_instances

    def get_filter_process(self, data_instances, hash_operator):
        filter = self.transfer_variable.intersect_filter_from_guest.get(idx=0)
        LOGGER.debug(f"got filter from guest")
        filtered_data = data_instances.filter(lambda k, v: filter.check(
            hash_operator.compute(k, suffix_salt=self.intersect_preprocess_params.preprocess_salt)))
        return filtered_data


class IntersectGuest(IntersectModelBase):
    def __init__(self):
        super().__init__()
        self.role = consts.GUEST

    def init_intersect_method(self):
        super().init_intersect_method()

        if self.intersect_method == consts.RSA:
            self.intersection_obj = RsaIntersectionGuest()

        elif self.intersect_method == consts.RAW:
            self.intersection_obj = RawIntersectionGuest()
            self.intersection_obj.tracker = self.tracker
            self.intersection_obj.task_version_id = self.task_version_id

        elif self.intersect_method == consts.DH:
            self.intersection_obj = DhIntersectionGuest()

        elif self.intersect_method == consts.ECDH:
            self.intersection_obj = EcdhIntersectionGuest()

        else:
            raise ValueError("intersect_method {} is not support yet".format(self.intersect_method))

        self.intersection_obj.guest_party_id = self.guest_party_id
        self.intersection_obj.host_party_id_list = self.host_party_id_list
        self.intersection_obj.load_params(self.model_param)

    def sync_use_match_id(self):
        host_use_match_id_flg = self.transfer_variable.use_match_id.get(idx=-1)
        LOGGER.info(f"received use_match_id flag from all hosts.")
        if any(flg != self.use_match_id_process for flg in host_use_match_id_flg):
            raise ValueError(f"Not all parties' input data have match_id, please check.")

    def make_filter_process(self, data_instances, hash_operator):
        filter = self.intersection_obj.construct_filter(data_instances,
                                                        self.intersect_preprocess_params.false_positive_rate,
                                                        self.intersect_preprocess_params.hash_method,
                                                        self.intersect_preprocess_params.random_state,
                                                        hash_operator,
                                                        self.intersect_preprocess_params.preprocess_salt)
        self.transfer_variable.intersect_filter_from_guest.remote(filter, role=consts.HOST, idx=-1)
        LOGGER.debug(f"filter sent to guest")

        return data_instances

    def get_filter_process(self, data_instances, hash_operator):
        filter_list = self.transfer_variable.intersect_filter_from_host.get(idx=-1)
        LOGGER.debug(f"got filter from all host")

        filtered_data_list = [
            data_instances.filter(
                lambda k,
                v: filter.check(
                    hash_operator.compute(
                        k,
                        suffix_salt=self.intersect_preprocess_params.preprocess_salt))) for filter in filter_list]
        filtered_data = self.intersection_obj.get_common_intersection(filtered_data_list, False)

        return filtered_data
