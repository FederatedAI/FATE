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

import functools
import math
import random

from sklearn.utils import resample

from fate_arch.session import computing_session as session
from federatedml.model_base import Metric
from federatedml.model_base import MetricMeta
from federatedml.model_base import ModelBase
from federatedml.param.sample_param import SampleParam
from federatedml.transfer_variable.transfer_class.sample_transfer_variable import SampleTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util.schema_check import assert_schema_consistent
from fate_arch.common.base_utils import fate_uuid


class RandomSampler(object):
    """
    Random Sampling Method

    Parameters
    ----------
    fraction : None or float,  sampling ratio, default: 0.1

    random_state: int, RandomState instance or None, optional, default: None

    method: str, supported "upsample", "downsample" only in this version, default: "downsample"

    """

    def __init__(self, fraction=0.1, random_state=None, method="downsample"):
        self.fraction = fraction
        self.random_state = random_state
        self.method = method
        self.tracker = None
        self._summary_buf = {}

    def set_tracker(self, tracker):
        self.tracker = tracker

    def sample(self, data_inst, sample_ids=None):
        """
        Interface to call random sample method

        Parameters
        ----------
        data_inst : Table
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's parameters,
            otherwise, it will be sample transform process, which means use the samples_ids to generate data

        Returns
        -------
        new_data_inst: Table
            the output sample data, same format with input

        sample_ids: list, return only if sample_ids is None


        """

        if sample_ids is None:
            new_data_inst, sample_ids = self.__sample(data_inst)
            return new_data_inst, sample_ids
        else:
            new_data_inst = self.__sample(data_inst, sample_ids)
            return new_data_inst

    def __sample(self, data_inst, sample_ids=None):
        """
        Random sample method, a line's occur probability is decide by fraction
            support down sample and up sample
                if use down sample: should give a float ratio between [0, 1]
                otherwise: should give a float ratio larger than 1.0

        Parameters
        ----------
        data_inst : Table
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's parameters,
            otherwise, it will be sample transform process, which means use the samples_ids to generate data

        Returns
        -------
        new_data_inst: Table
            the output sample data, same format with input

        sample_ids: list, return only if sample_ids is None


        """
        LOGGER.info("start to run random sampling")

        return_sample_ids = False
        if self.method == "downsample":
            if sample_ids is None:
                return_sample_ids = True
                idset = [key for key, value in data_inst.mapValues(lambda val: None).collect()]
                if self.fraction < 0 or self.fraction > 1:
                    raise ValueError("sapmle fractions should be a numeric number between 0 and 1inclusive")

                sample_num = max(1, int(self.fraction * len(idset)))

                sample_ids = resample(idset,
                                      replace=False,
                                      n_samples=sample_num,
                                      random_state=self.random_state)

            sample_dtable = session.parallelize(zip(sample_ids, range(len(sample_ids))),
                                                include_key=True,
                                                partition=data_inst.partitions)
            new_data_inst = data_inst.join(sample_dtable, lambda v1, v2: v1)

            callback(self.tracker, "random", [Metric("count", new_data_inst.count())], summary_dict=self._summary_buf)

            if return_sample_ids:
                return new_data_inst, sample_ids
            else:
                return new_data_inst

        elif self.method == "upsample":
            data_set = list(data_inst.collect())
            idset = [key for (key, value) in data_set]
            id_maps = dict(zip(idset, range(len(idset))))

            if sample_ids is None:
                return_sample_ids = True
                if self.fraction <= 0:
                    raise ValueError("sample fractions should be a numeric number large than 0")

                sample_num = int(self.fraction * len(idset))
                sample_ids = resample(idset,
                                      replace=True,
                                      n_samples=sample_num,
                                      random_state=self.random_state)

            new_data = []
            for i in range(len(sample_ids)):
                index = id_maps[sample_ids[i]]
                new_data.append((i, data_set[index][1]))

            new_data_inst = session.parallelize(new_data,
                                                include_key=True,
                                                partition=data_inst.partitions)

            callback(self.tracker, "random", [Metric("count", new_data_inst.count())], summary_dict=self._summary_buf)

            if return_sample_ids:
                return new_data_inst, sample_ids
            else:
                return new_data_inst

        else:
            raise ValueError("random sampler not support method {} yet".format(self.method))

    def get_summary(self):
        return self._summary_buf


class StratifiedSampler(object):
    """
    Stratified Sampling Method

    Parameters
    ----------
    fractions : None or list of (category, sample ratio) tuple,
        sampling ratios of each category, default: None
        e.g.
        [(0, 0.5), (1, 0.1]) in down sample, [(1, 1.5), (0, 1.8)], where 0\1 are the the occurred category.

    random_state: int, RandomState instance or None, optional, default: None

    method: str, supported "upsample", "downsample", default: "downsample"

    """

    def __init__(self, fractions=None, random_state=None, method="downsample"):
        self.fractions = fractions
        self.label_mapping = {}
        self.labels = []
        if fractions:
            for (label, frac) in fractions:
                self.label_mapping[label] = len(self.labels)
                self.labels.append(label)

            # self.label_mapping = [label for (label, frac) in fractions]

        self.random_state = random_state
        self.method = method
        self.tracker = None
        self._summary_buf = {}

    def set_tracker(self, tracker):
        self.tracker = tracker

    def sample(self, data_inst, sample_ids=None):
        """
        Interface to call stratified sample method

        Parameters
        ----------
        data_inst : Table
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's key by sample parameters,
            otherwise, it will be sample transform process, which means use the samples_ids to generate data

        Returns
        -------
        new_data_inst: Table
            the output sample data, same format with input

        sample_ids: list, return only if sample_ids is None


        """

        if sample_ids is None:
            new_data_inst, sample_ids = self.__sample(data_inst)
            return new_data_inst, sample_ids
        else:
            new_data_inst = self.__sample(data_inst, sample_ids)
            return new_data_inst

    def __sample(self, data_inst, sample_ids=None):
        """
        Stratified sample method, a line's occur probability is decide by fractions
            Input should be Table, every line should be an instance object with label
            To use this method, a list of ratio should be give, and the list length
                equals to the number of distinct labels
            support down sample and up sample
                if use down sample: should give a list of (category, ratio), where ratio is between [0, 1]
                otherwise: should give a list (category, ratio), where the float ratio should no less than 1.0


        Parameters
        ----------
        data_inst : Table
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's parameters,
            otherwise, it will be sample transform process, which means use the samples_ids the generate data

        Returns
        -------
        new_data_inst: Table
            the output sample data, sample format with input

        sample_ids: list, return only if sample_ids is None


        """

        LOGGER.info("start to run stratified sampling")
        return_sample_ids = False
        if self.method == "downsample":
            if sample_ids is None:
                idset = [[] for i in range(len(self.fractions))]
                for label, fraction in self.fractions:
                    if fraction < 0 or fraction > 1:
                        raise ValueError("sapmle fractions should be a numeric number between 0 and 1inclusive")

                return_sample_ids = True
                for key, inst in data_inst.collect():
                    label = inst.label
                    if label not in self.label_mapping:
                        raise ValueError("label not specify sample rate! check it please")
                    idset[self.label_mapping[label]].append(key)

                sample_ids = []

                callback_sample_metrics = []
                callback_original_metrics = []

                for i in range(len(idset)):
                    label_name = self.labels[i]
                    callback_original_metrics.append(Metric(label_name, len(idset[i])))

                    if idset[i]:
                        sample_num = max(1, int(self.fractions[i][1] * len(idset[i])))

                        _sample_ids = resample(idset[i],
                                               replace=False,
                                               n_samples=sample_num,
                                               random_state=self.random_state)

                        sample_ids.extend(_sample_ids)

                        callback_sample_metrics.append(Metric(label_name, len(_sample_ids)))
                    else:
                        callback_sample_metrics.append(Metric(label_name, 0))

                random.shuffle(sample_ids)

                callback(
                    self.tracker,
                    "stratified",
                    callback_sample_metrics,
                    callback_original_metrics,
                    self._summary_buf)

            sample_dtable = session.parallelize(zip(sample_ids, range(len(sample_ids))),
                                                include_key=True,
                                                partition=data_inst.partitions)
            new_data_inst = data_inst.join(sample_dtable, lambda v1, v2: v1)

            if return_sample_ids:
                return new_data_inst, sample_ids
            else:
                return new_data_inst

        elif self.method == "upsample":
            data_set = list(data_inst.collect())
            ids = [key for (key, inst) in data_set]
            id_maps = dict(zip(ids, range(len(ids))))

            return_sample_ids = False

            if sample_ids is None:
                idset = [[] for i in range(len(self.fractions))]
                for label, fraction in self.fractions:
                    if fraction <= 0:
                        raise ValueError("sapmle fractions should be a numeric number greater than 0")

                for key, inst in data_set:
                    label = inst.label
                    if label not in self.label_mapping:
                        raise ValueError("label not specify sample rate! check it please")
                    idset[self.label_mapping[label]].append(key)

                return_sample_ids = True

                sample_ids = []
                callback_sample_metrics = []
                callback_original_metrics = []

                for i in range(len(idset)):
                    label_name = self.labels[i]
                    callback_original_metrics.append(Metric(label_name, len(idset[i])))

                    if idset[i]:
                        sample_num = max(1, int(self.fractions[i][1] * len(idset[i])))

                        _sample_ids = resample(idset[i],
                                               replace=True,
                                               n_samples=sample_num,
                                               random_state=self.random_state)

                        sample_ids.extend(_sample_ids)

                        callback_sample_metrics.append(Metric(label_name, len(_sample_ids)))
                    else:
                        callback_sample_metrics.append(Metric(label_name, 0))

                random.shuffle(sample_ids)

                callback(
                    self.tracker,
                    "stratified",
                    callback_sample_metrics,
                    callback_original_metrics,
                    self._summary_buf)

            new_data = []
            for i in range(len(sample_ids)):
                index = id_maps[sample_ids[i]]
                new_data.append((i, data_set[index][1]))

            new_data_inst = session.parallelize(new_data,
                                                include_key=True,
                                                partition=data_inst.partitions)

            if return_sample_ids:
                return new_data_inst, sample_ids
            else:
                return new_data_inst

        else:
            raise ValueError("Stratified sampler not support method {} yet".format(self.method))

    def get_summary(self):
        return self._summary_buf


class ExactSampler(object):
    """
    Exact Sampling Method

    Parameters
    ----------
    """

    def __init__(self):
        self.tracker = None
        self._summary_buf = {}

    def set_tracker(self, tracker):
        self.tracker = tracker

    def get_sample_ids(self, data_inst):
        original_sample_count = data_inst.count()
        non_zero_data_inst = data_inst.filter(lambda k, v: v.weight > consts.FLOAT_ZERO)
        non_zero_sample_count = data_inst.count()
        if original_sample_count != non_zero_sample_count:
            sample_diff = original_sample_count - non_zero_sample_count
            LOGGER.warning(f"{sample_diff} zero-weighted sample encountered, will be discarded in final result.")

        def __generate_new_ids(v):
            if v.inst_id is None:
                raise ValueError(f"To sample with `exact_by_weight` mode, instances must have match id."
                                 f"Please check.")
            new_key_num = math.ceil(v.weight)
            new_sample_id_list = [fate_uuid() for _ in range(new_key_num)]
            return new_sample_id_list

        sample_ids = non_zero_data_inst.mapValues(lambda v: __generate_new_ids(v))
        return sample_ids

    def sample(self, data_inst, sample_ids=None):
        """
        Interface to call stratified sample method

        Parameters
        ----------
        data_inst : Table
            The input data

        sample_ids : Table
            use the samples_ids to generate data

        Returns
        -------
        new_data_inst: Table
            the output sample data, same format with input

        """
        LOGGER.info("start to generate exact sampling result")
        new_data_inst = self.__sample(data_inst, sample_ids)
        return new_data_inst

    def __sample(self, data_inst, sample_ids):
        """
        Exact sample method, duplicate samples by corresponding weight:
            if weight <= 0, discard sample; if round(weight) == 1, keep one,
            else duplicate round(weight) copies of sample

        Parameters
        ----------
        data_inst : Table
            The input data

        sample_ids : Table
            use the samples_ids the generate data

        Returns
        -------
        new_data_inst: Table
            the output sample data, sample format with input

        """
        sample_ids_map = data_inst.join(sample_ids, lambda v, ids: (v, ids))

        def __sample_new_id(k, v_id_map):
            v, id_map = v_id_map
            return [(new_id, v) for new_id in id_map]

        new_data_inst = sample_ids_map.flatMap(functools.partial(__sample_new_id))
        data_count = new_data_inst.count()
        if data_count is None:
            data_count = 0
            LOGGER.warning(f"All data instances discarded. Please check weight.")
        callback(self.tracker, "exact_by_weight", [Metric("count", data_count)], summary_dict=self._summary_buf)

        return new_data_inst

    def get_summary(self):
        return self._summary_buf


class Sampler(ModelBase):
    """
    Sampling Object

    Parameters
    ----------
    sample_param : object, self-define sample parameters,
        define in federatedml.param.sample_param

    """

    def __init__(self):
        super(Sampler, self).__init__()
        self.task_type = None
        # self.task_role = None
        self.flowid = 0
        self.model_param = SampleParam()

    def _init_model(self, sample_param):
        if sample_param.mode == "random":
            self.sampler = RandomSampler(sample_param.fractions,
                                         sample_param.random_state,
                                         sample_param.method)
            self.sampler.set_tracker(self.tracker)

        elif sample_param.mode == "stratified":
            self.sampler = StratifiedSampler(sample_param.fractions,
                                             sample_param.random_state,
                                             sample_param.method)
            self.sampler.set_tracker(self.tracker)
        elif sample_param.mode == "exact_by_weight":
            self.sampler = ExactSampler()
            self.sampler.set_tracker(self.tracker)
        else:
            raise ValueError("{} sampler not support yet".format(sample_param.mde))

        self.task_type = sample_param.task_type

    def _init_role(self, component_parameters):
        self.task_role = component_parameters["local"]["role"]

    def sample(self, data_inst, sample_ids=None):
        """
        Entry to use sample method

        Parameters
        ----------
        data_inst : Table
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's parameters,
            otherwise, it will be sample transform process, which means use the samples_ids the generate data

        Returns
        -------
        sample_data: Table
            the output sample data, same format with input


        """
        ori_schema = data_inst.schema
        sample_data = self.sampler.sample(data_inst, sample_ids)
        self.set_summary(self.sampler.get_summary())

        try:
            if len(sample_data) == 2:
                sample_data[0].schema = ori_schema
        except BaseException:
            sample_data.schema = ori_schema

        return sample_data

    def set_flowid(self, flowid="samole"):
        self.flowid = flowid

    def sync_sample_ids(self, sample_ids):
        transfer_inst = SampleTransferVariable()

        transfer_inst.sample_ids.remote(sample_ids,
                                        role="host",
                                        suffix=(self.flowid,))

    def recv_sample_ids(self):
        transfer_inst = SampleTransferVariable()

        sample_ids = transfer_inst.sample_ids.get(idx=0,
                                                  suffix=(self.flowid,))
        return sample_ids

    def run_sample(self, data_inst, task_type, task_role):
        """
        Sample running entry

        Parameters
        ----------
        data_inst : Table
            The input data

        task_type : "homo" or "hetero"
            if task_type is "homo", it will sample standalone
            if task_type is "heterl": then sampling will be done in one side, after that
                the side sync the sample ids to another side to generated the same sample result

        task_role: "guest" or "host":
            only consider this parameter when task_type is "hetero"
            if task_role is "guest", it will firstly sample ids, and sync it to "host"
                to generate data instances with sample ids
            if task_role is "host": it will firstly get the sample ids result of "guest",
                then generate sample data by the receiving ids

        Returns
        -------
        sample_data_inst: Table
            the output sample data, same format with input

        """
        LOGGER.info("begin to run sampling process")

        if task_type not in [consts.HOMO, consts.HETERO]:
            raise ValueError("{} task type not support yet".format(task_type))

        if task_type == consts.HOMO:
            return self.sample(data_inst)[0]

        elif task_type == consts.HETERO:
            if task_role == consts.GUEST:
                if self.model_param.mode == "exact_by_weight":
                    LOGGER.info("start to run exact sampling")
                    sample_ids = self.sampler.get_sample_ids(data_inst)
                    self.sync_sample_ids(sample_ids)
                    sample_data_inst = self.sample(data_inst, sample_ids)

                else:
                    sample_data_inst, sample_ids = self.sample(data_inst)
                    self.sync_sample_ids(sample_ids)

            elif task_role == consts.HOST:
                sample_ids = self.recv_sample_ids()
                sample_data_inst = self.sample(data_inst, sample_ids)

            else:
                raise ValueError("{} role not support yet".format(task_role))

            return sample_data_inst

    @assert_schema_consistent
    def fit(self, data_inst):
        return self.run_sample(data_inst, self.task_type, self.role)

    def transform(self, data_inst):
        return self.run_sample(data_inst, self.task_type, self.role)

    def check_consistency(self):
        pass

    def save_data(self):
        return self.data_output


def callback(tracker, method, callback_metrics, other_metrics=None, summary_dict=None):
    LOGGER.debug("callback: method is {}".format(method))
    if method == "random":
        tracker.log_metric_data("sample_count",
                                "random",
                                callback_metrics)

        tracker.set_metric_meta("sample_count",
                                "random",
                                MetricMeta(name="sample_count",
                                           metric_type="SAMPLE_TEXT"))

        summary_dict["sample_count"] = callback_metrics[0].value

    elif method == "stratified":
        LOGGER.debug(
            "callback: name {}, namespace {}, metrics_data {}".format("sample_count", "stratified", callback_metrics))

        tracker.log_metric_data("sample_count",
                                "stratified",
                                callback_metrics)

        tracker.set_metric_meta("sample_count",
                                "stratified",
                                MetricMeta(name="sample_count",
                                           metric_type="SAMPLE_TABLE"))

        tracker.log_metric_data("original_count",
                                "stratified",
                                other_metrics)

        tracker.set_metric_meta("original_count",
                                "stratified",
                                MetricMeta(name="original_count",
                                           metric_type="SAMPLE_TABLE"))

        summary_dict["sample_count"] = {}
        for sample_metric in callback_metrics:
            summary_dict["sample_count"][sample_metric.key] = sample_metric.value

        summary_dict["original_count"] = {}
        for sample_metric in other_metrics:
            summary_dict["original_count"][sample_metric.key] = sample_metric.value

    else:
        LOGGER.debug(
            f"callback: metrics_data {callback_metrics}, summary dict: {summary_dict}")

        tracker.log_metric_data("sample_count",
                                "exact_by_weight",
                                callback_metrics)

        tracker.set_metric_meta("sample_count",
                                "exact_by_weight",
                                MetricMeta(name="sample_count",
                                           metric_type="SAMPLE_TEXT"))

        summary_dict["sample_count"] = callback_metrics[0].value
