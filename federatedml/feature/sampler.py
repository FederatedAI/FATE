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

from arch.api import session
from arch.api.utils import log_utils
from sklearn.utils import resample
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.param.sample_param import SampleParam
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.sample_transfer_variable import SampleTransferVariable
from federatedml.model_base import ModelBase
import random

LOGGER = log_utils.getLogger()


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

    def set_tracker(self, tracker):
        self.tracker = tracker

    def sample(self, data_inst, sample_ids=None):
        """
        Interface to call random sample method

        Parameters
        ----------
        data_inst : DTable
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's parameters,
            otherwise, it will be sample transform process, which means use the samples_ids to generate data

        Returns
        -------
        new_data_inst: DTable
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
        data_inst : DTable
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's parameters,
            otherwise, it will be sample transform process, which means use the samples_ids to generate data

        Returns
        -------
        new_data_inst: DTable
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
                                                partition=data_inst._partitions)
            new_data_inst = data_inst.join(sample_dtable, lambda v1, v2: v1)

            callback(self.tracker, "random", [Metric("count", new_data_inst.count())])

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
                    raise ValueError("sapmle fractions should be a numeric number large than 0")

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
                                                partition=data_inst._partitions)

            callback(self.tracker, "random", [Metric("count", new_data_inst.count())])

            if return_sample_ids:
                return new_data_inst, sample_ids
            else:
                return new_data_inst

        else:
            raise ValueError("random sampler not support method {} yet".format(self.method))


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

    method: str, supported "upsample", "downsample" only in this version, default: "downsample"

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

    def set_tracker(self, tracker):
        self.tracker = tracker

    def sample(self, data_inst, sample_ids=None):
        """
        Interface to call stratified sample method

        Parameters
        ----------
        data_inst : DTable
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's key by sample parameters,
            otherwise, it will be sample transform process, which means use the samples_ids to generate data

        Returns
        -------
        new_data_inst: DTable
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
            Input should be DTable, every line should be an instance object with label
            To use this method, a list of ratio should be give, and the list length
                equals to the number of distinct labels
            support down sample and up sample
                if use down sample: should give a list of (category, ratio), where ratio is between [0, 1]
                otherwise: should give a list (category, ratio), where the float ratio should no less than 1.0


        Parameters
        ----------
        data_inst : DTable
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's parameters,
            otherwise, it will be sample transform process, which means use the samples_ids the generate data

        Returns
        -------
        new_data_inst: DTable
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

                callback(self.tracker, "stratified", callback_sample_metrics, callback_original_metrics)

            sample_dtable = session.parallelize(zip(sample_ids, range(len(sample_ids))),
                                                include_key=True,
                                                partition=data_inst._partitions)
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

                callback(self.tracker, "stratified", callback_sample_metrics, callback_original_metrics)

            new_data = []
            for i in range(len(sample_ids)):
                index = id_maps[sample_ids[i]]
                new_data.append((i, data_set[index][1]))

            new_data_inst = session.parallelize(new_data,
                                                include_key=True,
                                                partition=data_inst._partitions)

            if return_sample_ids:
                return new_data_inst, sample_ids
            else:
                return new_data_inst

        else:
            raise ValueError("Stratified sampler not support method {} yet".format(self.method))


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
        data_inst : DTable
            The input data

        sample_ids : None or list
            if None, will sample data from the class instance's parameters,
            otherwise, it will be sample transform process, which means use the samples_ids the generate data

        Returns
        -------
        sample_data: DTable
            the output sample data, same format with input


        """
        ori_schema = data_inst.schema
        sample_data = self.sampler.sample(data_inst, sample_ids)

        try:
            if len(sample_data) == 2:
                sample_data[0].schema = ori_schema
        except:
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
        data_inst : DTable
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
        sample_data_inst: DTable
            the output sample data, same format with input

        """
        LOGGER.info("begin to run sampling process")

        if task_type not in [consts.HOMO, consts.HETERO]:
            raise ValueError("{} task type not support yet".format(task_type))

        if task_type == consts.HOMO:
            return self.sample(data_inst)[0]

        elif task_type == consts.HETERO:
            if task_role == consts.GUEST:
                sample_data_inst, sample_ids = self.sample(data_inst)
                self.sync_sample_ids(sample_ids)

            elif task_role == consts.HOST:
                sample_ids = self.recv_sample_ids()
                sample_data_inst = self.sample(data_inst, sample_ids)

            else:
                raise ValueError("{} role not support yet".format(task_role))

            return sample_data_inst

    def fit(self, data_inst):
        return self.run_sample(data_inst, self.task_type, self.role)

    def transform(self, data_inst):
        return self.run_sample(data_inst, self.task_type, self.role)

    """
    def run(self, component_parameters, args=None):
        self._init_runtime_parameters(component_parameters)
        self._init_role(component_parameters)
        stage = "fit"
        if args.get("data", None) is None:
            return

        self._run_data(args["data"], stage)
    """

    def save_data(self):
        return self.data_output


def callback(tracker, method, callback_metrics, other_metrics=None):
    LOGGER.debug("callback: method is {}".format(method))
    if method == "random":
        tracker.log_metric_data("sample_count",
                                "random",
                                callback_metrics)

        tracker.set_metric_meta("sample_count",
                                "random",
                                MetricMeta(name="sample_count",
                                           metric_type="SAMPLE_TEXT"))

    else:
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
