#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import copy
import typing

import numpy as np
from fate_arch.common.base_utils import timestamp_to_date
from fate_arch.computing import is_table
from google.protobuf import json_format

from federatedml.param.evaluation_param import EvaluateParam
from federatedml.protobuf import deserialize_models
from federatedml.statistic.data_overview import header_alignment
from federatedml.util import LOGGER, abnormal_detection
from federatedml.util.io_check import assert_match_id_consistent
from federatedml.util.component_properties import ComponentProperties, RunningFuncs
from federatedml.callbacks.callback_list import CallbackList
from federatedml.feature.instance import Instance


def serialize_models(models):
    serialized_models: typing.Dict[str, typing.Tuple[str, bytes, dict]] = {}

    for model_name, buffer_object in models.items():
        serialized_string = buffer_object.SerializeToString()
        pb_name = type(buffer_object).__name__
        json_format_dict = json_format.MessageToDict(
            buffer_object, including_default_value_fields=True
        )

        serialized_models[model_name] = (
            pb_name,
            serialized_string,
            json_format_dict,
        )

    return serialized_models


class ComponentOutput:
    def __init__(self, data, models, cache: typing.List[tuple]) -> None:
        self._data = data
        if not isinstance(self._data, list):
            self._data = [data]

        self._models = models
        if self._models is None:
            self._models = {}

        self._cache = cache
        if not isinstance(self._cache, list):
            self._cache = [cache]

    @property
    def data(self) -> list:
        return self._data

    @property
    def model(self):
        return serialize_models(self._models)

    @property
    def cache(self):
        return self._cache


class MetricType:
    LOSS = "LOSS"


class Metric:
    def __init__(self, key, value: float, timestamp: float = None):
        self.key = key
        self.value = value
        self.timestamp = timestamp

    def to_dict(self):
        return dict(key=self.key, value=self.value, timestamp=self.timestamp)


class MetricMeta:
    def __init__(self, name: str, metric_type: MetricType, extra_metas: dict = None):
        self.name = name
        self.metric_type = metric_type
        self.metas = {}
        self.extra_metas = extra_metas

    def update_metas(self, metas: dict):
        self.metas.update(metas)

    def to_dict(self):
        return dict(
            name=self.name,
            metric_type=self.metric_type,
            metas=self.metas,
            extra_metas=self.extra_metas,
        )


class CallbacksVariable(object):
    def __init__(self):
        self.stop_training = False
        self.best_iteration = -1
        self.validation_summary = None


class WarpedTrackerClient:
    def __init__(self, tracker) -> None:
        self._tracker = tracker

    def log_metric_data(
        self, metric_namespace: str, metric_name: str, metrics: typing.List[Metric]
    ):
        return self._tracker.log_metric_data(
            metric_namespace=metric_namespace,
            metric_name=metric_name,
            metrics=[metric.to_dict() for metric in metrics],
        )

    def set_metric_meta(
        self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta
    ):
        return self._tracker.set_metric_meta(
            metric_namespace=metric_namespace,
            metric_name=metric_name,
            metric_meta=metric_meta.to_dict(),
        )

    def log_component_summary(self, summary_data: dict):
        return self._tracker.log_component_summary(summary_data=summary_data)


class ModelBase(object):
    component_name = None

    @classmethod
    def set_component_name(cls, name):
        cls.component_name = name

    @classmethod
    def get_component_name(cls):
        return cls.component_name

    def __init__(self):
        self.model_output = None
        self.mode = None
        self.role = None
        self.data_output = None
        self.cache_output = None
        self.model_param = None
        self.transfer_variable = None
        self.flowid = ""
        self.task_version_id = ""
        self.need_one_vs_rest = False
        self.callback_one_vs_rest = False
        self.checkpoint_manager = None
        self.cv_fold = 0
        self.validation_freqs = None
        self.component_properties = ComponentProperties()
        self._summary = dict()
        self._align_cache = dict()
        self._tracker = None
        self.step_name = "step_name"
        self.callback_list: CallbackList
        self.callback_variables = CallbacksVariable()

    @property
    def tracker(self) -> WarpedTrackerClient:
        if self._tracker is None:
            raise RuntimeError(f"use tracker before set")
        return self._tracker

    @tracker.setter
    def tracker(self, value):
        self._tracker = WarpedTrackerClient(value)

    @property
    def stop_training(self):
        return self.callback_variables.stop_training

    @property
    def need_cv(self):
        return self.component_properties.need_cv

    @property
    def need_run(self):
        return self.component_properties.need_run

    @need_run.setter
    def need_run(self, value: bool):
        self.component_properties.need_run = value

    def _init_model(self, model):
        pass

    def load_model(self, model_dict):
        pass

    def _parse_need_run(self, model_dict, model_meta_name):
        meta_obj = list(model_dict.get("model").values())[0].get(model_meta_name)
        need_run = meta_obj.need_run
        # self.need_run = need_run
        self.component_properties.need_run = need_run

    def run(self, cpn_input, retry: bool = True):
        self.task_version_id = cpn_input.task_version_id
        self.tracker = cpn_input.tracker
        self.checkpoint_manager = cpn_input.checkpoint_manager

        deserialize_models(cpn_input.models)

        # retry
        if (
            self._retry
            and self.checkpoint_manager is not None
            and self.checkpoint_manager.latest_checkpoint is not None
        ):
            self._retry(cpn_input=cpn_input)
        # normal
        else:
            self._run(cpn_input=cpn_input)

        return ComponentOutput(self.save_data(), self._export(), self.save_cache())

    def _export(self):
        # export model
        try:
            model = self._export_model()
            meta = self._export_meta()
            export_dict = {"Meta": meta, "Param": model}
        except NotImplementedError:
            export_dict = self.export_model()

            # export nothing, return
            if export_dict is None:
                return export_dict

            try:
                meta_name = [k for k in export_dict if k.endswith("Meta")][0]
            except BaseException:
                raise KeyError("Meta not found in export model")

            try:
                param_name = [k for k in export_dict if k.endswith("Param")][0]
            except BaseException:
                raise KeyError("Param not found in export model")

            meta = export_dict[meta_name]

        # set component name
        if hasattr(meta, "component"):
            meta.component = self.get_component_name()
        else:
            import warnings

            warnings.warn(f"{meta} should add `component` field")
        return export_dict

    def _export_meta(self):
        raise NotImplementedError("_export_meta not implemented")

    def _export_model(self):
        raise NotImplementedError("_export_model not implemented")

    def _run(self, cpn_input) -> None:
        # paramters
        self.model_param.update(cpn_input.parameters)
        self.model_param.check()
        self.component_properties.parse_component_param(
            cpn_input.roles, self.model_param
        )
        self.role = self.component_properties.role
        self.component_properties.parse_dsl_args(cpn_input.datasets, cpn_input.models)
        self.component_properties.parse_caches(cpn_input.caches)
        # init component, implemented by subclasses
        self._init_model(self.model_param)

        self.callback_list = CallbackList(self.role, self.mode, self)
        if hasattr(self.model_param, "callback_param"):
            callback_param = getattr(self.model_param, "callback_param")
            self.callback_list.init_callback_list(callback_param)

        running_funcs = self.component_properties.extract_running_rules(
            datasets=cpn_input.datasets, models=cpn_input.models, cpn=self
        )
        LOGGER.debug(f"running_funcs: {running_funcs.todo_func_list}")
        saved_result = []
        for func, params, save_result, use_previews in running_funcs:
            # for func, params in zip(todo_func_list, todo_func_params):
            if use_previews:
                if params:
                    real_param = [saved_result, params]
                else:
                    real_param = saved_result
                LOGGER.debug("func: {}".format(func))
                this_data_output = func(*real_param)
                saved_result = []
            else:
                this_data_output = func(*params)

            if save_result:
                saved_result.append(this_data_output)

        if len(saved_result) == 1:
            self.data_output = saved_result[0]
            # LOGGER.debug("One data: {}".format(self.data_output.first()[1].features))
        LOGGER.debug(
            "saved_result is : {}, data_output: {}".format(
                saved_result, self.data_output
            )
        )
        # self.check_consistency()
        self.save_summary()

    def _retry(self, cpn_input) -> None:
        self.model_param.update(cpn_input.parameters)
        self.model_param.check()
        self.component_properties.parse_component_param(
            cpn_input.roles, self.model_param
        )
        self.role = self.component_properties.role
        self.component_properties.parse_dsl_args(cpn_input.datasets, cpn_input.models)
        self.component_properties.parse_caches(cpn_input.caches)
        # init component, implemented by subclasses
        self._init_model(self.model_param)

        self.callback_list = CallbackList(self.role, self.mode, self)
        if hasattr(self.model_param, "callback_param"):
            callback_param = getattr(self.model_param, "callback_param")
            self.callback_list.init_callback_list(callback_param)

        (
            train_data,
            validate_data,
            test_data,
            data,
        ) = self.component_properties.extract_input_data(
            datasets=cpn_input.datasets, model=self
        )

        running_funcs = RunningFuncs()
        latest_checkpoint = self.get_latest_checkpoint()
        running_funcs.add_func(self.load_model, [latest_checkpoint])
        running_funcs = self.component_properties.warm_start_process(
            running_funcs, self, train_data, validate_data
        )
        LOGGER.debug(f"running_funcs: {running_funcs.todo_func_list}")
        self._execute_running_funcs(running_funcs)

    def _execute_running_funcs(self, running_funcs):
        saved_result = []
        for func, params, save_result, use_previews in running_funcs:
            # for func, params in zip(todo_func_list, todo_func_params):
            if use_previews:
                if params:
                    real_param = [saved_result, params]
                else:
                    real_param = saved_result
                LOGGER.debug("func: {}".format(func))
                detected_func = assert_match_id_consistent(func)
                this_data_output = detected_func(*real_param)
                saved_result = []
            else:
                detected_func = assert_match_id_consistent(func)
                this_data_output = detected_func(*params)

            if save_result:
                saved_result.append(this_data_output)

        if len(saved_result) == 1:
            self.data_output = saved_result[0]
        LOGGER.debug(
            "saved_result is : {}, data_output: {}".format(
                saved_result, self.data_output
            )
        )
        self.save_summary()

    def export_serialized_models(self):
        return serialize_models(self.export_model())

    def get_metrics_param(self):
        return EvaluateParam(eval_type="binary", pos_label=1)

    def check_consistency(self):
        if not is_table(self.data_output):
            return
        if (
            self.component_properties.input_data_count
            + self.component_properties.input_eval_data_count
            != self.data_output.count()
            and self.component_properties.input_data_count
            != self.component_properties.input_eval_data_count
        ):
            raise ValueError("Input data count does not match with output data count")

    def predict(self, data_inst):
        pass

    def fit(self, *args):
        pass

    def transform(self, data_inst):
        pass

    def cross_validation(self, data_inst):
        pass

    def stepwise(self, data_inst):
        pass

    def one_vs_rest_fit(self, train_data=None):
        pass

    def one_vs_rest_predict(self, train_data):
        pass

    def init_validation_strategy(self, train_data=None, validate_data=None):
        pass

    def save_data(self):
        return self.data_output

    def export_model(self):
        return self.model_output

    def save_cache(self):
        return self.cache_output

    def set_flowid(self, flowid):
        # self.flowid = '.'.join([self.task_version_id, str(flowid)])
        self.flowid = flowid
        self.set_transfer_variable()

    def set_transfer_variable(self):
        if self.transfer_variable is not None:
            LOGGER.debug(
                "set flowid to transfer_variable, flowid: {}".format(self.flowid)
            )
            self.transfer_variable.set_flowid(self.flowid)

    def set_task_version_id(self, task_version_id):
        """task_version_id: jobid + component_name, reserved variable"""
        self.task_version_id = task_version_id

    def get_metric_name(self, name_prefix):
        if not self.need_cv:
            return name_prefix

        return "_".join(map(str, [name_prefix, self.flowid]))

    def set_tracker(self, tracker):
        self._tracker = tracker

    def set_checkpoint_manager(self, checkpoint_manager):
        checkpoint_manager.load_checkpoints_from_disk()
        self.checkpoint_manager = checkpoint_manager

    @staticmethod
    def set_predict_data_schema(predict_datas, schemas):
        if predict_datas is None:
            return predict_datas
        if isinstance(predict_datas, list):
            predict_data = predict_datas[0]
            schema = schemas[0]
        else:
            predict_data = predict_datas
            schema = schemas
        if predict_data is not None:
            predict_data.schema = {
                "header": [
                    "label",
                    "predict_result",
                    "predict_score",
                    "predict_detail",
                    "type",
                ],
                "sid_name": schema.get("sid_name"),
                "content_type": "predict_result",
            }
        return predict_data

    @staticmethod
    def predict_score_to_output(
        data_instances, predict_score, classes=None, threshold=0.5
    ):
        """
        Get predict result output
        Parameters
        ----------
        data_instances: table, data used for prediction
        predict_score: table, probability scores
        classes: list or None, all classes/label names
        threshold: float, predict threshold, used for binary label

        Returns
        -------
        Table, predict result
        """

        # regression
        if classes is None:
            predict_result = data_instances.join(
                predict_score, lambda d, pred: [d.label, pred, pred, {"label": pred}]
            )
        # binary
        elif isinstance(classes, list) and len(classes) == 2:
            class_neg, class_pos = classes[0], classes[1]
            pred_label = predict_score.mapValues(
                lambda x: class_pos if x > threshold else class_neg
            )
            predict_result = data_instances.mapValues(lambda x: x.label)
            predict_result = predict_result.join(predict_score, lambda x, y: (x, y))
            class_neg_name, class_pos_name = str(class_neg), str(class_pos)
            predict_result = predict_result.join(
                pred_label,
                lambda x, y: [
                    x[0],
                    y,
                    x[1],
                    {class_neg_name: (1 - x[1]), class_pos_name: x[1]},
                ],
            )

        # multi-label: input = array of predicted score of all labels
        elif isinstance(classes, list) and len(classes) > 2:
            # pred_label = predict_score.mapValues(lambda x: classes[x.index(max(x))])
            classes = [str(val) for val in classes]
            predict_result = data_instances.mapValues(lambda x: x.label)
            predict_result = predict_result.join(
                predict_score,
                lambda x, y: [
                    x,
                    int(classes[np.argmax(y)]),
                    float(np.max(y)),
                    dict(zip(classes, list(y))),
                ],
            )
        else:
            raise ValueError(
                f"Model's classes type is {type(classes)}, classes must be None or list of length no less than 2."
            )

        def _transfer(instance, pred_res):
            return Instance(features=pred_res, inst_id=instance.inst_id)

        predict_result = data_instances.join(predict_result, _transfer)

        return predict_result

    def callback_meta(self, metric_name, metric_namespace, metric_meta: MetricMeta):
        if self.need_cv:
            metric_name = ".".join([metric_name, str(self.cv_fold)])
            flow_id_list = self.flowid.split(".")
            LOGGER.debug(
                "Need cv, change callback_meta, flow_id_list: {}".format(flow_id_list)
            )
            if len(flow_id_list) > 1:
                curve_name = ".".join(flow_id_list[1:])
                metric_meta.update_metas({"curve_name": curve_name})
        else:
            metric_meta.update_metas({"curve_name": metric_name})

        self.tracker.set_metric_meta(
            metric_name=metric_name,
            metric_namespace=metric_namespace,
            metric_meta=metric_meta,
        )

    def callback_metric(
        self, metric_name, metric_namespace, metric_data: typing.List[Metric]
    ):
        if self.need_cv:
            metric_name = ".".join([metric_name, str(self.cv_fold)])

        self.tracker.log_metric_data(
            metric_name=metric_name,
            metric_namespace=metric_namespace,
            metrics=metric_data,
        )

    def callback_warm_start_init_iter(self, iter_num):
        metric_meta = MetricMeta(
            name="train",
            metric_type="init_iter",
            extra_metas={
                "unit_name": "iters",
            },
        )

        self.callback_meta(
            metric_name="init_iter", metric_namespace="train", metric_meta=metric_meta
        )
        self.callback_metric(
            metric_name="init_iter",
            metric_namespace="train",
            metric_data=[Metric("init_iter", iter_num)],
        )

    def get_latest_checkpoint(self):
        return self.checkpoint_manager.latest_checkpoint.read()

    def save_summary(self):
        self.tracker.log_component_summary(summary_data=self.summary())

    def set_cv_fold(self, cv_fold):
        self.cv_fold = cv_fold

    def summary(self):
        return copy.deepcopy(self._summary)

    def set_summary(self, new_summary):
        """
        Model summary setter
        Parameters
        ----------
        new_summary: dict, summary to replace the original one

        Returns
        -------

        """

        if not isinstance(new_summary, dict):
            raise ValueError(
                f"summary should be of dict type, received {type(new_summary)} instead."
            )
        self._summary = copy.deepcopy(new_summary)

    def add_summary(self, new_key, new_value):
        """
        Add key:value pair to model summary
        Parameters
        ----------
        new_key: str
        new_value: object

        Returns
        -------

        """

        original_value = self._summary.get(new_key, None)
        if original_value is not None:
            LOGGER.warning(
                f"{new_key} already exists in model summary."
                f"Corresponding value {original_value} will be replaced by {new_value}"
            )
        self._summary[new_key] = new_value
        # LOGGER.debug(f"{new_key}: {new_value} added to summary.")

    def merge_summary(self, new_content, suffix=None, suffix_sep="_"):
        """
        Merge new content into model summary
        Parameters
        ----------
        new_content: dict, content to be merged into summary
        suffix: str or None, suffix used to create new key if any key in new_content already exixts in model summary
        suffix_sep: string, default '_', suffix separator used to create new key

        Returns
        -------

        """

        if not isinstance(new_content, dict):
            raise ValueError(
                f"To merge new content into model summary, "
                f"value must be of dict type, received {type(new_content)} instead."
            )
        new_summary = self.summary()
        keyset = new_summary.keys() | new_content.keys()
        for key in keyset:
            if key in new_summary and key in new_content:
                if suffix is not None:
                    new_key = f"{key}{suffix_sep}{suffix}"
                else:
                    new_key = key
                new_value = new_content.get(key)
                new_summary[new_key] = new_value
            elif key in new_content:
                new_summary[key] = new_content.get(key)
            else:
                pass
        self.set_summary(new_summary)

    @staticmethod
    def extract_data(data: dict):
        LOGGER.debug("In extract_data, data input: {}".format(data))
        if len(data) == 0:
            return data
        if len(data) == 1:
            return list(data.values())[0]
        return data

    @staticmethod
    def check_schema_content(schema):
        """
        check for repeated header & illegal/non-printable chars except for space
        allow non-ascii chars
        :param schema: dict
        :return:
        """
        abnormal_detection.check_legal_schema(schema)

    def align_data_header(self, data_instances, pre_header):
        """
        align features of given data, raise error if value in given schema not found
        :param data_instances: data table
        :param pre_header: list, header of model
        :return: dtable, aligned data
        """
        result_data = self._align_cache.get(id(data_instances))
        if result_data is None:
            result_data = header_alignment(
                data_instances=data_instances, pre_header=pre_header
            )
            self._align_cache[id(data_instances)] = result_data
        return result_data

    @staticmethod
    def pass_data(data):
        if isinstance(data, dict) and len(data) >= 1:
            data = list(data.values())[0]
        return data

    def obtain_data(self, data_list):
        if isinstance(data_list, list):
            return data_list[0]
        return data_list
