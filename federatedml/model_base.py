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

from arch.api.utils import log_utils
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.statistic.data_overview import header_alignment, check_legal_schema
from federatedml.util.component_properties import ComponentProperties
from federatedml.util.param_extract import ParamExtract

LOGGER = log_utils.getLogger()


class ModelBase(object):
    def __init__(self):
        self.model_output = None
        self.mode = None
        self.role = None
        self.data_output = None
        self.model_param = None
        self.transfer_variable = None
        self.flowid = ''
        self.taskid = ''
        self.need_one_vs_rest = False
        self.tracker = None
        self.cv_fold = 0
        self.validation_freqs = None
        self.component_properties = ComponentProperties()
        self._summary = dict()

    def _init_runtime_parameters(self, component_parameters):
        param_extractor = ParamExtract()
        param = param_extractor.parse_param_from_config(self.model_param, component_parameters)
        param.check()
        self.role = self.component_properties.parse_component_param(component_parameters, param).role
        self._init_model(param)
        return param

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
        meta_obj = list(model_dict.get('model').values())[0].get(model_meta_name)
        need_run = meta_obj.need_run
        # self.need_run = need_run
        self.component_properties.need_run = need_run

    def run(self, component_parameters=None, args=None):
        self._init_runtime_parameters(component_parameters)
        self.component_properties.parse_dsl_args(args)

        running_funcs = self.component_properties.extract_running_rules(args, self)
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
        LOGGER.debug("saved_result is : {}, data_output: {}".format(saved_result, self.data_output))
        self.check_consistency()
        # self.save_summary()

    def get_metrics_param(self):
        return EvaluateParam(eval_type="binary",
                             pos_label=1)

    def check_consistency(self):
        if not type(self.data_output) in ["DTable", "RDDTable"]:
            return
        if self.component_properties.input_data_count + self.component_properties.input_eval_data_count != \
                self.data_output.count():
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

    def set_flowid(self, flowid):
        # self.flowid = '.'.join([self.taskid, str(flowid)])
        self.flowid = flowid
        self.set_transfer_variable()

    def set_transfer_variable(self):
        if self.transfer_variable is not None:
            LOGGER.debug("set flowid to transfer_variable, flowid: {}".format(self.flowid))
            self.transfer_variable.set_flowid(self.flowid)

    def set_taskid(self, taskid):
        """ taskid: jobid + component_name, reserved variable """
        self.taskid = taskid

    def get_metric_name(self, name_prefix):
        if not self.need_cv:
            return name_prefix

        return '_'.join(map(str, [name_prefix, self.flowid]))

    def set_tracker(self, tracker):
        self.tracker = tracker

    def set_predict_data_schema(self, predict_datas, schemas):
        if predict_datas is None:
            return predict_datas
        if isinstance(predict_datas, list):
            predict_data = predict_datas[0]
            schema = schemas[0]
        else:
            predict_data = predict_datas
            schema = schemas
        if predict_data is not None:
            predict_data.schema = {"header": ["label", "predict_result", "predict_score", "predict_detail", "type"],
                                   "sid_name": schema.get('sid_name')}
        return predict_data

    def predict_score_to_output(self, data_instances, predict_score, classes=None, threshold=0.5):
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
            predict_result = data_instances.join(predict_score, lambda d, pred: [d.label, pred, pred, {"label": pred}])
        # binary
        elif isinstance(classes, list) and len(classes) == 2:
            class_neg, class_pos = classes[0], classes[1]
            pred_label = predict_score.mapValues(lambda x: class_pos if x > threshold else class_neg)
            predict_result = data_instances.mapValues(lambda x: x.label)
            predict_result = predict_result.join(predict_score, lambda x, y: (x, y))
            class_neg_name, class_pos_name = str(class_neg), str(class_pos)
            predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1],
                                                                           {class_neg_name: (1 - x[1]),
                                                                            class_pos_name: x[1]}])

        # multi-label: input = array of predicted score of all labels
        elif isinstance(classes, list) and len(classes) > 2:
            # pred_label = predict_score.mapValues(lambda x: classes[x.index(max(x))])
            classes = [str(val) for val in classes]
            predict_result = data_instances.mapValues(lambda x: x.label)
            predict_result = predict_result.join(predict_score, lambda x, y: [x, int(classes[y.argmax()]),
                                                                              y.max(), dict(zip(classes, list(y)))])
        else:
            raise ValueError(f"Model's classes type is {type(classes)}, classes must be None or list.")

        return predict_result

    def callback_meta(self, metric_name, metric_namespace, metric_meta):
        if self.need_cv:
            metric_name = '.'.join([metric_name, str(self.cv_fold)])
            flow_id_list = self.flowid.split('.')
            LOGGER.debug("Need cv, change callback_meta, flow_id_list: {}".format(flow_id_list))
            if len(flow_id_list) > 1:
                curve_name = '.'.join(flow_id_list[1:])
                metric_meta.update_metas({'curve_name': curve_name})
        else:
            metric_meta.update_metas({'curve_name': metric_name})

        self.tracker.set_metric_meta(metric_name=metric_name,
                                     metric_namespace=metric_namespace,
                                     metric_meta=metric_meta)

    def callback_metric(self, metric_name, metric_namespace, metric_data):
        if self.need_cv:
            metric_name = '.'.join([metric_name, str(self.cv_fold)])

        self.tracker.log_metric_data(metric_name=metric_name,
                                     metric_namespace=metric_namespace,
                                     metrics=metric_data)

    def save_summary(self):
        self.tracker.save_component_summary(summary_data=self.summary())

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
            raise ValueError(f"summary should be of dict type, received {type(new_summary)} instead.")
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
            LOGGER.warning(f"{new_key} already exists in model summary."
                           f"Corresponding value {original_value} will be replaced by {new_value}")
        self._summary[new_key] = new_value
        LOGGER.debug(f"{new_key}: {new_value} added to summary.")

    def merge_summary(self, new_content, suffix=None, suffix_sep='_'):
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
            raise ValueError(f"To merge new content into model summary, "
                             f"value must be of dict type, received {type(new_content)} instead.")
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
        check_legal_schema(schema)

    @staticmethod
    def align_data_header(data_instances, pre_header):
        """
        align features of given data, raise error if value in given schema not found
        :param data_instances: data table
        :param pre_header: list, header of model
        :return: dtable, aligned data
        """
        result_data = header_alignment(data_instances=data_instances, pre_header=pre_header)
        return result_data

    @staticmethod
    def pass_data(data):
        if isinstance(data, dict) and len(data) >= 1:
            data = list(data.values())[0]
        return data

    @staticmethod
    def obtain_data(data_list):
        if isinstance(data_list, list):
            return data_list[0]
        return data_list
