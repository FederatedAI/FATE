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
from arch.api.utils import log_utils
from federatedml.param.evaluation_param import EvaluateParam
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

    def _init_runtime_parameters(self, component_parameters):
        param_extracter = ParamExtract()
        param = param_extracter.parse_param_from_config(self.model_param, component_parameters)
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

    def get_metrics_param(self):
        return EvaluateParam(eval_type="binary",
                             pos_label=1)

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

    def set_cv_fold(self, cv_fold):
        self.cv_fold = cv_fold

    @staticmethod
    def extract_data(data: dict):
        LOGGER.debug("In extract_data, data input: {}".format(data))
        if len(data) == 0:
            return data
        if len(data) == 1:
            return list(data.values())[0]
        return data
