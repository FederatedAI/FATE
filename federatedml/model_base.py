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
from federatedml.util.param_extract import ParamExtract
from federatedml.param.evaluation_param import EvaluateParam
from arch.api.utils import log_utils

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
        self.need_run = True
        self.need_cv = False
        self.need_one_vs_rest = False
        self.tracker = None
        self.cv_fold = 0
        self.validation_freqs = None

    def _init_runtime_parameters(self, component_parameters):
        param_extracter = ParamExtract()

        param = param_extracter.parse_param_from_config(self.model_param, component_parameters)

        param.check()
        self._init_model(param)
        try:
            need_cv = param.cv_param.need_cv
        except AttributeError:
            need_cv = False
        self.need_cv = need_cv
        try:
            need_run = param.need_run
        except AttributeError:
            need_run = True
        self.need_run = need_run

        LOGGER.debug("need_run: {}, need_cv: {}".format(self.need_run, self.need_cv))

    def _init_model(self, model):
        pass

    def _load_model(self, model_dict):
        pass

    def _parse_need_run(self, model_dict, model_meta_name):
        meta_obj = list(model_dict.get('model').values())[0].get(model_meta_name)
        need_run = meta_obj.need_run
        self.need_run = need_run

    def _run_data(self, data_sets=None, stage=None):
        train_data = None
        eval_data = None
        data = None

        for data_key in data_sets:
            if data_sets[data_key].get("train_data", None):
                train_data = data_sets[data_key]["train_data"]

            if data_sets[data_key].get("eval_data", None):
                eval_data = data_sets[data_key]["eval_data"]

            if data_sets[data_key].get("data", None):
                data = data_sets[data_key]["data"]

        if not self.need_run:
            self.data_output = data
            return data

        if stage == 'cross_validation':
            LOGGER.info("Need cross validation.")
            self.cross_validation(train_data)

        elif train_data is not None:
            self.set_flowid('fit')
            self.fit(train_data, eval_data)
            self.set_flowid('predict')
            self.data_output = self.predict(train_data)

            if self.data_output:
                self.data_output = self.data_output.mapValues(lambda value: value + ["train"])

            if eval_data:
                self.set_flowid('validate')
                eval_data_output = self.predict(eval_data)

                if eval_data_output:
                    eval_data_output = eval_data_output.mapValues(lambda value: value + ["validation"])

                if self.data_output and eval_data_output:
                    self.data_output = self.data_output.union(eval_data_output)
                elif not self.data_output and eval_data_output:
                    self.data_output = eval_data_output

            self.set_predict_data_schema(self.data_output, train_data.schema)

        elif eval_data is not None:
            self.set_flowid('predict')
            self.data_output = self.predict(eval_data)

            if self.data_output:
                self.data_output = self.data_output.mapValues(lambda value: value + ["test"])

            self.set_predict_data_schema(self.data_output, eval_data.schema)

        else:
            if stage == "fit":
                self.set_flowid('fit')
                self.data_output = self.fit(data)
            else:
                self.set_flowid('transform')
                self.data_output = self.transform(data)

        if self.data_output:
            # LOGGER.debug("data is {}".format(self.data_output.first()[1].features))
            LOGGER.debug("In model base, data_output schema: {}".format(self.data_output.schema))

    def run(self, component_parameters=None, args=None):
        self._init_runtime_parameters(component_parameters)

        if self.need_cv:
            stage = 'cross_validation'
        elif "model" in args:
            self._load_model(args)
            stage = "transform"
        elif "isometric_model" in args:
            self._load_model(args)
            stage = "fit"
        else:
            stage = "fit"

        if args.get("data", None) is None:
            return

        self._run_data(args["data"], stage)

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

    def set_predict_data_schema(self, predict_data, schema):
        if predict_data is not None:
            predict_data.schema = {"header": ["label", "predict_result", "predict_score", "predict_detail", "type"],
                                   "sid_name": schema.get('sid_name')}

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
