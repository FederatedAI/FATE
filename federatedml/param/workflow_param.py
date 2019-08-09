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
################################################################################
#
#
################################################################################

import copy

from arch.api.utils import log_utils
from federatedml.param.base_param import BaseParam
from federatedml.param.dataio_param import DataIOParam
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.param.predict_param import PredictParam

LOGGER = log_utils.getLogger()


class WorkFlowParam(BaseParam):
    """
    Define Workflow parameters used in federated ml.
    Parameters
    ----------
    method : str, 'train', 'predict', 'intersect' or 'cross_validation'. default: 'train'
        The working method of this task.
    train_input_table : str, default: None
        Required when method is 'train'. Specify the table name of input data in database.
    train_input_namespace : str, default: None
        Required when method is 'train'. Specify the namespace of input data in database.
    model_table : str, default: None
        Required when method is 'train', 'predict' or 'cross_validation'.
        Specify the table name to save or load model. When method is 'train' or 'cross_validation', this parameter
        is used to save model. When method is predict, it is used to load model.
    model_namespace : str, default: None
        Required when method is 'train', 'predict' or 'cross_validation'.
        Specify the namespace to save or load model. When method is 'train' or 'cross_validation', this parameter
        is used to save model. When method is predict, it is used to load model.
    predict_input_table : str, default: None
        Required when method is 'predict'. Specify the table name of predict input data.
    predict_input_namespace : str, default: None
        Required when method is 'predict'. Specify the namespace of predict input data in database.
    predict_result_partition : int, default: 1
        The partition number used for predict result.
    predict_output_table : str, default: None
        Required when method is 'predict'. Specify the table name of predict output data.
    predict_output_namespace : str, default: None
        Required when method is 'predict'. Specify the namespace of predict output data in database.
    evaluation_output_table : str, default: None
        Required when method is 'train', 'predict' or 'cross_validation'.
         Specify the table name of evalation output data.
    evaluation_output_namespace : str, default: None
        Required when method is 'train', 'predict' or 'cross_validation'.
         Specify the namespace of predict output data in database.
    data_input_table : str, defalut: None
        Required when method is 'cross_validation'. Specify the table name of input data.
    data_input_namespace : str, defalut: None
        Required when method is 'cross_validation'. Specify the namespace of input data.
    intersect_data_output_table : str, defalut: None
        Required when method is 'intersect'. Specify the table name of output data.
    intersect_data_output_namespace : str, defalut: None
        Required when method is 'intersect'. Specify the namespace of output data.
    do_cross_validation : Abandonded.
    work_mode: int, 0 or 1. default: 0
        Specify the work mode. 0 means standalone version, 1 represent for cluster version.
    n_splits: int, default: 5
        The number of fold used in KFold validation. It is required in 'cross_validation' only.
    need_intersect: bool, default: True
        Whether this task need to do intersect. No need to specify in Homo task.
    need_sample: bool, default: False
        Whether this task need to do feature selection or not.
    need_feature_selection: bool, default: False
        Whether this task need to do feature selection or not.
    """

    def __init__(self, method='train', train_input_table=None, train_input_namespace=None, model_table=None,
                 model_namespace=None, predict_input_table=None, predict_input_namespace=None,
                 predict_result_partition=1, predict_output_table=None, predict_output_namespace=None,
                 evaluation_output_table=None, evaluation_output_namespace=None,
                 data_input_table=None, data_input_namespace=None, intersect_data_output_table=None,
                 intersect_data_output_namespace=None, dataio_param=DataIOParam(), predict_param=PredictParam(),
                 evaluate_param=EvaluateParam(), do_cross_validation=False, work_mode=0,
                 n_splits=5, need_intersect=True, need_sample=False, need_feature_selection=False, need_scale=False,
                 one_vs_rest=False, need_one_hot=False):
        self.method = method
        self.train_input_table = train_input_table
        self.train_input_namespace = train_input_namespace
        self.model_table = model_table
        self.model_namespace = model_namespace
        self.predict_input_table = predict_input_table
        self.predict_input_namespace = predict_input_namespace
        self.predict_output_table = predict_output_table
        self.predict_output_namespace = predict_output_namespace
        self.predict_result_partition = predict_result_partition
        self.evaluation_output_table = evaluation_output_table
        self.evaluation_output_namespace = evaluation_output_namespace
        self.data_input_table = data_input_table
        self.data_input_namespace = data_input_namespace
        self.intersect_data_output_table = intersect_data_output_table
        self.intersect_data_output_namespace = intersect_data_output_namespace
        self.dataio_param = copy.deepcopy(dataio_param)
        self.do_cross_validation = do_cross_validation
        self.n_splits = n_splits
        self.work_mode = work_mode
        self.predict_param = copy.deepcopy(predict_param)
        self.evaluate_param = copy.deepcopy(evaluate_param)
        self.need_intersect = need_intersect
        self.need_sample = need_sample
        self.need_feature_selection = need_feature_selection
        self.need_scale = need_scale
        self.need_one_hot = need_one_hot
        self.one_vs_rest = one_vs_rest

    def check(self):

        descr = "workflow param's "

        self.method = self.check_and_change_lower(self.method,
                                                  ['train', 'predict', 'cross_validation',
                                                   'intersect', 'binning', 'feature_select',
                                                   'one_vs_rest_train', "one_vs_rest_predict"],
                                                  descr)

        if self.method in ['train', 'binning', 'feature_select']:
            if type(self.train_input_table).__name__ != "str":
                raise ValueError(
                    "workflow param's train_input_table {} not supported, should be str type".format(
                        self.train_input_table))

            if type(self.train_input_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's train_input_namespace {} not supported, should be str type".format(
                        self.train_input_namespace))

        if self.method in ["train", "predict", "cross_validation"]:
            if type(self.model_table).__name__ != "str":
                raise ValueError(
                    "workflow param's model_table {} not supported, should be str type".format(
                        self.model_table))

            if type(self.model_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's model_namespace {} not supported, should be str type".format(
                        self.model_namespace))

        if self.method == 'predict':
            if type(self.predict_input_table).__name__ != "str":
                raise ValueError(
                    "workflow param's predict_input_table {} not supported, should be str type".format(
                        self.predict_input_table))

            if type(self.predict_input_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's predict_input_namespace {} not supported, should be str type".format(
                        self.predict_input_namespace))

            if type(self.predict_output_table).__name__ != "str":
                raise ValueError(
                    "workflow param's predict_output_table {} not supported, should be str type".format(
                        self.predict_output_table))

            if type(self.predict_output_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's predict_output_namespace {} not supported, should be str type".format(
                        self.predict_output_namespace))

        if self.method in ["train", "predict", "cross_validation"]:
            if type(self.predict_result_partition).__name__ != "int":
                raise ValueError(
                    "workflow param's predict_result_partition {} not supported, should be int type".format(
                        self.predict_result_partition))

            if type(self.evaluation_output_table).__name__ != "str":
                raise ValueError(
                    "workflow param's evaluation_output_table {} not supported, should be str type".format(
                        self.evaluation_output_table))

            if type(self.evaluation_output_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's evaluation_output_namespace {} not supported, should be str type".format(
                        self.evaluation_output_namespace))

        if self.method == 'cross_validation':
            if type(self.data_input_table).__name__ != "str":
                raise ValueError(
                    "workflow param's data_input_table {} not supported, should be str type".format(
                        self.data_input_table))

            if type(self.data_input_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's data_input_namespace {} not supported, should be str type".format(
                        self.data_input_namespace))

            if type(self.n_splits).__name__ != "int":
                raise ValueError(
                    "workflow param's n_splits {} not supported, should be int type".format(
                        self.n_splits))
            elif self.n_splits <= 0:
                raise ValueError(
                    "workflow param's n_splits must be greater or equal to 1")

        if self.intersect_data_output_table is not None:
            if type(self.intersect_data_output_table).__name__ != "str":
                raise ValueError(
                    "workflow param's intersect_data_output_table {} not supported, should be str type".format(
                        self.intersect_data_output_table))

        if self.intersect_data_output_namespace is not None:
            if type(self.intersect_data_output_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's intersect_data_output_namespace {} not supported, should be str type".format(
                        self.intersect_data_output_namespace))

        DataIOParam.check(self.dataio_param)

        if type(self.work_mode).__name__ != "int":
            raise ValueError(
                "workflow param's work_mode {} not supported, should be int type".format(
                    self.work_mode))
        elif self.work_mode not in [0, 1]:
            raise ValueError(
                "workflow param's work_mode must be 0 (represent to standalone mode) or 1 (represent to cluster mode)")

        if self.method in ["train", "predict", "cross_validation"]:
            PredictParam.check(self.predict_param)
            EvaluateParam.check(self.evaluate_param)

        LOGGER.debug("Finish workerflow parameter check!")
        return True
