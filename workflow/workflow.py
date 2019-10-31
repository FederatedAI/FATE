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

import argparse
import json
import os

import numpy as np

from arch.api import session
from arch.api import federation
from arch.api.model_manager import manager as model_manager
from federatedml.protobuf.generated import pipeline_pb2
from arch.api.utils import log_utils
from federatedml.feature.hetero_feature_binning.hetero_binning_guest import HeteroFeatureBinningGuest
from federatedml.feature.hetero_feature_binning.hetero_binning_host import HeteroFeatureBinningHost
from federatedml.feature.hetero_feature_selection.feature_selection_guest import HeteroFeatureSelectionGuest
from federatedml.feature.hetero_feature_selection.feature_selection_host import HeteroFeatureSelectionHost
from federatedml.feature.one_hot_encoder import OneHotEncoder
from federatedml.feature.sampler import Sampler
from federatedml.feature.scaler import Scaler
from federatedml.model_selection import KFold
from federatedml.param import IntersectParam
from federatedml.param import WorkFlowParam
from federatedml.param import param as param_generator
from federatedml.param.param import OneVsRestParam
from federatedml.param.param import SampleParam
from federatedml.param.param import ScaleParam
from federatedml.statistic.intersect import RawIntersectionHost, RawIntersectionGuest
from federatedml.util import ParamExtract, DenseFeatureReader, SparseFeatureReader
from federatedml.util import WorkFlowParamChecker
from federatedml.util import consts
from federatedml.util import param_checker
from federatedml.util.data_io import SparseTagReader
from federatedml.util.param_checker import AllChecker
from federatedml.util.transfer import HeteroWorkFlowTransferVariable
from workflow import status_tracer_decorator

from federatedml.one_vs_rest.one_vs_rest import OneVsRest

LOGGER = log_utils.getLogger()


class WorkFlow(object):
    def __init__(self):
        # self._initialize(config_path)
        self.model = None
        self.role = None
        self.job_id = None
        self.mode = None
        self.workflow_param = None
        self.intersection = None
        self.pipeline = None

    def _initialize(self, config_path):
        LOGGER.debug("Get in base workflow initialize")
        self._initialize_role_and_mode()
        self._initialize_model(config_path)
        self._initialize_workflow_param(config_path)

    def _initialize_role_and_mode(self):
        self.role = consts.GUEST
        self.mode = consts.HETERO

    def _initialize_intersect(self, config):
        raise NotImplementedError("method init must be define")

    def _initialize_model(self, config):
        raise NotImplementedError("method init must be define")

    def _synchronize_data(self, data_instance, flowid, data_application=None):
        header = data_instance.schema.get('header')

        if data_application is None:
            LOGGER.warning("not data_application!")
            return

        transfer_variable = HeteroWorkFlowTransferVariable()
        if data_application == consts.TRAIN_DATA:
            transfer_id = transfer_variable.train_data
        elif data_application == consts.TEST_DATA:
            transfer_id = transfer_variable.test_data
        else:
            LOGGER.warning("data_application error!")
            return

        if self.role == consts.GUEST:
            data_sid = data_instance.mapValues(lambda v: 1)

            federation.remote(data_sid,
                              name=transfer_id.name,
                              tag=transfer_variable.generate_transferid(transfer_id, flowid),
                              role=consts.HOST,
                              idx=0)
            LOGGER.info("remote {} to host".format(data_application))
            return None
        elif self.role == consts.HOST:
            data_sid = federation.get(name=transfer_id.name,
                                      tag=transfer_variable.generate_transferid(transfer_id, flowid),
                                      idx=0)

            LOGGER.info("get {} from guest".format(data_application))
            join_data_insts = data_sid.join(data_instance, lambda s, d: d)
            join_data_insts.schema['header'] = header
            return join_data_insts

    def _initialize_workflow_param(self, config_path):
        workflow_param = WorkFlowParam()
        self.workflow_param = ParamExtract.parse_param_from_config(workflow_param, config_path)

        WorkFlowParamChecker.check_param(self.workflow_param)

    def _init_logger(self, LOGGER_path):
        pass

    def train(self, train_data, validation_data=None):
        if self.mode == consts.HETERO and self.role != consts.ARBITER:
            LOGGER.debug("Enter train function")
            LOGGER.debug("Star intersection before train")
            intersect_flowid = "train_0"
            train_data = self.intersect(train_data, intersect_flowid)
            if validation_data is not None:
                intersect_flowid = "predict_0"
                LOGGER.debug("Star intersection before predict")
                validation_data = self.intersect(validation_data, intersect_flowid)
                LOGGER.debug("End intersection before predict")

            LOGGER.debug("End intersection before train")

        sample_flowid = "train_sample_0"
        train_data = self.sample(train_data, sample_flowid)
        train_data = self.feature_selection_fit(train_data)
        if self.mode == consts.HETERO and self.role != consts.ARBITER:
            train_data, cols_scale_value = self.scale(train_data)
        train_data = self.one_hot_encoder_fit_transform(train_data)

        if validation_data is not None:
            validation_data = self.feature_selection_transform(validation_data)
            if self.mode == consts.HETERO and self.role != consts.ARBITER:
                validation_data, cols_scale_value = self.scale(validation_data, cols_scale_value)

            validation_data = self.one_hot_encoder_transform(validation_data)

        if self.workflow_param.one_vs_rest:
            one_vs_rest_param = OneVsRestParam()
            self.one_vs_rest_param = ParamExtract.parse_param_from_config(one_vs_rest_param, self.config_path)
            one_vs_rest = OneVsRest(self.model, self.role, self.mode, self.one_vs_rest_param)
            self.model = one_vs_rest

        self.model.fit(train_data)
        self.save_model()
        LOGGER.debug("finish saving, self role: {}".format(self.role))
        if self.role == consts.GUEST or self.role == consts.HOST or \
                self.mode == consts.HOMO:
            eval_result = {}
            LOGGER.debug("predicting...")
            predict_result = self.model.predict(train_data,
                                                self.workflow_param.predict_param)

            LOGGER.debug("evaluating...")
            train_eval = self.evaluate(predict_result)
            eval_result[consts.TRAIN_EVALUATE] = train_eval
            if validation_data is not None:
                self.model.set_flowid("1")
                val_pred = self.model.predict(validation_data,
                                              self.workflow_param.predict_param)
                val_eval = self.evaluate(val_pred)
                eval_result[consts.VALIDATE_EVALUATE] = val_eval
            LOGGER.info("{} eval_result: {}".format(self.role, eval_result))
            self.save_eval_result(eval_result)

    def one_vs_rest_train(self, train_data, validation_data=None):
        one_vs_rest_param = OneVsRestParam()
        self.one_vs_rest_param = ParamExtract.parse_param_from_config(one_vs_rest_param, self.config_path)
        one_vs_rest = OneVsRest(self.model, self.role, self.mode, self.one_vs_rest_param)
        LOGGER.debug("Start OneVsRest train")
        one_vs_rest.fit(train_data)
        LOGGER.debug("Start OneVsRest predict")
        one_vs_rest.predict(validation_data, self.workflow_param.predict_param)
        save_result = one_vs_rest.save_model(self.workflow_param.model_table, self.workflow_param.model_namespace)
        if save_result is None:
            return

        for meta_buffer_type, param_buffer_type in save_result:
            self.pipeline.node_meta.append(meta_buffer_type)
            self.pipeline.node_param.append(param_buffer_type)

    def one_vs_rest_predict(self, data_instance):
        if self.mode == consts.HETERO:
            LOGGER.debug("Star intersection before predict")
            intersect_flowid = "predict_module_0"
            data_instance = self.intersect(data_instance, intersect_flowid)
            LOGGER.debug("End intersection before predict")

        # data_instance = self.feature_selection_transform(data_instance)

        # data_instance, fit_config = self.scale(data_instance)
        one_vs_rest_param = OneVsRestParam()
        self.one_vs_rest_param = ParamExtract.parse_param_from_config(one_vs_rest_param, self.config_path)
        one_vs_rest = OneVsRest(self.model, self.role, self.mode, self.one_vs_rest_param)
        one_vs_rest.load_model(self.workflow_param.model_table, self.workflow_param.model_namespace)
        predict_result = one_vs_rest.predict(data_instance, self.workflow_param.predict_param)

        if not predict_result:
            return None

        if predict_result.count() > 10:
            local_predict = predict_result.collect()
            n = 0
            while n < 10:
                result = local_predict.__next__()
                LOGGER.debug("predict result: {}".format(result))
                n += 1

        return predict_result

    def save_eval_result(self, eval_data):
        session.parallelize([eval_data],
                            include_key=False,
                            name=self.workflow_param.evaluation_output_table,
                            namespace=self.workflow_param.evaluation_output_namespace,
                            error_if_exist=False,
                            persistent=True
                            )

    def predict(self, data_instance):
        if self.mode == consts.HETERO:
            LOGGER.debug("Start intersection before predict")
            intersect_flowid = "predict_module_0"
            data_instance = self.intersect(data_instance, intersect_flowid)
            LOGGER.debug("End intersection before predict")

        data_instance = self.feature_selection_transform(data_instance)

        data_instance, fit_config = self.scale(data_instance)

        data_instance = self.one_hot_encoder_transform(data_instance)

        predict_result = self.model.predict(data_instance,
                                            self.workflow_param.predict_param)

        if self.role == consts.GUEST:
            self.save_predict_result(predict_result)
            if self.workflow_param.dataio_param.with_label:
                eval_result = self.evaluate(predict_result)
                LOGGER.info("eval_result: {}".format(eval_result))
                self.save_eval_result(eval_result)
        if self.mode == consts.HOMO and self.role == consts.HOST:
            self.save_predict_result(predict_result)

        if not predict_result:
            return None
        LOGGER.debug("predict result: {}".format(predict_result))
        if predict_result.count() > 10:
            local_predict = predict_result.collect()
            n = 0
            while n < 10:
                result = local_predict.__next__()
                LOGGER.debug("predict result: {}".format(result))
                n += 1

        return predict_result

    def intersect(self, data_instance, intersect_flowid=''):
        if data_instance is None:
            return data_instance

        if self.workflow_param.need_intersect:
            header = data_instance.schema.get('header')
            LOGGER.info("need_intersect: true!")
            intersect_param = IntersectParam()
            self.intersect_params = ParamExtract.parse_param_from_config(intersect_param, self.config_path)

            LOGGER.info("Start intersection!")
            if self.role == consts.HOST:
                intersect_operator = RawIntersectionHost(self.intersect_params)
            elif self.role == consts.GUEST:
                intersect_operator = RawIntersectionGuest(self.intersect_params)
            elif self.role == consts.ARBITER:
                return data_instance
            else:
                raise ValueError("Unknown role of workflow")
            intersect_operator.set_flowid(intersect_flowid)
            intersect_ids = intersect_operator.run(data_instance)
            LOGGER.info("finish intersection!")

            return intersect_ids
        else:
            LOGGER.info("need_intersect: false!")
            return data_instance

    def feature_binning(self, data_instances, flow_id='sample_flowid'):
        if self.mode == consts.HOMO:
            LOGGER.info("Homo feature selection is not supporting yet. Coming soon")
            return data_instances

        if data_instances is None:
            return data_instances

        LOGGER.info("Start feature binning")
        feature_binning_param = param_generator.FeatureBinningParam()
        feature_binning_param = ParamExtract.parse_param_from_config(feature_binning_param, self.config_path)
        param_checker.FeatureBinningParamChecker.check_param(feature_binning_param)

        if self.role == consts.HOST:
            feature_binning_obj = HeteroFeatureBinningHost(feature_binning_param)
        elif self.role == consts.GUEST:
            feature_binning_obj = HeteroFeatureBinningGuest(feature_binning_param)
        elif self.role == consts.ARBITER:
            return data_instances
        else:
            raise ValueError("Unknown role of workflow")

        feature_binning_obj.set_flowid(flow_id)
        if feature_binning_param.local_only:
            data_instances = feature_binning_obj.fit_local(data_instances)
        else:
            data_instances = feature_binning_obj.fit(data_instances)
        save_result = feature_binning_obj.save_model(self.workflow_param.model_table,
                                                     self.workflow_param.model_namespace)
        # Save model result in pipeline
        for meta_buffer_type, param_buffer_type in save_result:
            self.pipeline.node_meta.append(meta_buffer_type)
            self.pipeline.node_param.append(param_buffer_type)

        LOGGER.info("Finish feature selection")
        return data_instances

    def feature_selection_fit(self, data_instance, flow_id='sample_flowid', without_transform=False):
        if self.mode == consts.HOMO:
            LOGGER.info("Homo feature selection is not supporting yet. Coming soon")
            return data_instance

        if data_instance is None:
            return data_instance

        if self.workflow_param.need_feature_selection:
            LOGGER.info("Start feature selection fit")
            feature_select_param = param_generator.FeatureSelectionParam()
            feature_select_param = ParamExtract.parse_param_from_config(feature_select_param, self.config_path)
            param_checker.FeatureSelectionParamChecker.check_param(feature_select_param)

            filter_methods = feature_select_param.filter_method

            if 'iv_value_thres' in filter_methods or 'iv_percentile' in filter_methods:
                binning_flowid = '_'.join(['feature_binning', str(flow_id)])
                LOGGER.debug("Current binning flowid: {}".format(binning_flowid))
                data_instance = self.feature_binning(data_instances=data_instance, flow_id=binning_flowid)

            if self.role == consts.HOST:
                feature_selector = HeteroFeatureSelectionHost(feature_select_param)
            elif self.role == consts.GUEST:
                feature_selector = HeteroFeatureSelectionGuest(feature_select_param)
            elif self.role == consts.ARBITER:
                return data_instance
            else:
                raise ValueError("Unknown role of workflow")

            feature_selector.set_flowid(flow_id)
            filter_methods = feature_select_param.filter_method
            previous_model = {}
            if 'iv_value_thres' in filter_methods or 'iv_percentile' in filter_methods:
                binning_model = {
                    'name': self.workflow_param.model_table,
                    'namespace': self.workflow_param.model_namespace
                }
                previous_model['binning_model'] = binning_model
            feature_selector.init_previous_model(**previous_model)

            if without_transform:
                data_instance = feature_selector.fit(data_instance)
            else:
                data_instance = feature_selector.fit_transform(data_instance)
            save_result = feature_selector.save_model(self.workflow_param.model_table,
                                                      self.workflow_param.model_namespace)

            LOGGER.debug(
                "Role: {}, in fit feature selector left_cols: {}".format(self.role, feature_selector.left_cols))
            # Save model result in pipeline
            for meta_buffer_type, param_buffer_type in save_result:
                self.pipeline.node_meta.append(meta_buffer_type)
                self.pipeline.node_param.append(param_buffer_type)

            LOGGER.info("Finish feature selection")
            return data_instance
        else:
            LOGGER.info("No need to do feature selection")
            return data_instance

    def feature_selection_transform(self, data_instance, flow_id='sample_flowid'):
        if self.mode == consts.HOMO:
            LOGGER.info("Homo feature selection is not supporting yet. Coming soon")
            return data_instance

        if data_instance is None:
            return data_instance

        if self.workflow_param.need_feature_selection:
            LOGGER.info("Start feature selection transform")
            feature_select_param = param_generator.FeatureSelectionParam()
            feature_select_param = ParamExtract.parse_param_from_config(feature_select_param, self.config_path)
            param_checker.FeatureSelectionParamChecker.check_param(feature_select_param)

            if self.role == consts.HOST:
                feature_selector = HeteroFeatureSelectionHost(feature_select_param)
            elif self.role == consts.GUEST:
                feature_selector = HeteroFeatureSelectionGuest(feature_select_param)
            elif self.role == consts.ARBITER:
                return data_instance
            else:
                raise ValueError("Unknown role of workflow")

            feature_selector.set_flowid(flow_id)

            feature_selector.load_model(self.workflow_param.model_table, self.workflow_param.model_namespace)

            LOGGER.debug(
                "Role: {}, in transform feature selector left_cols: {}".format(self.role, feature_selector.left_cols))

            data_instance = feature_selector.transform(data_instance)

            LOGGER.info("Finish feature selection")
            return data_instance
        else:
            LOGGER.info("No need to do feature selection")
            return data_instance

    def one_hot_encoder_fit_transform(self, data_instance):
        if data_instance is None:
            return data_instance

        if self.workflow_param.need_one_hot:
            LOGGER.info("Start one-hot encode")
            one_hot_param = param_generator.OneHotEncoderParam()
            one_hot_param = ParamExtract.parse_param_from_config(one_hot_param, self.config_path)
            param_checker.OneHotEncoderParamChecker.check_param(one_hot_param)

            one_hot_encoder = OneHotEncoder(one_hot_param)

            data_instance = one_hot_encoder.fit_transform(data_instance)
            save_result = one_hot_encoder.save_model(self.workflow_param.model_table,
                                                     self.workflow_param.model_namespace)
            # Save model result in pipeline
            for meta_buffer_type, param_buffer_type in save_result:
                self.pipeline.node_meta.append(meta_buffer_type)
                self.pipeline.node_param.append(param_buffer_type)

            LOGGER.info("Finish one-hot encode")
            return data_instance
        else:
            LOGGER.info("No need to do one-hot encode")
            return data_instance

    def one_hot_encoder_transform(self, data_instance):
        if data_instance is None:
            return data_instance

        if self.workflow_param.need_one_hot:
            LOGGER.info("Start one-hot encode")
            one_hot_param = param_generator.OneHotEncoderParam()
            one_hot_param = ParamExtract.parse_param_from_config(one_hot_param, self.config_path)
            param_checker.OneHotEncoderParamChecker.check_param(one_hot_param)

            one_hot_encoder = OneHotEncoder(one_hot_param)
            one_hot_encoder.load_model(self.workflow_param.model_table, self.workflow_param.model_namespace)

            data_instance = one_hot_encoder.transform(data_instance)

            LOGGER.info("Finish one-hot encode")
            return data_instance
        else:
            LOGGER.info("No need to do one-hot encode")
            return data_instance

    def sample(self, data_instance, sample_flowid="sample_flowid"):
        if not self.workflow_param.need_sample:
            LOGGER.info("need_sample: false!")
            return data_instance

        if self.role == consts.ARBITER:
            LOGGER.info("arbiter not need sample")
            return data_instance

        LOGGER.info("need_sample: true!")

        sample_param = SampleParam()
        sample_param = ParamExtract.parse_param_from_config(sample_param, self.config_path)
        sampler = Sampler(sample_param)

        sampler.set_flowid(sample_flowid)
        data_instance = sampler.run(data_instance, self.mode, self.role)
        LOGGER.info("sample result size is {}".format(data_instance.count()))
        return data_instance

    def cross_validation(self, data_instance):
        if self.mode == consts.HETERO:
            cv_results = self.hetero_cross_validation(data_instance)
        elif self.mode == consts.HOMO:
            cv_results = self.homo_cross_validation(data_instance)
        else:
            cv_results = {}

        LOGGER.debug("cv_result: {}".format(cv_results))
        if self.role == consts.GUEST or (self.role == consts.HOST and self.mode == consts.HOMO):
            format_cv_result = {}
            for eval_result in cv_results:
                for eval_name, eval_r in eval_result.items():
                    if not isinstance(eval_r, list):
                        if eval_name not in format_cv_result:
                            format_cv_result[eval_name] = []
                        format_cv_result[eval_name].append(eval_r)
                    else:
                        for e_r in eval_r:
                            e_name = "{}_thres_{}".format(eval_name, e_r[0])
                            if e_name not in format_cv_result:
                                format_cv_result[e_name] = []
                            format_cv_result[e_name].append(e_r[1])

            for eval_name, eva_result_list in format_cv_result.items():
                mean_value = np.around(np.mean(eva_result_list), 4)
                std_value = np.around(np.std(eva_result_list), 4)
                LOGGER.info("{}，evaluate name: {}, mean: {}, std: {}".format(self.role,
                                                                             eval_name, mean_value, std_value))

    def hetero_cross_validation(self, data_instance):
        LOGGER.debug("Enter train function")
        LOGGER.debug("Start intersection before train")
        intersect_flowid = "cross_validation_0"
        data_instance = self.intersect(data_instance, intersect_flowid)
        LOGGER.debug("End intersection before train")

        n_splits = self.workflow_param.n_splits
        if self.role == consts.GUEST:
            LOGGER.info("In hetero cross_validation Guest")
            k_fold_obj = KFold(n_splits=n_splits)
            kfold_data_generator = k_fold_obj.split(data_instance)
            flowid = 0
            cv_results = []
            for train_data, test_data in kfold_data_generator:
                self._init_pipeline()
                LOGGER.info("flowid:{}".format(flowid))
                self._synchronize_data(train_data, flowid, consts.TRAIN_DATA)
                LOGGER.info("synchronize train data")
                self._synchronize_data(test_data, flowid, consts.TEST_DATA)
                LOGGER.info("synchronize test data")

                LOGGER.info("Start sample before train")
                sample_flowid = "sample_" + str(flowid)
                train_data = self.sample(train_data, sample_flowid)
                LOGGER.info("End sample before_train")

                feature_selection_flowid = "feature_selection_fit_" + str(flowid)
                train_data = self.feature_selection_fit(train_data, feature_selection_flowid)
                LOGGER.info("End feature selection fit_transform")

                train_data, cols_scale_value = self.scale(train_data)

                train_data = self.one_hot_encoder_fit_transform(train_data)

                self.model.set_flowid(flowid)
                self.model.fit(train_data)

                feature_selection_flowid = "feature_selection_transform_" + str(flowid)
                test_data = self.feature_selection_transform(test_data, feature_selection_flowid)
                LOGGER.info("End feature selection transform")

                test_data, cols_scale_value = self.scale(test_data, cols_scale_value)

                test_data = self.one_hot_encoder_transform(test_data)
                pred_res = self.model.predict(test_data, self.workflow_param.predict_param)
                evaluation_results = self.evaluate(pred_res)
                cv_results.append(evaluation_results)
                flowid += 1
                LOGGER.info("cv" + str(flowid) + " evaluation:" + str(evaluation_results))
                self._initialize_model(self.config_path)

            LOGGER.info("total cv evaluation:{}".format(cv_results))
            return cv_results

        elif self.role == consts.HOST:
            LOGGER.info("In hetero cross_validation Host")
            for flowid in range(n_splits):
                self._init_pipeline()
                LOGGER.info("flowid:{}".format(flowid))
                train_data = self._synchronize_data(data_instance, flowid, consts.TRAIN_DATA)
                LOGGER.info("synchronize train data")
                test_data = self._synchronize_data(data_instance, flowid, consts.TEST_DATA)
                LOGGER.info("synchronize test data")

                LOGGER.info("Start sample before train")
                sample_flowid = "sample_" + str(flowid)
                train_data = self.sample(train_data, sample_flowid)
                LOGGER.info("End sample before_train")

                feature_selection_flowid = "feature_selection_fit_" + str(flowid)
                train_data = self.feature_selection_fit(train_data, feature_selection_flowid)
                LOGGER.info("End feature selection fit_transform")
                train_data = self.one_hot_encoder_fit_transform(train_data)
                self.model.set_flowid(flowid)
                self.model.fit(train_data)

                feature_selection_flowid = "feature_selection_transform_" + str(flowid)
                test_data = self.feature_selection_transform(test_data, feature_selection_flowid)
                LOGGER.info("End feature selection transform")

                test_data = self.one_hot_encoder_transform(test_data)
                self.model.predict(test_data)
                flowid += 1
                self._initialize_model(self.config_path)

        elif self.role == consts.ARBITER:
            LOGGER.info("In hetero cross_validation Arbiter")
            for flowid in range(n_splits):
                LOGGER.info("flowid:{}".format(flowid))
                self.model.set_flowid(flowid)
                self.model.fit()
                flowid += 1
                self._initialize_model(self.config_path)

    def load_eval_result(self):
        eval_data = session.table(
            name=self.workflow_param.evaluation_output_table,
            namespace=self.workflow_param.evaluation_output_namespace,
        )
        LOGGER.debug("Evaluate result loaded: {}".format(eval_data))
        return eval_data

    def homo_cross_validation(self, data_instance):
        n_splits = self.workflow_param.n_splits
        k_fold_obj = KFold(n_splits=n_splits)
        kfold_data_generator = k_fold_obj.split(data_instance)
        cv_result = []
        flowid = 0
        LOGGER.info("Doing Homo cross validation")
        for train_data, test_data in kfold_data_generator:
            LOGGER.info("This is the {}th fold".format(flowid))

            LOGGER.info("Start sample before train")
            sample_flowid = "sample_" + str(flowid)
            train_data = self.sample(train_data, sample_flowid)
            LOGGER.info("End sample before_train")

            train_data = self.one_hot_encoder_fit_transform(train_data)

            self.model.set_flowid(flowid)
            self.model.fit(train_data)
            # self.save_model()
            test_data = self.one_hot_encoder_transform(test_data)
            predict_result = self.model.predict(test_data, self.workflow_param.predict_param)
            flowid += 1
            eval_result = self.evaluate(predict_result)
            cv_result.append(eval_result)
            self._initialize_model(self.config_path)

        return cv_result

    def save_model(self):
        LOGGER.debug("save model, model table: {}, model namespace: {}".format(
            self.workflow_param.model_table, self.workflow_param.model_namespace))
        save_result = self.model.save_model(self.workflow_param.model_table, self.workflow_param.model_namespace)
        if save_result is None:
            return
        for meta_buffer_type, param_buffer_type in save_result:
            self.pipeline.node_meta.append(meta_buffer_type)
            self.pipeline.node_param.append(param_buffer_type)

    def load_model(self):
        self.model.load_model(self.workflow_param.model_table, self.workflow_param.model_namespace)

    def save_predict_result(self, predict_result):
        predict_result.save_as(self.workflow_param.predict_output_table, self.workflow_param.predict_output_namespace)

    def save_intersect_result(self, intersect_result):
        if intersect_result:
            LOGGER.info("Save intersect results to name:{}, namespace:{}".format(
                self.workflow_param.intersect_data_output_table, self.workflow_param.intersect_data_output_namespace))
            intersect_result.save_as(self.workflow_param.intersect_data_output_table,
                                     self.workflow_param.intersect_data_output_namespace)
        else:
            LOGGER.info("Not intersect_result, do not save it!")

    def scale(self, data_instance, fit_config=None):
        if self.workflow_param.need_scale:
            scale_params = ScaleParam()
            self.scale_params = ParamExtract.parse_param_from_config(scale_params, self.config_path)
            param_checker.ScaleParamChecker.check_param(self.scale_params)

            scale_obj = Scaler(self.scale_params)
            if self.workflow_param.method == "predict":
                fit_config = scale_obj.load_model(name=self.workflow_param.model_table,
                                                  namespace=self.workflow_param.model_namespace,
                                                  header=data_instance.schema.get("header"))

            if not fit_config:
                data_instance, fit_config = scale_obj.fit(data_instance)
                save_results = scale_obj.save_model(name=self.workflow_param.model_table,
                                                    namespace=self.workflow_param.model_namespace)
                if save_results:
                    for meta_buffer_type, param_buffer_type in save_results:
                        self.pipeline.node_meta.append(meta_buffer_type)
                        self.pipeline.node_param.append(param_buffer_type)
            else:
                data_instance = scale_obj.transform(data_instance, fit_config)
        else:
            LOGGER.debug("workflow param need_scale is False")

        return data_instance, fit_config

    def evaluate(self, eval_data):
        if eval_data is None:
            LOGGER.info("not eval_data!")
            return None

        eval_data_local = eval_data.collect()
        labels = []
        pred_prob = []
        pred_labels = []
        data_num = 0
        for data in eval_data_local:
            data_num += 1
            labels.append(data[1][0])
            pred_prob.append(data[1][1])
            pred_labels.append(data[1][2])

        labels = np.array(labels)
        pred_prob = np.array(pred_prob)
        pred_labels = np.array(pred_labels)

        evaluation_result = self.model.evaluate(labels, pred_prob, pred_labels,
                                                evaluate_param=self.workflow_param.evaluate_param)
        return evaluation_result

    def gen_data_instance(self, table, namespace, mode="fit"):
        reader = None
        if self.workflow_param.dataio_param.input_format == "dense":
            reader = DenseFeatureReader(self.workflow_param.dataio_param)
        elif self.workflow_param.dataio_param.input_format == "sparse":
            reader = SparseFeatureReader(self.workflow_param.dataio_param)
        else:
            reader = SparseTagReader(self.workflow_param.dataio_param)

        LOGGER.debug("mode is {}".format(mode))

        if mode == "transform":
            reader.load_model(self.workflow_param.model_table,
                              self.workflow_param.model_namespace)

        data_instance = reader.read_data(table,
                                         namespace,
                                         mode=mode)

        if mode == "fit":
            save_result = reader.save_model(self.workflow_param.model_table,
                                            self.workflow_param.model_namespace)

            for meta_buffer_type, param_buffer_type in save_result:
                self.pipeline.node_meta.append(meta_buffer_type)
                self.pipeline.node_param.append(param_buffer_type)

        return data_instance

    def _init_pipeline(self):
        pipeline_obj = pipeline_pb2.Pipeline()
        # pipeline_obj.node_meta = []
        # pipeline_obj.node_param = []
        self.pipeline = pipeline_obj
        LOGGER.debug("finish init pipeline")

    def _save_pipeline(self):
        buffer_type = "Pipeline"

        model_manager.save_model(buffer_type=buffer_type,
                                 proto_buffer=self.pipeline,
                                 name=self.workflow_param.model_table,
                                 namespace=self.workflow_param.model_namespace)

    def _load_pipeline(self):
        buffer_type = "Pipeline"
        pipeline_obj = pipeline_pb2.Pipeline()
        pipeline_obj = model_manager.read_model(buffer_type=buffer_type,
                                                proto_buffer=pipeline_obj,
                                                name=self.workflow_param.model_table,
                                                namespace=self.workflow_param.model_namespace)
        pipeline_obj.node_meta = list(pipeline_obj.node_meta)
        pipeline_obj.node_param = list(pipeline_obj.node_param)
        self.pipeline = pipeline_obj

    def _init_argument(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', required=True, type=str, help="Specify a config json file path")
        parser.add_argument('-j', '--job_id', type=str, required=True, help="Specify the job id")
        # parser.add_argument('-p', '--party_id', type=str, required=True, help="Specify the party id")
        # parser.add_argument('-l', '--LOGGER_path', type=str, required=True, help="Specify the LOGGER path")
        args = parser.parse_args()
        config_path = args.config
        self.config_path = config_path
        if not args.config:
            LOGGER.error("Config File should be provided")
            exit(-100)
        self.job_id = args.job_id

        home_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        param_validation_path = home_dir + "/conf/param_validation.json"
        all_checker = AllChecker(config_path, param_validation_path)
        all_checker.check_all()
        LOGGER.debug("Finish all parameter checkers")
        self._initialize(config_path)
        with open(config_path) as conf_f:
            runtime_json = json.load(conf_f)
        session.init(self.job_id, self.workflow_param.work_mode)
        LOGGER.debug("The job id is {}".format(self.job_id))
        federation.init(self.job_id, runtime_json)
        LOGGER.debug("Finish eggroll and federation init")
        self._init_pipeline()

    @status_tracer_decorator.status_trace
    def run(self):
        self._init_argument()

        if self.workflow_param.method == "train":

            # create a new pipeline

            LOGGER.debug("In running function, enter train method")
            train_data_instance = None
            predict_data_instance = None
            if self.role != consts.ARBITER:
                LOGGER.debug("Input table:{}, input namesapce: {}".format(
                    self.workflow_param.train_input_table, self.workflow_param.train_input_namespace
                ))
                train_data_instance = self.gen_data_instance(self.workflow_param.train_input_table,
                                                             self.workflow_param.train_input_namespace)
                LOGGER.debug("gen_data_finish")
                if self.workflow_param.predict_input_table is not None and self.workflow_param.predict_input_namespace is not None:
                    LOGGER.debug("Input table:{}, input namesapce: {}".format(
                        self.workflow_param.predict_input_table, self.workflow_param.predict_input_namespace
                    ))
                    predict_data_instance = self.gen_data_instance(self.workflow_param.predict_input_table,
                                                                   self.workflow_param.predict_input_namespace,
                                                                   mode='transform')

            self.train(train_data_instance, validation_data=predict_data_instance)
            self._save_pipeline()

        elif self.workflow_param.method == "predict":
            data_instance = self.gen_data_instance(self.workflow_param.predict_input_table,
                                                   self.workflow_param.predict_input_namespace,
                                                   mode='transform')

            if self.workflow_param.one_vs_rest:
                one_vs_rest_param = OneVsRestParam()
                self.one_vs_rest_param = ParamExtract.parse_param_from_config(one_vs_rest_param, self.config_path)
                one_vs_rest = OneVsRest(self.model, self.role, self.mode, self.one_vs_rest_param)
                self.model = one_vs_rest

            self.load_model()
            self.predict(data_instance)

        elif self.workflow_param.method == "intersect":
            LOGGER.debug("[Intersect]Input table:{}, input namesapce: {}".format(
                self.workflow_param.data_input_table,
                self.workflow_param.data_input_namespace
            ))
            data_instance = self.gen_data_instance(self.workflow_param.data_input_table,
                                                   self.workflow_param.data_input_namespace)
            self.intersect(data_instance)

        elif self.workflow_param.method == "cross_validation":
            data_instance = None
            if self.role != consts.ARBITER:
                data_instance = self.gen_data_instance(self.workflow_param.data_input_table,
                                                       self.workflow_param.data_input_namespace)
            self.cross_validation(data_instance)

        elif self.workflow_param.method == "one_vs_rest_train":
            LOGGER.debug("In running function, enter one_vs_rest method")
            train_data_instance = None
            predict_data_instance = None
            if self.role != consts.ARBITER:
                LOGGER.debug("Input table:{}, input namesapce: {}".format(
                    self.workflow_param.train_input_table, self.workflow_param.train_input_namespace
                ))
                train_data_instance = self.gen_data_instance(self.workflow_param.train_input_table,
                                                             self.workflow_param.train_input_namespace)
                LOGGER.debug("gen_data_finish")
                if self.workflow_param.predict_input_table is not None and self.workflow_param.predict_input_namespace is not None:
                    LOGGER.debug("Input table:{}, input namesapce: {}".format(
                        self.workflow_param.predict_input_table, self.workflow_param.predict_input_namespace
                    ))
                    predict_data_instance = self.gen_data_instance(self.workflow_param.predict_input_table,
                                                                   self.workflow_param.predict_input_namespace)

            self.one_vs_rest_train(train_data_instance, validation_data=predict_data_instance)
            # self.one_vs_rest_predict(predict_data_instance)
            self._save_pipeline()

        else:
            raise TypeError("method %s is not support yet" % (self.workflow_param.method))


if __name__ == "__main__":
    pass
    """
    method_list
    param_init
    method.run(params)
    """
