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
from arch.api.utils import log_utils
from federatedml.param import param
from federatedml.util import consts
from federatedml.util.param_extract import ParamExtract
import inspect
import json

LOGGER = log_utils.getLogger()


class DataIOParamChecker(object):
    @staticmethod
    def check_param(dataio_param):
        if type(dataio_param).__name__ != "DataIOParam":
            raise ValueError("dataio_param {} not supported, should be DataIOParam object".format(dataio_param))

        descr = "dataio param's"

        dataio_param.input_format = check_and_change_lower(dataio_param.input_format,
                                                           ["dense", "sparse", "tag"],
                                                           descr)

        dataio_param.output_format = check_and_change_lower(dataio_param.output_format,
                                                            ["dense", "sparse"],
                                                            descr)

        dataio_param.data_type = check_and_change_lower(dataio_param.data_type,
                                                        ["int", "int64", "float", "float64", "str", "long"],
                                                        descr)

        if type(dataio_param.missing_fill).__name__ != 'bool':
            raise ValueError("dataio param's missing_fill {} not supported".format(dataio_param.missing_fill))

        if dataio_param.missing_fill_method is not None:
            dataio_param.missing_fill_method = check_and_change_lower(dataio_param.missing_fill_method,
                                                                      ['min', 'max', 'mean', 'designated'],
                                                                      descr)

        if dataio_param.outlier_replace_method is not None:
            dataio_param.outlier_replace_method = check_and_change_lower(dataio_param.outlier_replace_method,
                                                                         ['min', 'max', 'mean', 'designated'],
                                                                         descr)

        if type(dataio_param.with_label).__name__ != 'bool':
            raise ValueError("dataio param's with_label {} not supported".format(dataio_param.with_label))

        if dataio_param.with_label:
            if type(dataio_param.label_idx).__name__ not in ["long", "int"]:
                raise ValueError("dataio param's label_idx {} not supported".format(dataio_param.label_idx))

            dataio_param.label_type = check_and_change_lower(dataio_param.label_type,
                                                             ["int", "int64", "float", "float64", "str", "long"],
                                                             descr)
        return True


class ObjectiveParamChecker(object):
    @staticmethod
    def check_param(objective_param, task_type=None):
        if type(objective_param).__name__ != "ObjectiveParam":
            raise ValueError("objective param {} not supportd, should be ObjectiveParam object".format(objective_param))

        if objective_param.objective is None:
            return True

        descr = "objective param's"

        if task_type not in [consts.CLASSIFICATION, consts.REGRESSION]:
            objective_param.objective = check_and_change_lower(objective_param.objective,
                                                               ["cross_entropy", "lse", "lae", "huber", "fair",
                                                                "log_cosh", "tweedie"],
                                                               descr)

        if task_type == consts.CLASSIFICATION:
            if objective_param.objective != "cross_entropy":
                raise ValueError("objective param's objective {} not supported".format(objective_param.objective))

        elif task_type == consts.REGRESSION:
            objective_param.objective = check_and_change_lower(objective_param.objective,
                                                               ["lse", "lae", "huber", "fair", "log_cosh", "tweedie"],
                                                               descr)

            params = objective_param.params
            if objective_param.objective in ["huber", "fair", "tweedie"]:
                if type(params).__name__ != 'list' or len(params) < 1:
                    raise ValueError(
                        "objective param's params {} not supported, should be non-empty list".format(params))

                if type(params[0]).__name__ not in ["float", "int", "long"]:
                    raise ValueError("objective param's params[0] {} not supported".format(objective_param.params[0]))

                if objective_param.objective == 'tweedie':
                    if params[0] < 1 or params[0] >= 2:
                        raise ValueError("in tweedie regression, objective params[0] should betweend [1, 2)")

                if objective_param.objective == 'fair' or 'huber':
                    if params[0] <= 0.0:
                        raise ValueError("in {} regression, objective params[0] should greater than 0.0".format(
                            objective_param.objective))
        return True


class EncryptParamChecker(object):
    @staticmethod
    def check_param(encrypt_param):
        if type(encrypt_param.method).__name__ != "str":
            raise ValueError(
                "encrypt_param's method {} not supported, should be str type".format(
                    encrypt_param.method))
        else:
            user_input = encrypt_param.method.lower()
            if user_input == 'paillier':
                encrypt_param.method = consts.PAILLIER

        if type(encrypt_param.key_length).__name__ != "int":
            raise ValueError(
                "encrypt_param's key_length {} not supported, should be int type".format(encrypt_param.key_length))
        elif encrypt_param.key_length <= 0:
            raise ValueError(
                "encrypt_param's key_length must be greater or equal to 1")

        LOGGER.debug("Finish encrypt parameter check!")
        return True


class EncryptedModeCalculatorParamChecker(object):
    @staticmethod
    def check_param(encrypted_mode_calculator):
        if type(encrypted_mode_calculator).__name__ != "EncryptedModeCalculatorParam":
            raise ValueError("param class not match EncryptedModeCalculatorParam")

        descr = "encrypted_mode_calculator param"
        encrypted_mode_calculator.mode = check_and_change_lower(encrypted_mode_calculator.mode,
                                                                ["strict", "fast", "balance"],
                                                                descr)

        if encrypted_mode_calculator.mode == "balance":
            if type(encrypted_mode_calculator.re_encrypted_rate).__name__ not in ["int", "long", "float"]:
                raise ValueError("re_encrypted_rate should be a numeric number")

        return True


class SampleParamChecker(object):
    @staticmethod
    def check_param(sample_param):
        if type(sample_param).__name__ != "SampleParam":
            raise ValueError("sample param {} not supported, should be SampleParam object".format(sample_param))

        descr = "sample param"
        sample_param.mode = check_and_change_lower(sample_param.mode,
                                                   ["random", "stratified"],
                                                   descr)

        sample_param.method = check_and_change_lower(sample_param.method,
                                                     ["upsample", "downsample"],
                                                     descr)

        return True


class DecisionTreeParamChecker(object):
    @staticmethod
    def check_param(tree_param):
        if type(tree_param).__name__ != "DecisionTreeParam":
            raise ValueError(
                "decision tree param {} not supported, should be DecisionTreeParam object".format(tree_param))

        descr = "decision tree param"

        tree_param.criterion_method = check_and_change_lower(tree_param.criterion_method,
                                                             ["xgboost"],
                                                             descr)

        if type(tree_param.criterion_params).__name__ != "list":
            raise ValueError("decision tree param's criterion_params {} not supported, should be list".format(
                tree_param.criterion_params))

        if len(tree_param.criterion_params) == 0:
            raise ValueError("decisition tree param's criterio_params should be non empty")

        if type(tree_param.criterion_params[0]).__name__ not in ["int", "long", "float"]:
            raise ValueError("decision tree param's criterion_params element shoubld be numeric")

        if type(tree_param.max_depth).__name__ not in ["int", "long"]:
            raise ValueError("decision tree param's max_depth {} not supported, should be integer".format(
                tree_param.max_depth))

        if tree_param.max_depth < 1:
            raise ValueError("decision tree param's max_depth should be positive integer, no less than 1")

        if type(tree_param.min_sample_split).__name__ not in ["int", "long"]:
            raise ValueError("decision tree param's min_sample_split {} not supported, should be integer".format(
                tree_param.min_sample_split))

        if type(tree_param.min_impurity_split).__name__ not in ["int", "long", "float"]:
            raise ValueError("decision tree param's min_impurity_split {} not supported, should be numeric".format(
                tree_param.min_impurity_split))

        if type(tree_param.min_leaf_node).__name__ not in ["int", "long"]:
            raise ValueError("decision tree param's min_leaf_node {} not supported, should be integer".format(
                tree_param.min_leaf_node))

        if type(tree_param.max_split_nodes).__name__ not in ["int", "long"] or tree_param.max_split_nodes < 1:
            raise ValueError("decision tree param's max_split_nodes {} not supported, " + \
                             "should be positive integer between 1 and {}".format(tree_param.max_split_nodes,
                                                                                  consts.MAX_SPLIT_NODES))

        if type(tree_param.n_iter_no_change).__name__ != "bool":
            raise ValueError("decision tree param's n_iter_no_change {} not supported, should be bool type".format(
                tree_param.n_iter_no_change))

        if type(tree_param.tol).__name__ not in ["float", "int", "long"]:
            raise ValueError("decision tree param's tol {} not supported, should be numeric".format(tree_param.tol))

        tree_param.feature_importance_type = check_and_change_lower(tree_param.feature_importance_type,
                                                                    ["split", "gain"],
                                                                    descr)

        return True


class BoostingTreeParamChecker(object):
    @staticmethod
    def check_param(boost_param):
        DecisionTreeParamChecker.check_param(boost_param.tree_param)

        descr = "boosting tree param's"

        if boost_param.task_type not in [consts.CLASSIFICATION, consts.REGRESSION]:
            raise ValueError("boosting tree param's task_type {} not supported, should be {} or {}".format(
                boost_param.task_type, consts.CLASSIFICATION, consts.REGRESSION))

        ObjectiveParamChecker.check_param(boost_param.objective_param, boost_param.task_type)

        if type(boost_param.learning_rate).__name__ not in ["float", "int", "long"]:
            raise ValueError("boosting tree param's learning_rate {} not supported, should be numeric".format(
                boost_param.learning_rate))

        if type(boost_param.num_trees).__name__ not in ["int", "long"] or boost_param.num_trees < 1:
            raise ValueError("boosting tree param's num_trees {} not supported, should be postivie integer".format(
                boost_param.num_trees))

        if type(boost_param.subsample_feature_rate).__name__ not in ["float", "int", "long"] or \
                boost_param.subsample_feature_rate < 0 or boost_param.subsample_feature_rate > 1:
            raise ValueError("boosting tree param's subsample_feature_rate should be a numeric number between 0 and 1")

        if type(boost_param.n_iter_no_change).__name__ != "bool":
            raise ValueError("boosting tree param's n_iter_no_change {} not supported, should be bool type".format(
                boost_param.n_iter_no_change))

        if type(boost_param.tol).__name__ not in ["float", "int", "long"]:
            raise ValueError("boosting tree param's tol {} not supported, should be numeric".format(boost_param.tol))

        EncryptParamChecker.check_param(boost_param.encrypt_param)

        boost_param.quantile_method = check_and_change_lower(boost_param.quantile_method,
                                                             ["bin_by_data_block", "bin_by_sample_data"],
                                                             "boosting tree param's quantile_method")

        if type(boost_param.bin_num).__name__ not in ["int", "long"] or boost_param.bin_num < 2:
            raise ValueError(
                "boosting tree param's bin_num {} not supported, should be positive integer greater than 1".format(
                    boost_param.bin_num))

        if type(boost_param.bin_gap).__name__ not in ["float", "int", "long"]:
            raise ValueError(
                "boosting tree param's bin_gap {} not supported, should be numeric".format(boost_param.bin_gap))

        if boost_param.quantile_method == "bin_by_sample_data":
            if type(boost_param.bin_sample_num).__name__ not in ["int", "long"] or boost_param.bin_sample_num < 1:
                raise ValueError("boosting tree param's sample_num {} not supported, should be positive integer".format(
                    boost_param.bin_sample_num))

        return True


class EncodeParamChecker(object):
    @staticmethod
    def check_param(encode_param):
        if type(encode_param.salt).__name__ != "str":
            raise ValueError(
                "encode param's salt {} not supported, should be str type".format(
                    encode_param.salt))

        descr = "encode param's "

        encode_param.encode_method = check_and_change_lower(encode_param.encode_method,
                                                            ["none", "md5", "sha1", "sha224", "sha256", "sha384",
                                                             "sha512"],
                                                            descr)

        if type(encode_param.base64).__name__ != "bool":
            raise ValueError(
                "encode param's base64 {} not supported, should be bool type".format(encode_param.base64))

        LOGGER.debug("Finish encode parameter check!")
        return True


class IntersectParamChecker(object):
    @staticmethod
    def check_param(intersect_param):
        descr = "intersect param's"

        intersect_param.intersect_method = check_and_change_lower(intersect_param.intersect_method,
                                                                  [consts.RSA, consts.RAW],
                                                                  descr)

        if type(intersect_param.random_bit).__name__ not in ["int"]:
            raise ValueError("intersect param's random_bit {} not supported, should be positive integer".format(
                intersect_param.random_bit))

        if type(intersect_param.is_send_intersect_ids).__name__ != "bool":
            raise ValueError(
                "intersect param's is_send_intersect_ids {} not supported, should be bool type".format(
                    intersect_param.is_send_intersect_ids))

        if type(intersect_param.is_get_intersect_ids).__name__ != "bool":
            raise ValueError(
                "intersect param's is_get_intersect_ids {} not supported, should be bool type".format(
                    intersect_param.is_get_intersect_ids))

        intersect_param.join_role = check_and_change_lower(intersect_param.join_role,
                                                           [consts.GUEST, consts.HOST],
                                                           descr)

        if type(intersect_param.with_encode).__name__ != "bool":
            raise ValueError(
                "intersect param's with_encode {} not supported, should be bool type".format(
                    intersect_param.with_encode))

        if type(intersect_param.only_output_key).__name__ != "bool":
            raise ValueError(
                "intersect param's only_output_key {} not supported, should be bool type".format(
                    intersect_param.is_send_intersect_ids))

        EncodeParamChecker.check_param(intersect_param.encode_params)
        LOGGER.debug("Finish intersect parameter check!")
        return True


class PredictParamChecker(object):
    @staticmethod
    def check_param(predict_param):
        if type(predict_param.with_proba).__name__ != "bool":
            raise ValueError(
                "predict param's with_proba {} not supported, should be bool type".format(predict_param.with_proba))

        if type(predict_param.threshold).__name__ not in ["float", "int"]:
            raise ValueError("predict param's predict_param {} not supported, should be float or int".format(
                predict_param.threshold))

        LOGGER.debug("Finish predict parameter check!")
        return True


class OneVsRestChecker(object):
    @staticmethod
    def check_param(one_vs_rest_param):
        if type(one_vs_rest_param.has_arbiter).__name__ != "bool":
            raise ValueError(
                "one_vs_rest param's has_arbiter {} not supported, should be bool type".format(
                    one_vs_rest_param.with_proba))

        LOGGER.debug("Finish one_vs_rest parameter check!")
        return True


class EvaluateParamChecker(object):
    @staticmethod
    def check_param(evaluate_param):
        if type(evaluate_param.metrics).__name__ != "list":
            raise ValueError("evaluate param's metrics {} not supported, should be list".format(evaluate_param.metrics))
        else:
            descr = "evaluate param's metrics"
            for idx, metric in enumerate(evaluate_param.metrics):
                evaluate_param.metrics[idx] = check_and_change_lower(metric,
                                                                     [consts.AUC, consts.KS, consts.LIFT,
                                                                      consts.PRECISION, consts.RECALL, consts.ACCURACY,
                                                                      consts.EXPLAINED_VARIANCE,
                                                                      consts.MEAN_ABSOLUTE_ERROR,
                                                                      consts.MEAN_SQUARED_ERROR,
                                                                      consts.MEAN_SQUARED_LOG_ERROR,
                                                                      consts.MEDIAN_ABSOLUTE_ERROR,
                                                                      consts.R2_SCORE, consts.ROOT_MEAN_SQUARED_ERROR],
                                                                     descr)
        descr = "evaluate param's "

        evaluate_param.classi_type = check_and_change_lower(evaluate_param.classi_type,
                                                            [consts.BINARY, consts.MULTY, consts.REGRESSION],
                                                            descr)

        if type(evaluate_param.pos_label).__name__ not in ["str", "float", "int"]:
            raise ValueError(
                "evaluate param's pos_label {} not supported, should be str or float or int type".format(
                    evaluate_param.pos_label))

        if type(evaluate_param.thresholds).__name__ != "list":
            raise ValueError(
                "evaluate param's thresholds {} not supported, should be list".format(evaluate_param.thresholds))
        else:
            for threshold in evaluate_param.thresholds:
                if type(threshold).__name__ not in ["float", "int"]:
                    raise ValueError(
                        "threshold {} in evaluate param's thresholds not supported, should be positive integer".format(
                            threshold))

        LOGGER.debug("Finish evaluation parameter check!")
        return True


class WorkFlowParamChecker(object):
    @staticmethod
    def check_param(workflow_param):

        descr = "workflow param's "

        workflow_param.method = check_and_change_lower(workflow_param.method,
                                                       ['train', 'predict', 'cross_validation',
                                                        'intersect', 'binning', 'feature_select', 'one_vs_rest_train',
                                                        "one_vs_rest_predict"],
                                                       descr)

        if workflow_param.method in ['train', 'binning', 'feature_select']:
            if type(workflow_param.train_input_table).__name__ != "str":
                raise ValueError(
                    "workflow param's train_input_table {} not supported, should be str type".format(
                        workflow_param.train_input_table))

            if type(workflow_param.train_input_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's train_input_namespace {} not supported, should be str type".format(
                        workflow_param.train_input_namespace))

        if workflow_param.method in ["train", "predict", "cross_validation"]:
            if type(workflow_param.model_table).__name__ != "str":
                raise ValueError(
                    "workflow param's model_table {} not supported, should be str type".format(
                        workflow_param.model_table))

            if type(workflow_param.model_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's model_namespace {} not supported, should be str type".format(
                        workflow_param.model_namespace))

        if workflow_param.method == 'predict':
            if type(workflow_param.predict_input_table).__name__ != "str":
                raise ValueError(
                    "workflow param's predict_input_table {} not supported, should be str type".format(
                        workflow_param.predict_input_table))

            if type(workflow_param.predict_input_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's predict_input_namespace {} not supported, should be str type".format(
                        workflow_param.predict_input_namespace))

            if type(workflow_param.predict_output_table).__name__ != "str":
                raise ValueError(
                    "workflow param's predict_output_table {} not supported, should be str type".format(
                        workflow_param.predict_output_table))

            if type(workflow_param.predict_output_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's predict_output_namespace {} not supported, should be str type".format(
                        workflow_param.predict_output_namespace))

        if workflow_param.method in ["train", "predict", "cross_validation"]:
            if type(workflow_param.predict_result_partition).__name__ != "int":
                raise ValueError(
                    "workflow param's predict_result_partition {} not supported, should be int type".format(
                        workflow_param.predict_result_partition))

            if type(workflow_param.evaluation_output_table).__name__ != "str":
                raise ValueError(
                    "workflow param's evaluation_output_table {} not supported, should be str type".format(
                        workflow_param.evaluation_output_table))

            if type(workflow_param.evaluation_output_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's evaluation_output_namespace {} not supported, should be str type".format(
                        workflow_param.evaluation_output_namespace))

        if workflow_param.method == 'cross_validation':
            if type(workflow_param.data_input_table).__name__ != "str":
                raise ValueError(
                    "workflow param's data_input_table {} not supported, should be str type".format(
                        workflow_param.data_input_table))

            if type(workflow_param.data_input_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's data_input_namespace {} not supported, should be str type".format(
                        workflow_param.data_input_namespace))

            if type(workflow_param.n_splits).__name__ != "int":
                raise ValueError(
                    "workflow param's n_splits {} not supported, should be int type".format(
                        workflow_param.n_splits))
            elif workflow_param.n_splits <= 0:
                raise ValueError(
                    "workflow param's n_splits must be greater or equal to 1")

        if workflow_param.intersect_data_output_table is not None:
            if type(workflow_param.intersect_data_output_table).__name__ != "str":
                raise ValueError(
                    "workflow param's intersect_data_output_table {} not supported, should be str type".format(
                        workflow_param.intersect_data_output_table))

        if workflow_param.intersect_data_output_namespace is not None:
            if type(workflow_param.intersect_data_output_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's intersect_data_output_namespace {} not supported, should be str type".format(
                        workflow_param.intersect_data_output_namespace))

        DataIOParamChecker.check_param(workflow_param.dataio_param)

        if type(workflow_param.work_mode).__name__ != "int":
            raise ValueError(
                "workflow param's work_mode {} not supported, should be int type".format(
                    workflow_param.work_mode))
        elif workflow_param.work_mode not in [0, 1]:
            raise ValueError(
                "workflow param's work_mode must be 0 (represent to standalone mode) or 1 (represent to cluster mode)")

        if workflow_param.method in ["train", "predict", "cross_validation"]:
            PredictParamChecker.check_param(workflow_param.predict_param)
            EvaluateParamChecker.check_param(workflow_param.evaluate_param)

        if type(workflow_param.one_vs_rest).__name__ != "bool":
            raise ValueError(
                "workflow_param param's one_vs_rest {} not supported, should be bool type".format(
                    workflow_param.one_vs_rest))

        LOGGER.debug("Finish workerflow parameter check!")
        return True


class InitParamChecker(object):
    @staticmethod
    def check_param(init_param):
        if type(init_param.init_method).__name__ != "str":
            raise ValueError(
                "Init param's init_method {} not supported, should be str type".format(init_param.init_method))
        else:
            init_param.init_method = init_param.init_method.lower()
            if init_param.init_method not in ['random_uniform', 'random_normal', 'ones', 'zeros', 'const']:
                raise ValueError(
                    "Init param's init_method {} not supported, init_method should in 'random_uniform',"
                    " 'random_normal' 'ones', 'zeros' or 'const'".format(init_param.init_method))

        if type(init_param.init_const).__name__ not in ['int', 'float']:
            raise ValueError(
                "Init param's init_const {} not supported, should be int or float type".format(init_param.init_const))

        if type(init_param.fit_intercept).__name__ != 'bool':
            raise ValueError(
                "Init param's fit_intercept {} not supported, should be bool type".format(init_param.fit_intercept))

        LOGGER.debug("Finish init parameter check!")
        return True


class LogisticParamChecker(object):
    @staticmethod
    def check_param(logistic_param):
        descr = "logistic_param's"

        if type(logistic_param.penalty).__name__ != "str":
            raise ValueError(
                "logistic_param's penalty {} not supported, should be str type".format(logistic_param.penalty))
        else:
            logistic_param.penalty = logistic_param.penalty.upper()
            if logistic_param.penalty not in ['L1', 'L2', 'NONE']:
                raise ValueError(
                    "logistic_param's penalty not supported, penalty should be 'L1', 'L2' or 'none'")

        if type(logistic_param.eps).__name__ != "float":
            raise ValueError(
                "logistic_param's eps {} not supported, should be float type".format(logistic_param.eps))

        if type(logistic_param.alpha).__name__ != "float":
            raise ValueError(
                "logistic_param's alpha {} not supported, should be float type".format(logistic_param.alpha))

        if type(logistic_param.optimizer).__name__ != "str":
            raise ValueError(
                "logistic_param's optimizer {} not supported, should be str type".format(logistic_param.optimizer))
        else:
            logistic_param.optimizer = logistic_param.optimizer.lower()
            if logistic_param.optimizer not in ['sgd', 'rmsprop', 'adam', 'adagrad']:
                raise ValueError(
                    "logistic_param's optimizer not supported, optimizer should be"
                    " 'sgd', 'rmsprop', 'adam' or 'adagrad'")

        if type(logistic_param.batch_size).__name__ != "int":
            raise ValueError(
                "logistic_param's batch_size {} not supported, should be int type".format(logistic_param.batch_size))
        if logistic_param.batch_size != -1:
            if type(logistic_param.batch_size).__name__ not in ["int", "long"] \
                    or logistic_param.batch_size < consts.MIN_BATCH_SIZE:
                raise ValueError(descr + " {} not supported, should be larger than 10 or "
                                         "-1 represent for all data".format(logistic_param.batch_size))

        if type(logistic_param.learning_rate).__name__ != "float":
            raise ValueError(
                "logistic_param's learning_rate {} not supported, should be float type".format(
                    logistic_param.learning_rate))

        InitParamChecker.check_param(logistic_param.init_param)

        if type(logistic_param.max_iter).__name__ != "int":
            raise ValueError(
                "logistic_param's max_iter {} not supported, should be int type".format(logistic_param.max_iter))
        elif logistic_param.max_iter <= 0:
            raise ValueError(
                "logistic_param's max_iter must be greater or equal to 1")

        if type(logistic_param.converge_func).__name__ != "str":
            raise ValueError(
                "logistic_param's converge_func {} not supported, should be str type".format(
                    logistic_param.converge_func))
        else:
            logistic_param.converge_func = logistic_param.converge_func.lower()
            if logistic_param.converge_func not in ['diff', 'abs']:
                raise ValueError(
                    "logistic_param's converge_func not supported, converge_func should be"
                    " 'diff' or 'abs'")

        EncryptParamChecker.check_param(logistic_param.encrypt_param)

        if type(logistic_param.re_encrypt_batches).__name__ != "int":
            raise ValueError(
                "logistic_param's re_encrypt_batches {} not supported, should be int type".format(
                    logistic_param.re_encrypt_batches))
        elif logistic_param.re_encrypt_batches < 0:
            raise ValueError(
                "logistic_param's re_encrypt_batches must be greater or equal to 0")

        if type(logistic_param.model_path).__name__ != "str":
            raise ValueError(
                "logistic_param's model_path {} not supported, should be str type".format(
                    logistic_param.model_path))

        if type(logistic_param.table_name).__name__ != "str":
            raise ValueError(
                "logistic_param's table_name {} not supported, should be str type".format(
                    logistic_param.table_name))

        if type(logistic_param.party_weight).__name__ not in ["int", 'float']:
            raise ValueError(
                "logistic_param's party_weight {} not supported, should be 'int' or 'float'".format(
                    logistic_param.party_weight))

        LOGGER.debug("Finish logistic parameter check!")
        return True


class FeatureBinningParamChecker(object):
    @staticmethod
    def check_param(binning_param):
        descr = "hetero binning param's"
        check_string(binning_param.method, descr)
        binning_param.method = binning_param.method.lower()
        check_valid_value(binning_param.method, descr, [consts.QUANTILE])
        check_positive_integer(binning_param.compress_thres, descr)
        check_positive_integer(binning_param.head_size, descr)
        check_decimal_float(binning_param.error, descr)
        check_positive_integer(binning_param.bin_num, descr)
        check_defined_type(binning_param.cols, descr, ['list', 'int', 'RepeatedScalarContainer'])
        check_open_unit_interval(binning_param.adjustment_factor, descr)
        # check_string(binning_param.result_table, descr)
        # check_string(binning_param.result_namespace, descr)
        check_defined_type(binning_param.display_result, descr, ['list'])
        for idx, d_s in enumerate(binning_param.display_result):
            binning_param.display_result[idx] = check_and_change_lower(d_s,
                                                                       ['iv', 'woe_array', 'iv_array',
                                                                        'event_count_array', 'non_event_count_array',
                                                                        'event_rate_array', 'bin_nums', 'split_points',
                                                                        'non_event_rate_array', 'is_woe_monotonic'],
                                                                       descr)


class FeatureSelectionParamChecker(object):
    @staticmethod
    def check_param(feature_param):
        descr = "hetero feature selection param's"
        feature_param.method = check_and_change_lower(feature_param.method,
                                                      ['fit', 'fit_transform', 'transform'], descr)
        check_defined_type(feature_param.filter_method, descr, ['list'])

        for idx, method in enumerate(feature_param.filter_method):
            method = method.lower()
            check_valid_value(method, descr, ["unique_value", "iv_value_thres", "iv_percentile",
                                              "coefficient_of_variation_value_thres",
                                              "outlier_cols"])
            feature_param.filter_method[idx] = method
        if "iv_value_thres" in feature_param.filter_method and "iv_percentile" in feature_param.filter_method:
            raise ValueError("Two iv methods should not exist at the same time.")

        check_defined_type(feature_param.select_cols, descr, ['list', 'int'])

        check_boolean(feature_param.local_only, descr)
        UniqueValueParamChecker.check_param(feature_param.unique_param)
        IVValueSelectionParamChecker.check_param(feature_param.iv_value_param)
        IVPercentileSelectionParamChecker.check_param(feature_param.iv_percentile_param)
        CoeffOfVarSelectionParamChecker.check_param(feature_param.coe_param)
        OutlierColsSelectionParamChecker.check_param(feature_param.outlier_param)
        FeatureBinningParamChecker.check_param(feature_param.bin_param)
        return True


class UniqueValueParamChecker(object):
    @staticmethod
    def check_param(feature_param):
        descr = "Unique value param's"
        check_positive_number(feature_param.eps, descr)
        return True


class IVValueSelectionParamChecker(object):
    @staticmethod
    def check_param(feature_param):
        descr = "IV selection param's"
        check_positive_number(feature_param.value_threshold, descr)
        return True


class IVPercentileSelectionParamChecker(object):
    @staticmethod
    def check_param(feature_param):
        descr = "IV selection param's"
        check_decimal_float(feature_param.percentile_threshold, descr)
        return True


class CoeffOfVarSelectionParamChecker(object):
    @staticmethod
    def check_param(feature_param):
        descr = "Coff of Variances param's"
        check_positive_number(feature_param.value_threshold, descr)
        return True


class OutlierColsSelectionParamChecker(object):
    @staticmethod
    def check_param(feature_param):
        descr = "Outlier Filter param's"
        check_decimal_float(feature_param.percentile, descr)
        check_defined_type(feature_param.upper_threshold, descr, ['float', 'int'])
        return True


class OneHotEncoderParamChecker(object):
    @staticmethod
    def check_param(param):
        descr = "One-hot encoder param's"
        check_defined_type(param.cols, descr, ['list', 'int'])
        return True


class FTLModelParamChecker(object):
    @staticmethod
    def check_param(ftl_model_param):
        model_param_descr = "ftl model param's "
        check_positive_integer(ftl_model_param.max_iter, model_param_descr + "max_iter")
        check_positive_number(ftl_model_param.eps, model_param_descr + "eps")
        check_positive_number(ftl_model_param.alpha, model_param_descr + "alpha")
        check_boolean(ftl_model_param.is_encrypt, model_param_descr + "is_encrypt")
        return True


class LocalModelParamChecker(object):
    @staticmethod
    def check_param(local_model_param):
        model_param_descr = "local model param's "
        if local_model_param.input_dim is not None:
            check_positive_integer(local_model_param.input_dim, model_param_descr + "input_dim")
        check_positive_integer(local_model_param.encode_dim, model_param_descr + "encode_dim")
        check_open_unit_interval(local_model_param.learning_rate, model_param_descr + "learning_rate")
        return True


class FTLDataParamChecker(object):
    @staticmethod
    def check_param(ftl_data_param):
        model_param_descr = "ftl data model param's "
        if ftl_data_param.file_path is not None:
            check_string(ftl_data_param.file_path, model_param_descr + "file_path")
        if ftl_data_param.num_samples is not None:
            check_positive_integer(ftl_data_param.num_samples, model_param_descr + "num_samples")

        check_positive_integer(ftl_data_param.n_feature_guest, model_param_descr + "n_feature_guest")
        check_positive_integer(ftl_data_param.n_feature_host, model_param_descr + "n_feature_host")
        check_boolean(ftl_data_param.balanced, model_param_descr + "balanced")
        check_boolean(ftl_data_param.is_read_table, model_param_descr + "is_read_table")
        check_open_unit_interval(ftl_data_param.overlap_ratio, model_param_descr + "overlap_ratio")
        check_open_unit_interval(ftl_data_param.guest_split_ratio, model_param_descr + "guest_split_ratio")
        return True


class FTLValidDataParamChecker(object):
    @staticmethod
    def check_param(ftl_valid_data_param):
        model_param_descr = "ftl validation data model param's "
        if ftl_valid_data_param.file_path is not None:
            check_string(ftl_valid_data_param.file_path, model_param_descr + "file_path")
        if ftl_valid_data_param.num_samples is not None:
            check_positive_integer(ftl_valid_data_param.num_samples, model_param_descr + "num_samples")

        check_boolean(ftl_valid_data_param.is_read_table, model_param_descr + "is_read_table")
        return True


class ScaleParamChecker(object):
    @staticmethod
    def check_param(scale_param):
        if scale_param.method is not None:
            descr = "scale param's method"
            scale_param.method = check_and_change_lower(scale_param.method,
                                                        [consts.MINMAXSCALE, consts.STANDARDSCALE],
                                                        descr)

        descr = "scale param's mode"
        scale_param.mode = check_and_change_lower(scale_param.mode,
                                                  [consts.NORMAL, consts.CAP],
                                                  descr)

        descr = "scale param's area"
        scale_param.area = check_and_change_lower(scale_param.area,
                                                  [consts.ALL, consts.COL],
                                                  descr)
        if scale_param.area == consts.ALL:
            if scale_param.feat_lower is not None:
                if type(scale_param.feat_lower).__name__ not in ["float", "int"]:
                    raise ValueError(
                        "scale param's feat_lower {} not supported, should be float or int type".format(
                            scale_param.feat_lower))

            if scale_param.feat_upper is not None:
                if type(scale_param.feat_upper).__name__ not in ["float", "int"]:
                    raise ValueError(
                        "scale param's feat_upper {} not supported, should be float or int type".format(
                            scale_param.feat_upper))

            if scale_param.out_lower is not None:
                if type(scale_param.out_lower).__name__ not in ["float", "int"]:
                    raise ValueError(
                        "scale param's out_lower {} not supported, should be float or int type".format(
                            scale_param.out_lower))

            if scale_param.out_upper is not None:
                if type(scale_param.out_upper).__name__ not in ["float", "int"]:
                    raise ValueError(
                        "scale param's out_upper {} not supported, should be float or int type".format(
                            scale_param.out_upper))
        elif scale_param.area == consts.COL:
            descr = "scale param's feat_lower"
            check_defined_type(scale_param.feat_lower, descr, ['list'])

            descr = "scale param's feat_upper"
            check_defined_type(scale_param.feat_upper, descr, ['list'])

            descr = "scale param's out_lower"
            check_defined_type(scale_param.out_lower, descr, ['list'])

            descr = "scale param's out_upper"
            check_defined_type(scale_param.out_upper, descr, ['list'])

        check_boolean(scale_param.with_mean, "scale_param with_mean")
        check_boolean(scale_param.with_std, "scale_param with_std")

        LOGGER.debug("Finish scale parameter check!")
        return True


def check_string(param, descr):
    if type(param).__name__ not in ["str"]:
        raise ValueError(descr + " {} not supported, should be string type".format(param))


def check_positive_integer(param, descr):
    if type(param).__name__ not in ["int", "long"] or param <= 0:
        raise ValueError(descr + " {} not supported, should be positive integer".format(param))


def check_positive_number(param, descr):
    if type(param).__name__ not in ["float", "int", "long"] or param <= 0:
        raise ValueError(descr + " {} not supported, should be positive numeric".format(param))


def check_decimal_float(param, descr):
    if type(param).__name__ not in ["float"] or param < 0 or param > 1:
        raise ValueError(descr + " {} not supported, should be a float number in range [0, 1]".format(param))


def check_boolean(param, descr):
    if type(param).__name__ != "bool":
        raise ValueError(descr + " {} not supported, should be bool type".format(param))


def check_open_unit_interval(param, descr):
    if type(param).__name__ not in ["float"] or param <= 0 or param >= 1:
        raise ValueError(descr + " should be a numeric number between 0 and 1 exclusively")


def check_valid_value(param, descr, valid_values):
    if param not in valid_values:
        raise ValueError(descr + " {} is not supported, it should be in {}".format(param, valid_values))


def check_defined_type(param, descr, types):
    if type(param).__name__ not in types:
        raise ValueError(descr + " {} not supported, should be one of {}".format(param, types))


# Used when param is a string.
def check_and_change_lower(param, valid_list, descr=''):
    if type(param).__name__ != 'str':
        raise ValueError(descr + " {} not supported, should be one of {}".format(param, valid_list))

    lower_param = param.lower()
    if lower_param in valid_list:
        return lower_param
    else:
        raise ValueError(descr + " {} not supported, should be one of {}".format(param, valid_list))


class AllChecker(object):
    def __init__(self, config_path, param_restricted_path=None):
        self.config_path = config_path
        self.param_restricted_path = param_restricted_path
        self.func = {"ge": self._greater_equal_than,
                     "le": self._less_equal_than,
                     "in": self._in,
                     "not_in": self._not_in,
                     "range": self._range
                     }

    def check_all(self):
        self._check(param.DataIOParam, DataIOParamChecker)
        self._check(param.EncryptParam, EncryptParamChecker)
        self._check(param.EncryptedModeCalculatorParam, EncryptedModeCalculatorParamChecker)
        self._check(param.SampleParam, SampleParamChecker)
        self._check(param.EvaluateParam, EvaluateParamChecker)
        self._check(param.ObjectiveParam, ObjectiveParamChecker)
        self._check(param.PredictParam, PredictParamChecker)
        self._check(param.WorkFlowParam, WorkFlowParamChecker)
        self._check(param.InitParam, InitParamChecker)
        self._check(param.EncodeParam, EncodeParamChecker)
        self._check(param.IntersectParam, IntersectParamChecker)
        self._check(param.LogisticParam, LogisticParamChecker)
        self._check(param.DecisionTreeParam, DecisionTreeParamChecker)
        self._check(param.BoostingTreeParam, BoostingTreeParamChecker)
        self._check(param.FTLModelParam, FTLModelParamChecker)
        self._check(param.LocalModelParam, LocalModelParamChecker)
        self._check(param.FTLDataParam, FTLDataParamChecker)
        self._check(param.FTLValidDataParam, FTLValidDataParamChecker)
        self._check(param.FeatureBinningParam, FeatureBinningParamChecker)
        self._check(param.FeatureSelectionParam, FeatureSelectionParamChecker)
        self._check(param.ScaleParam, ScaleParamChecker)
        self._check(param.OneVsRestParam, OneVsRestChecker)

    def _check(self, Param, Checker):
        """
        check if parameters define in Param Ojbect is valid or not.
            validity of parameters decide by the following two ways:
                1. match the definition in ParamObject, which will be check in checker
                2. match the param restriction of user definition, define in workflow/conf/param_validation.json

        Parameters
        ----------  
        Param: object, define in federatedml/param/param.py

        Checker: object, define in this module, see above

        """

        param_obj = Param()
        param_obj = ParamExtract.parse_param_from_config(param_obj, self.config_path)
        Checker.check_param(param_obj)

        if self.param_restricted_path is not None:
            with open(self.param_restricted_path, "r") as fin:
                validation_json = json.loads(fin.read())

            param_classes = [class_info[0] for class_info in inspect.getmembers(param, inspect.isclass)]
            self.validate_restricted_param(param_obj, validation_json, param_classes)

    def validate_restricted_param(self, param_obj, validation_json, param_classes):
        """
        Validate the param restriction of user definition recursively.
            It will only validation parameters define both in param_obj and validation_json

        Parameters
        ---------- 
        param_obj: object, parameter object define in federatedml/param/param.py

        validation_json: dict, parameter restriction of user-define.

        param_classes: list, all object define in federatedml/param/param.py
  
        """

        default_section = type(param_obj).__name__
        var_list = param_obj.__dict__

        for variable in var_list:
            attr = getattr(param_obj, variable)
            if type(attr).__name__ in param_classes:
                self.validate_restricted_param(attr, validation_json, param_classes)
            else:
                if default_section in validation_json and variable in validation_json[default_section]:
                    validation_dict = validation_json[default_section][variable]
                    value = getattr(param_obj, variable)
                    value_legal = False

                    for op_type in validation_dict:
                        if self.func[op_type](value, validation_dict[op_type]):
                            value_legal = True
                            break

                    if not value_legal:
                        raise ValueError(
                            "Plase check runtime conf, {} = {} does not match user-parameter restriction".format(
                                variable, value))

    def _greater_equal_than(self, value, limit):
        return value >= limit - consts.FLOAT_ZERO

    def _less_equal_than(self, value, limit):
        return value <= limit + consts.FLOAT_ZERO

    def _range(self, value, ranges):
        in_range = False
        for left_limit, right_limit in ranges:
            if value >= left_limit - consts.FLOAT_ZERO and value <= right_limit + consts.FLOAT_ZERO:
                in_range = True
                break

        return in_range

    def _in(self, value, right_value_list):
        return value in right_value_list

    def _not_in(self, value, wrong_value_list):
        return value not in wrong_value_list
