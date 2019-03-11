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
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class DataIOParamChecker(object):
    @staticmethod
    def check_param(dataio_param):
        if type(dataio_param).__name__ != "DataIOParam":
            raise ValueError("dataio_param {} not supported, should be DataIOParam object".format(dataio_param))

        if dataio_param.input_format not in ["dense", "sparse"]:
            raise ValueError("dataio param's input format {} not supported".format(dataio_param.input_format))

        if dataio_param.output_format not in ["dense", "sparse"]:
            raise ValueError("dataio param's output format {} not supported".format(dataio_param.output_format))

        if dataio_param.data_type not in ["int", "int64", "float", "float64", "str", "long"]:
            raise ValueError("dataio param's data_type {} not supported".format(dataio_param.data_type))

        if type(dataio_param.missing_fill).__name__ != 'bool':
            raise ValueError("dataio param's missing_fill {} not supported".format(dataio_param.missing_fill))

        if type(dataio_param.with_label).__name__ != 'bool':
            raise ValueError("dataio param's with_label {} not supported".format(dataio_param.with_label))

        if dataio_param.with_label:
            if type(dataio_param.label_idx).__name__ not in ["long", "int"]:
                raise ValueError("dataio param's label_idx {} not supported".format(dataio_param.label_idx))

            if dataio_param.label_type not in ["int", "int64", "float", "float64", "str", "long"]:
                raise ValueError("dataio param's label_type {} not supported".format(dataio_param.label_type))

        return True


class ObjectiveParamChecker(object):
    @staticmethod
    def check_param(objective_param, task_type=None):
        if type(objective_param).__name__ != "ObjectiveParam":
            raise ValueError("objective param {} not supportd, should be ObjectiveParam object".format(objective_param))

        if objective_param.objective is None:
            return True

        if task_type not in [consts.CLASSIFICATION, consts.REGRESSION]:
            if objective_param.objective not in ["cross_entropy", "lse", "lae", "huber", "fair", "log_cosh", "tweedie"]:
                raise ValueError("objective param's objective {} not supported".format(objective_param.objective))

        if task_type == consts.CLASSIFICATION:
            if objective_param.objective != "cross_entropy":
                raise ValueError("objective param's objective {} not supported".format(objective_param.objective))

        elif task_type == consts.REGRESSION:
            if objective_param.objective not in ["lse", "lae", "huber", "fair", "log_cosh", "tweedie"]:
                raise ValueError("objective param's objective {} not supported".format(objective_param.objective))

            params = objective_param.params
            if objective_param.objective in ["huber", "fair", "tweedie"]:
                if type(params).__name__ != 'list' or len(params) < 1:
                    raise ValueError("objective param's params {} not supported, should be non-empty list".format(params))

                if type(params[0]).__name__ not in ["float", "int", "long"]:
                    raise ValueError("objective param's params[0] {} not supported".format(objective_param.params[0]))

                if objective_param.objective == 'tweedie':
                    if params[0] < 1 or params[0] >= 2:
                        raise ValueError("in tweedie regression, objective params[0] should betweend [1, 2)")

                if objective_param.objective == 'fair' or 'huber':
                    if params[0] <= 0.0:
                        raise ValueError("in {} regression, objective params[0] should greater than 0.0".format(objective_param.objective))
                
        return True


class EncryptParamChecker(object):
    @staticmethod
    def check_param(encrypt_param):
        if type(encrypt_param.method).__name__ != "str":
            raise ValueError(
                "encrypt_param's method {] not supported, should be str type".format(
                    encrypt_param.method))
        else:
            user_input = encrypt_param.method.lower()
            if user_input == 'paillier':
                encrypt_param.method = 'Paillier'
            # else:
            #     raise ValueError(
            #         "encode encrypt_param's method {} not supported, Supported 'Paillier' only")

        if type(encrypt_param.key_length).__name__ != "int":
            raise ValueError(
                "encrypt_param's key_length {} not supported, should be int type".format(encrypt_param.key_length))
        elif encrypt_param.key_length <= 0:
            raise ValueError(
                "encrypt_param's key_length must be greater or equal to 1")

        LOGGER.debug("Finish encrypt parameter check!")
        return True


class DecisionTreeParamChecker(object):
    @staticmethod
    def check_param(tree_param):
        if type(tree_param).__name__ != "DecisionTreeParam":
            raise ValueError(
                "decision tree param {} not supported, should be DecisionTreeParam object".format(tree_param))

        if tree_param.criterion_method not in ["xgboost"]:
            raise ValueError(
                "decision tree param's criterion_method {} not supported, now just supported xgboost".format(
                    tree_param.criterion_method))

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
            raise ValueError("decision tree param's n_iter_no_change {] not supported, should be bool type".format(
                tree_param.n_iter_no_change))

        if type(tree_param.tol).__name__ not in ["float", "int", "long"]:
            raise ValueError("decision tree param's tol {} not supported, should be numeric".format(tree_param.tol))

        return True


class BoostingTreeParamChecker(object):
    @staticmethod
    def check_param(boost_param):
        DecisionTreeParamChecker.check_param(boost_param.tree_param)

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
            raise ValueError("boosting tree param's n_iter_no_change {] not supported, should be bool type".format(
                boost_param.n_iter_no_change))

        if type(boost_param.tol).__name__ not in ["float", "int", "long"]:
            raise ValueError("boosting tree param's tol {} not supported, should be numeric".format(boost_param.tol))

        EncryptParamChecker.check_param(boost_param.encrypt_param)

        if boost_param.quantile_method not in ["bin_by_data_block", "bin_by_sample_data"]:
            raise ValueError(
                "boosting tree param's quantile_method {} not supported, should be bin_by_data_block/bin_by_sample_data")

        if type(boost_param.bin_num).__name__ not in ["int", "long"] or boost_param.bin_num < 2:
            raise ValueError(
                "boosting tree param's bin_num {} not supported, should be positive integer greater than 1".format(boost_param.bin_num))

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
                "encode param's salt {] not supported, should be str type".format(
                    encode_param.salt))

        if encode_param.encode_method not in ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"]:
            raise ValueError(
                "encode param's encode_method {} not supported, now just supported md5, sha1, sha224, sha256, sha384, sha512".format(
                    encode_param.encode_method))

        if type(encode_param.base64).__name__ != "bool":
            raise ValueError(
                "encode param's base64 {] not supported, should be bool type".format(encode_param.base64))

        LOGGER.debug("Finish encode parameter check!")
        return True

class IntersectParamChecker(object):
    @staticmethod
    def check_param(intersect_param):
        if intersect_param.intersect_method not in [consts.RSA, consts.RAW]:
            raise ValueError(
                "intersect param's intersect_method {} not supported, now just supported rsa and raw".format(
                    intersect_param.intersect_method))

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

        if intersect_param.join_role not in [consts.GUEST, consts.HOST]:
            raise ValueError(
                "intersect param's join_role {} not supported, now just supported guest and host".format(
                    intersect_param.join_role))

        if type(intersect_param.with_encode).__name__ != "bool":
            raise ValueError(
                "intersect param's with_encode {} not supported, should be bool type".format(
                    intersect_param.with_encode))

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
                predict_param.cross_validation))
        
        LOGGER.debug("Finish predict parameter check!")
        return True


class EvaluateParamChecker(object):
    @staticmethod
    def check_param(evaluate_param):
        if type(evaluate_param.metrics).__name__ != "list":
            raise ValueError("evaluate param's metrics {} not supported, should be list".format(evaluate_param.metrics))
        else:
            for metric in evaluate_param.metrics:
                if metric not in [consts.AUC, consts.KS, consts.LIFT, consts.PRECISION, consts.RECALL, consts.ACCURACY,
                                  consts.EXPLAINED_VARIANCE,
                                  consts.MEAN_ABSOLUTE_ERROR, consts.MEAN_SQUARED_ERROR, consts.MEAN_SQUARED_LOG_ERROR,
                                  consts.MEDIAN_ABSOLUTE_ERROR,
                                  consts.R2_SCORE, consts.ROOT_MEAN_SQUARED_ERROR]:
                    raise ValueError("evaluate param's metrics {} not supported".format(metric))

        if evaluate_param.classi_type not in [consts.BINARY, consts.MULTY, consts.REGRESSION]:
            raise ValueError(
                "evaluate param's classi_type {} not supported, now just supported binary, multi and regression".format(
                    evaluate_param.classi_type))

        if type(evaluate_param.pos_label).__name__ not in [ "str", "float", "int" ]:
            raise ValueError(
                "evaluate param's pos_label {} not supported, should be str or float or int type".format(evaluate_param.pos_label))

        if type(evaluate_param.thresholds).__name__ != "list":
            raise ValueError("evaluate param's thresholds {} not supported, should be list".format(evaluate_param.thresholds))
        else:
            for threshold in evaluate_param.thresholds:
                if type(threshold).__name__ not in ["float", "int"]:
                    raise ValueError("threshold {} in evaluate param's thresholds not supported, should be positive integer".format(thresholds))

        LOGGER.debug("Finish evaluation parameter check!")
        return True


class WorkFlowParamChecker(object):
    @staticmethod
    def check_param(workflow_param):
        if type(workflow_param.method).__name__ != "str":
            raise ValueError(
                "workflow param's method {} not supported, should be str type".format(workflow_param.method))
        elif workflow_param.method not in ['train', 'predict', 'cross_validation', 'intersect']:
            raise ValueError("workflow param's method {} not supported".format(workflow_param.method))

        if workflow_param.method == 'train':
            if type(workflow_param.train_input_table).__name__ != "str":
                raise ValueError(
                    "workflow param's train_input_table {} not supported, should be str type".format(
                        workflow_param.train_input_table))

            if type(workflow_param.train_input_namespace).__name__ != "str":
                raise ValueError(
                    "workflow param's train_input_namespace {} not supported, should be str type".format(
                        workflow_param.train_input_namespace))
        
        if workflow_param.method in [ "train", "predict", "cross_validation"]:
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

        if workflow_param.method in [ "train", "predict", "cross_validation"]:
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

        if type(workflow_param.n_splits).__name__ != "int":
            raise ValueError(
                "workflow param's n_splits {} not supported, should be int type".format(
                    workflow_param.n_splits))
        elif workflow_param.n_splits <= 0:
            raise ValueError(
                "workflow param's n_splits must be greater or equal to 1")

        if type(workflow_param.work_mode).__name__ != "int":
            raise ValueError(
                "workflow param's work_mode {} not supported, should be int type".format(
                    workflow_param.work_mode))
        elif workflow_param.work_mode not in [0, 1]:
            raise ValueError(
                "workflow param's work_mode must be 0 (represent to standalone mode) or 1 (represent to cluster mode)")
        

        if workflow_param.method in [ "train", "predict", "cross_validation"]:
            PredictParamChecker.check_param(workflow_param.predict_param)
            EvaluateParamChecker.check_param(workflow_param.evaluate_param)
        
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


class FTLModelParamChecker(object):

    @staticmethod
    def check_param(ftl_model_param):
        model_param_descr = "ftl model param's "
        check_positive_integer(ftl_model_param.max_iter, model_param_descr + "max_iter")
        check_positive_number(ftl_model_param.eps, model_param_descr + "eps")
        check_positive_number(ftl_model_param.alpha, model_param_descr + "alpha")
        check_boolean(ftl_model_param.is_encrypt, model_param_descr + "is_encrypt")
        return True


class FTLLocalModelParamChecker(object):

    @staticmethod
    def check_param(ftl_local_model_param):
        model_param_descr = "ftl local model param's "
        check_positive_integer(ftl_local_model_param.encode_dim, model_param_descr + "encode_dim")
        check_open_unit_interval(ftl_local_model_param.learning_rate, model_param_descr + "learning_rate")
        return True


class FTLDataParamChecker(object):

    @staticmethod
    def check_param(ftl_data_param):
        model_param_descr = "ftl data model param's "
        check_string(ftl_data_param.file_path, model_param_descr + "file_path")
        check_positive_integer(ftl_data_param.n_feature_guest, model_param_descr + "n_feature_guest")
        check_positive_integer(ftl_data_param.n_feature_host, model_param_descr + "n_feature_host")
        check_positive_integer(ftl_data_param.num_samples, model_param_descr + "num_samples")
        check_boolean(ftl_data_param.balanced, model_param_descr + "balanced")
        check_boolean(ftl_data_param.is_read_table, model_param_descr + "is_read_table")
        check_open_unit_interval(ftl_data_param.overlap_ratio, model_param_descr + "overlap_ratio")
        check_open_unit_interval(ftl_data_param.guest_split_ratio, model_param_descr + "guest_split_ratio")
        return True


# class FTLValidDataParamChecker(object):
#
#     @staticmethod
#     def check_param(ftl_valid_data_param):
#         model_param_descr = "ftl validation data model param's "
#         check_string(ftl_valid_data_param.file_path, model_param_descr + "file_path")
#         check_positive_integer(ftl_valid_data_param.num_samples, model_param_descr + "num_samples")
#         check_boolean(ftl_valid_data_param.is_read_table, model_param_descr + "is_read_table")
#         return True


def check_string(param, descr):
    if type(param).__name__ not in ["str"]:
        raise ValueError(descr + " {} not supported, should be string type".format(param))


def check_positive_integer(param, descr):
    if type(param).__name__ not in ["int", "long"] or param <= 0:
        raise ValueError(descr + " {} not supported, should be positive integer".format(param))


def check_positive_number(param, descr):
    if type(param).__name__ not in ["float", "int", "long"] or param <= 0:
        raise ValueError(descr + " {} not supported, should be positive numeric".format(param))


def check_boolean(param, descr):
    if type(param).__name__ != "bool":
        raise ValueError(descr + " {} not supported, should be bool type".format(param))


def check_open_unit_interval(param, descr):
    if type(param).__name__ not in ["float"] or param <= 0 or param >= 1:
        raise ValueError(descr + " should be a numeric number between 0 and 1 exclusively")
