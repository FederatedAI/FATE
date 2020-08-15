from abc import ABC
import abc

from numpy import random
import numpy as np

from federatedml.param.boosting_param import BoostingParam
from federatedml.param.feature_binning_param import FeatureBinningParam

from federatedml.model_selection.k_fold import KFold
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.util.validation_strategy import ValidationStrategy
from federatedml.feature.sparse_vector import SparseVector
from federatedml.model_base import ModelBase
from federatedml.feature.fate_element_type import NoneType
from federatedml.ensemble.basic_algorithms import BasicAlgorithms

from federatedml.loss import FairLoss
from federatedml.loss import HuberLoss
from federatedml.loss import LeastAbsoluteErrorLoss
from federatedml.loss import LeastSquaredErrorLoss
from federatedml.loss import LogCoshLoss
from federatedml.loss import SigmoidBinaryCrossEntropyLoss
from federatedml.loss import SoftmaxCrossEntropyLoss
from federatedml.loss import TweedieLoss

from federatedml.param.evaluation_param import EvaluateParam

from federatedml.ensemble.boosting.boosting_core.predict_cache import PredictDataCache

from federatedml.statistic import data_overview

from federatedml.optim.convergence import converge_func_factory

from arch.api.utils import log_utils

import functools
import typing

LOGGER = log_utils.getLogger()


class Boosting(ModelBase, ABC):
    def __init__(self):

        super(Boosting, self).__init__()

        # input hyper parameter
        self.task_type = None
        self.objective_param = None
        self.learning_rate = None
        self.boosting_round = None
        self.subsample_feature_rate = None
        self.n_iter_no_change = None
        self.early_stopping = -1
        self.tol = 0.0
        self.bin_num = None
        self.calculated_mode = None
        self.predict_param = None
        self.cv_param = None
        self.validation_freqs = None
        self.feature_name_fid_mapping = {}
        self.mode = None
        self.model_param = BoostingParam()
        self.subsample_feature_rate = 1.0
        self.model_name = 'default'  # model name
        self.early_stopping_rounds = None
        self.use_first_metric_only = False

        # running variable

        # data
        self.binning_class = None  # class used for data binning
        self.binning_obj = None  # instance of self.binning_class
        self.data_bin = None  # data with transformed features
        self.bin_split_points = None  # feature split points
        self.bin_sparse_points = None  # feature sparse points
        self.use_missing = False  # should handle missing value or not
        self.zero_as_missing = False  # set missing value as value or not

        # booster
        self.booster_dim = 1  # booster dimension
        self.booster_meta = None  # booster's hyper parameters
        self.boosting_model_list = []  # list hol\ds boosters

        # training
        self.feature_num = None  # feature number
        self.init_score = None  # init score
        self.num_classes = 2  # number of classes
        self.convergence = None  # function to check loss convergence
        self.classes_ = []  # class indices
        self.y = None   # label
        self.y_hat = None  # accumulated predict value
        self.loss = None   # loss func
        self.predict_y_hat = None  # accumulated predict value for predicting mode
        self.history_loss = []  # list holds loss history
        self.validation_strategy = None
        self.metrics = None
        self.is_converged = False

        # cache and header alignment
        self.predict_data_cache = PredictDataCache()
        self.data_alignment_map = {}

        # federation
        self.transfer_variable = None

    def _init_model(self, boosting_param: BoostingParam):
        self.task_type = boosting_param.task_type
        self.objective_param = boosting_param.objective_param
        self.learning_rate = boosting_param.learning_rate
        self.boosting_round = boosting_param.num_trees
        self.n_iter_no_change = boosting_param.n_iter_no_change
        self.tol = boosting_param.tol
        self.bin_num = boosting_param.bin_num
        self.predict_param = boosting_param.predict_param
        self.cv_param = boosting_param.cv_param
        self.validation_freqs = boosting_param.validation_freqs
        self.metrics = boosting_param.metrics

    @staticmethod
    def data_format_transform(row):
        if type(row.features).__name__ != consts.SPARSE_VECTOR:
            feature_shape = row.features.shape[0]
            indices = []
            data = []

            for i in range(feature_shape):
                if np.isnan(row.features[i]):
                    indices.append(i)
                    data.append(NoneType())
                elif np.abs(row.features[i]) < consts.FLOAT_ZERO:
                    continue
                else:
                    indices.append(i)
                    data.append(row.features[i])

            row.features = SparseVector(indices, data, feature_shape)
        else:
            sparse_vec = row.features.get_sparse_vector()
            for key in sparse_vec:
                if sparse_vec.get(key) == NoneType() or np.isnan(sparse_vec.get(key)):
                    sparse_vec[key] = NoneType()

            row.features.set_sparse_vector(sparse_vec)

        return row

    def init_validation_strategy(self, train_data=None, validate_data=None):
        validation_strategy = ValidationStrategy(self.role, self.mode, self.validation_freqs,
                                                 self.early_stopping_rounds, self.use_first_metric_only)
        validation_strategy.set_train_data(train_data)
        validation_strategy.set_validate_data(validate_data)
        return validation_strategy

    def cross_validation(self, data_instances):
        if not self.need_run:
            return data_instances
        kflod_obj = KFold()
        cv_param = self._get_cv_param()
        kflod_obj.run(cv_param, data_instances, self, host_do_evaluate=False)
        return data_instances

    def convert_feature_to_bin(self, data_instance, handle_missing_value=False):
        LOGGER.info("convert feature to bins")
        param_obj = FeatureBinningParam(bin_num=self.bin_num)

        if handle_missing_value:
            self.binning_obj = self.binning_class(param_obj, abnormal_list=[NoneType()])
        else:
            self.binning_obj = self.binning_class(param_obj)

        self.binning_obj.fit_split_points(data_instance)
        LOGGER.info("convert feature to bins over")
        return self.binning_obj.convert_feature_to_bin(data_instance)

    def sample_valid_features(self):
        LOGGER.info("sample valid features")

        self.feature_num = self.bin_split_points.shape[0]
        choose_feature = random.choice(range(0, self.feature_num), \
                                       max(1, int(self.subsample_feature_rate * self.feature_num)), replace=False)

        valid_features = [False for i in range(self.feature_num)]
        for fid in choose_feature:
            valid_features[fid] = True
        return valid_features

    def get_loss_function(self):
        loss_type = self.objective_param.objective
        params = self.objective_param.params
        LOGGER.info("set objective, objective is {}".format(loss_type))
        if self.task_type == consts.CLASSIFICATION:
            if loss_type == "cross_entropy":
                if self.num_classes == 2:
                    loss_func = SigmoidBinaryCrossEntropyLoss()
                else:
                    loss_func = SoftmaxCrossEntropyLoss()
            else:
                raise NotImplementedError("objective %s not supported yet" % (loss_type))
        elif self.task_type == consts.REGRESSION:
            if loss_type == "lse":
                loss_func = LeastSquaredErrorLoss()
            elif loss_type == "lae":
                loss_func = LeastAbsoluteErrorLoss()
            elif loss_type == "huber":
                loss_func = HuberLoss(params[0])
            elif loss_type == "fair":
                loss_func = FairLoss(params[0])
            elif loss_type == "tweedie":
                loss_func = TweedieLoss(params[0])
            elif loss_type == "log_cosh":
                loss_func = LogCoshLoss()
            else:
                raise NotImplementedError("objective %s not supported yet" % (loss_type))
        else:
            raise NotImplementedError("objective %s not supported yet" % (loss_type))

        return loss_func

    def get_metrics_param(self):
        if self.task_type == consts.CLASSIFICATION:
            if self.num_classes == 2:
                return EvaluateParam(eval_type="binary",
                                     pos_label=self.classes_[1], metrics=self.metrics)
            else:
                return EvaluateParam(eval_type="multi", metrics=self.metrics)
        else:
            return EvaluateParam(eval_type="regression", metrics=self.metrics)

    def compute_loss(self, y_hat, y):
        LOGGER.info("compute loss")
        if self.task_type == consts.CLASSIFICATION:
            loss_method = self.loss
            y_predict = y_hat.mapValues(lambda val: loss_method.predict(val))
            loss = loss_method.compute_loss(y, y_predict)
        elif self.task_type == consts.REGRESSION:
            if self.objective_param.objective in ["lse", "lae", "logcosh", "tweedie", "log_cosh", "huber"]:
                loss_method = self.loss
                loss = loss_method.compute_loss(y, y_hat)
            else:
                loss_method = self.loss
                y_predict = y_hat.mapValues(lambda val: loss_method.predict(val))
                loss = loss_method.compute_loss(y, y_predict)

        return float(loss)

    def check_convergence(self, loss):
        LOGGER.info("check convergence")
        if self.convergence is None:
            self.convergence = converge_func_factory("diff", self.tol)

        return self.convergence.is_converge(loss)

    @staticmethod
    def data_alignment(data_inst):
        abnormal_detection.empty_table_detection(data_inst)
        abnormal_detection.empty_feature_detection(data_inst)

        schema = data_inst.schema
        new_data_inst = data_inst.mapValues(lambda row: Boosting.data_format_transform(row))

        new_data_inst.schema = schema

        return new_data_inst

    @staticmethod
    def gen_feature_fid_mapping(schema):
        header = schema.get("header")
        feature_name_fid_mapping = dict(zip(range(len(header)), header))
        LOGGER.debug("fid_mapping is {}".format(feature_name_fid_mapping))
        return feature_name_fid_mapping

    @staticmethod
    def accumulate_y_hat(val, new_val, lr=0.1, idx=0):
        val[idx] += lr * new_val
        return val

    def generate_flowid(self, round_num, dim):
        LOGGER.info("generate flowid, flowid {}".format(self.flowid))
        return ".".join(map(str, [self.flowid, round_num, dim]))

    def get_new_predict_score(self, y_hat, cur_sample_weights, dim=0):
        func = functools.partial(self.accumulate_y_hat, lr=self.learning_rate, idx=dim)
        return y_hat.join(cur_sample_weights, func)

    def _get_cv_param(self):
        self.model_param.cv_param.role = self.role
        self.model_param.cv_param.mode = self.mode
        return self.model_param.cv_param

    def predict_proba(self, data_inst):
        pass

    def save_data(self):
        return self.data_output

    def save_model(self):
        pass

    """
    fit and predict
    """

    @abc.abstractmethod
    def fit(self, data_inst, validate_data=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, data_inst):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_summary(self) -> dict:
        raise NotImplementedError()

    """
    Training Procedure
    """

    def prepare_data(self, data_inst):

        """
        prepare data: data alignment, and transform feature to bin id
        Args:
            data_inst: training data
        Returns: data_bin, data_split_points, data_sparse_point
        """
        self.feature_name_fid_mapping = self.gen_feature_fid_mapping(data_inst.schema)
        data_inst = self.data_alignment(data_inst)
        return self.convert_feature_to_bin(data_inst, self.use_missing)

    @abc.abstractmethod
    def check_label(self, *args) -> typing.Tuple[typing.List[int], int, int]:
        """
        Returns: get classes indices, class number and booster dimension and class
        """
        raise NotImplementedError()

    def get_init_score(self, y, num_classes: int):

        if num_classes > 2:
            y_hat, init_score = self.loss.initialize(y, num_classes)
        else:
            y_hat, init_score = self.loss.initialize(y)

        return y_hat, init_score

    @ staticmethod
    def get_label(data_bin):
        """
        Args:
            data_inst: extract y label from DTable
        """
        y = data_bin.mapValues(lambda instance: instance.label)
        return y

    @abc.abstractmethod
    def fit_a_booster(self, *args) -> BasicAlgorithms:
        """
        Returns: get a booster model instance
        """
        raise NotImplementedError()

    def check_stop_condition(self, loss):

        should_stop_a, should_stop_b = False, False

        if self.validation_strategy is not None:
            if self.validation_strategy.need_stop():
                should_stop_a = True

        if self.n_iter_no_change and self.check_convergence(loss):
            should_stop_b = True
            self.is_converged = True

        if should_stop_a or should_stop_b:
            LOGGER.debug('stop triggered, stop triggered by {}'.
                         format('early stop' if should_stop_a else 'n_iter_no change'))

        return should_stop_a or should_stop_b

    """
    Prediction Procedure
    """

    @abc.abstractmethod
    def load_booster(self, *args):
        raise NotImplementedError()

    def score_to_predict_result(self, data_inst, y_hat):

        predicts = None
        if self.task_type == consts.CLASSIFICATION:
            loss_method = self.loss
            if self.num_classes == 2:
                predicts = y_hat.mapValues(lambda f: float(loss_method.predict(f)))
            else:
                predicts = y_hat.mapValues(lambda f: loss_method.predict(f).tolist())

        elif self.task_type == consts.REGRESSION:
            if self.objective_param.objective in ["lse", "lae", "huber", "log_cosh", "fair", "tweedie"]:
                predicts = y_hat
            else:
                raise NotImplementedError("objective {} not supprted yet".format(self.objective_param.objective))

        if self.task_type == consts.CLASSIFICATION:

            predict_result = self.predict_score_to_output(data_inst, predict_score=predicts, classes=self.classes_,
                                                          threshold=self.predict_param.threshold)

        elif self.task_type == consts.REGRESSION:
            predict_result = data_inst.join(predicts, lambda inst, pred: [inst.label, float(pred), float(pred),
                                                                          {"label": float(pred)}])

        else:
            raise NotImplementedError("task type {} not supported yet".format(self.task_type))
        return predict_result

    def data_and_header_alignment(self, data_inst):

        """
        turn data into sparse and align header
        """

        cache_dataset_key = self.predict_data_cache.get_data_key(data_inst)

        if cache_dataset_key in self.data_alignment_map:
            processed_data = self.data_alignment_map[cache_dataset_key]
        else:
            data_inst_tmp = self.data_alignment(data_inst)
            header = [None] * len(self.feature_name_fid_mapping)
            for idx, col in self.feature_name_fid_mapping.items():
                header[idx] = col
            processed_data = data_overview.header_alignment(data_inst_tmp, header)
            self.data_alignment_map[cache_dataset_key] = processed_data

        return processed_data
    """
    Model IO
    """

    @abc.abstractmethod
    def get_model_meta(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model_param(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_model_meta(self, model_meta):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_model_param(self, model_param):
        raise NotImplementedError()

    def get_cur_model(self):
        meta_name, meta_protobuf = self.get_model_meta()
        param_name, param_protobuf = self.get_model_param()
        return {meta_name: meta_protobuf,
                param_name: param_protobuf
                }

    def export_model(self):
        if self.need_cv:
            return None
        return self.get_cur_model()

    def load_model(self, model_dict):
        model_param = None
        model_meta = None
        for _, value in model_dict["model"].items():
            for model in value:
                if model.endswith("Meta"):
                    model_meta = value[model]
                if model.endswith("Param"):
                    model_param = value[model]
        LOGGER.info("load model")

        self.set_model_meta(model_meta)
        self.set_model_param(model_param)
        self.loss = self.get_loss_function()






