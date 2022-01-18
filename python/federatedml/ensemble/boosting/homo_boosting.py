from abc import ABC
import abc
import numpy as np
from federatedml.ensemble.boosting.boosting import Boosting
from federatedml.feature.homo_feature_binning.homo_split_points import HomoFeatureBinningClient, \
    HomoFeatureBinningServer
from federatedml.util.classify_label_checker import ClassifyLabelChecker, RegressionLabelChecker
from federatedml.util import consts
from federatedml.util.homo_label_encoder import HomoLabelEncoderClient, HomoLabelEncoderArbiter
from federatedml.transfer_variable.transfer_class.homo_boosting_transfer_variable import HomoBoostingTransferVariable
from typing import List
from federatedml.feature.fate_element_type import NoneType
from federatedml.util import LOGGER
from federatedml.ensemble.boosting.homo_boosting_aggregator import HomoBoostArbiterAggregator, \
    HomoBoostClientAggregator
from federatedml.optim.convergence import converge_func_factory
from federatedml.param.boosting_param import HomoSecureBoostParam
from federatedml.model_base import Metric
from federatedml.model_base import MetricMeta
from federatedml.util.io_check import assert_io_num_rows_equal

from federatedml.feature.homo_feature_binning import recursive_query_binning
from federatedml.param.feature_binning_param import HomoFeatureBinningParam


class HomoBoostingClient(Boosting, ABC):

    def __init__(self):
        super(HomoBoostingClient, self).__init__()
        self.transfer_inst = HomoBoostingTransferVariable()
        self.model_param = HomoSecureBoostParam()
        self.aggregator = None
        self.binning_obj = None
        self.mode = consts.HOMO

    def federated_binning(self, data_instance):

        binning_param = HomoFeatureBinningParam(method=consts.RECURSIVE_QUERY, bin_num=self.bin_num,
                                                error=self.binning_error)

        if self.use_missing:
            self.binning_obj = recursive_query_binning.Client(params=binning_param, abnormal_list=[NoneType()],
                                                              role=self.role)
            LOGGER.debug('use missing')
        else:
            self.binning_obj = recursive_query_binning.Client(params=binning_param, role=self.role)

        self.binning_obj.fit_split_points(data_instance)

        return self.binning_obj.convert_feature_to_bin(data_instance)

    def check_label(self, data_inst, ) -> List[int]:

        LOGGER.debug('checking labels')

        classes_ = None
        if self.task_type == consts.CLASSIFICATION:
            num_classes, classes_ = ClassifyLabelChecker.validate_label(data_inst)
        else:
            RegressionLabelChecker.validate_label(data_inst)

        return classes_

    @staticmethod
    def check_label_starts_from_zero(aligned_labels):
        """
        in current version, labels should start from 0 and
        are consecutive integers
        """
        if aligned_labels[0] != 0:
            raise ValueError('label should starts from 0')
        for prev, aft in zip(aligned_labels[:-1], aligned_labels[1:]):
            if prev + 1 != aft:
                raise ValueError('labels should be a sequence of consecutive integers, '
                                 'but got {} and {}'.format(prev, aft))

    def sync_feature_num(self):
        self.transfer_inst.feature_number.remote(self.feature_num, role=consts.ARBITER, idx=-1, suffix=('feat_num',))

    def sync_start_round_and_end_round(self):
        self.transfer_inst.start_and_end_round.remote((self.start_round, self.boosting_round),
                                                      role=consts.ARBITER, idx=-1)

    def data_preporcess(self, data_inst):
        # transform to sparse and binning
        data_inst = self.data_alignment(data_inst)
        self.data_bin, self.bin_split_points, self.bin_sparse_points = self.federated_binning(data_inst)

    def fit(self, data_inst, validate_data=None):

        # init federation obj
        self.aggregator = HomoBoostClientAggregator()
        self.binning_obj = HomoFeatureBinningClient()

        # binning
        self.data_preporcess(data_inst)

        # fid mapping and warm start check
        if not self.is_warm_start:
            self.feature_name_fid_mapping = self.gen_feature_fid_mapping(data_inst.schema)
        else:
            self.feat_name_check(data_inst, self.feature_name_fid_mapping)

        # set feature_num
        self.feature_num = self.bin_split_points.shape[0]

        # sync feature num
        self.sync_feature_num()

        # initialize validation strategy
        self.callback_list.on_train_begin(data_inst, validate_data)

        # check labels
        local_classes = self.check_label(self.data_bin)

        # set start round
        self.start_round = len(self.boosting_model_list) // self.booster_dim

        # sync label class and set y
        if self.task_type == consts.CLASSIFICATION:

            aligned_label, new_label_mapping = HomoLabelEncoderClient().label_alignment(local_classes)
            if self.is_warm_start:
                assert set(aligned_label) == set(self.classes_), 'warm start label alignment failed, differences: {}'. \
                    format(set(aligned_label).symmetric_difference(set(self.classes_)))
            self.classes_ = aligned_label
            self.check_label_starts_from_zero(self.classes_)
            # set labels
            self.num_classes = len(new_label_mapping)
            LOGGER.info('aligned labels are {}, num_classes is {}'.format(aligned_label, self.num_classes))
            self.y = self.data_bin.mapValues(lambda instance: new_label_mapping[instance.label])
            # set tree dimension
            self.booster_dim = self.num_classes if self.num_classes > 2 else 1

        else:
            self.y = self.data_bin.mapValues(lambda instance: instance.label)

        # set loss function
        self.loss = self.get_loss_function()

        # set y_hat_val, if warm start predict cur samples
        if self.is_warm_start:
            self.y_hat = self.predict(data_inst, ret_format='raw')
            self.boosting_round += self.start_round
            self.callback_warm_start_init_iter(self.start_round)
        else:
            self.y_hat, self.init_score = self.get_init_score(self.y, self.num_classes)

        # sync start round and end round
        self.sync_start_round_and_end_round()

        self.preprocess()

        LOGGER.info('begin to fit a boosting tree')
        for epoch_idx in range(self.start_round, self.boosting_round):

            LOGGER.info('cur epoch idx is {}'.format(epoch_idx))

            self.callback_list.on_epoch_begin(epoch_idx)

            for class_idx in range(self.booster_dim):

                # fit a booster
                model = self.fit_a_learner(epoch_idx, class_idx)
                booster_meta, booster_param = model.get_model()
                if booster_meta is not None and booster_param is not None:
                    self.booster_meta = booster_meta
                    self.boosting_model_list.append(booster_param)

                # update predict score
                cur_sample_weights = model.get_sample_weights()
                self.y_hat = self.get_new_predict_score(self.y_hat, cur_sample_weights, dim=class_idx)

            local_loss = self.compute_loss(self.y_hat, self.y)
            self.aggregator.send_local_loss(local_loss, self.data_bin.count(), suffix=(epoch_idx,))

            validation_strategy = self.callback_list.get_validation_strategy()
            if validation_strategy:
                validation_strategy.set_precomputed_train_scores(self.score_to_predict_result(data_inst, self.y_hat))
            self.callback_list.on_epoch_end(epoch_idx)

            # check stop flag if n_iter_no_change is True
            if self.n_iter_no_change:
                should_stop = self.aggregator.get_converge_status(suffix=(str(epoch_idx),))
                if should_stop:
                    LOGGER.info('n_iter_no_change stop triggered')
                    break

        self.postprocess()
        self.callback_list.on_train_end()
        self.set_summary(self.generate_summary())

    @assert_io_num_rows_equal
    def predict(self, data_inst):
        # predict is implemented in homo_secureboost
        raise NotImplementedError('predict func is not implemented')

    @abc.abstractmethod
    def fit_a_learner(self, epoch_idx: int, booster_dim: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_learner(self, model_meta, model_param, epoch_idx, booster_idx):
        raise NotImplementedError()


class HomoBoostingArbiter(Boosting, ABC):

    def __init__(self):
        super(HomoBoostingArbiter, self).__init__()
        self.transfer_inst = HomoBoostingTransferVariable()
        self.check_convergence_func = None
        self.aggregator = None
        self.binning_obj = None

    def federated_binning(self, ):

        binning_param = HomoFeatureBinningParam(method=consts.RECURSIVE_QUERY, bin_num=self.bin_num,
                                                error=self.binning_error)

        if self.use_missing:
            self.binning_obj = recursive_query_binning.Server(binning_param, abnormal_list=[NoneType()])
        else:
            self.binning_obj = recursive_query_binning.Server(binning_param, abnormal_list=[])

        self.binning_obj.fit_split_points(None)

    def sync_feature_num(self):
        feature_num_list = self.transfer_inst.feature_number.get(idx=-1, suffix=('feat_num',))
        for num in feature_num_list[1:]:
            assert feature_num_list[0] == num

        return feature_num_list[0]

    def sync_start_round_and_end_round(self):
        r_list = self.transfer_inst.start_and_end_round.get(-1)
        LOGGER.info('get start/end round from clients: {}'.format(r_list))
        self.start_round, self.boosting_round = r_list[0]

    def check_label(self):
        pass

    def fit(self, data_inst, validate_data=None):

        # init binning obj
        self.aggregator = HomoBoostArbiterAggregator()
        self.binning_obj = HomoFeatureBinningServer()

        self.federated_binning()
        # initializing
        self.feature_num = self.sync_feature_num()

        if self.task_type == consts.CLASSIFICATION:
            label_mapping = HomoLabelEncoderArbiter().label_alignment()
            LOGGER.info('label mapping is {}'.format(label_mapping))
            self.booster_dim = len(label_mapping) if len(label_mapping) > 2 else 1

        if self.n_iter_no_change:
            self.check_convergence_func = converge_func_factory("diff", self.tol)

        # sync start round and end round
        self.sync_start_round_and_end_round()

        LOGGER.info('begin to fit a boosting tree')
        self.preprocess()
        for epoch_idx in range(self.start_round, self.boosting_round):

            LOGGER.info('cur epoch idx is {}'.format(epoch_idx))

            for class_idx in range(self.booster_dim):
                model = self.fit_a_learner(epoch_idx, class_idx)

            global_loss = self.aggregator.aggregate_loss(suffix=(epoch_idx,))
            self.history_loss.append(global_loss)
            LOGGER.debug('cur epoch global loss is {}'.format(global_loss))

            self.callback_metric("loss",
                                 "train",
                                 [Metric(epoch_idx, global_loss)])

            if self.n_iter_no_change:
                should_stop = self.aggregator.broadcast_converge_status(self.check_convergence, (global_loss,),
                                                                        suffix=(epoch_idx,))
                LOGGER.debug('stop flag sent')
                if should_stop:
                    break

        self.callback_meta("loss",
                           "train",
                           MetricMeta(name="train",
                                      metric_type="LOSS",
                                      extra_metas={"Best": min(self.history_loss)}))
        self.postprocess()
        self.callback_list.on_train_end()
        self.set_summary(self.generate_summary())

    def predict(self, data_inst=None):
        LOGGER.debug('arbiter skip prediction')

    @abc.abstractmethod
    def fit_a_learner(self, epoch_idx: int, booster_dim: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_learner(self, model_meta, model_param, epoch_idx, booster_idx):
        raise NotImplementedError()
