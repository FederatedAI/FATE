from abc import ABC
import abc
from federatedml.ensemble.boosting.boosting_core import Boosting
from federatedml.feature.homo_feature_binning.homo_split_points import HomoFeatureBinningClient, \
                                                                      HomoFeatureBinningServer
from federatedml.util.classify_label_checker import ClassifyLabelChecker, RegressionLabelChecker
from federatedml.util import consts
from federatedml.util.homo_label_encoder import HomoLabelEncoderClient, HomoLabelEncoderArbiter
from federatedml.transfer_variable.transfer_class.homo_boosting_transfer_variable import HomoBoostingTransferVariable
from typing import List
from federatedml.feature.fate_element_type import NoneType
from federatedml.util import LOGGER
from federatedml.ensemble.boosting.boosting_core.homo_boosting_aggregator import HomoBoostArbiterAggregator, \
    HomoBoostClientAggregator
from federatedml.optim.convergence import converge_func_factory
from federatedml.param.boosting_param import HomoSecureBoostParam
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.param.feature_binning_param import FeatureBinningParam


class HomoBoostingClient(Boosting, ABC):

    def __init__(self):
        super(HomoBoostingClient, self).__init__()
        self.transfer_inst = HomoBoostingTransferVariable()
        self.aggregator = HomoBoostClientAggregator()
        self.model_param = HomoSecureBoostParam()
        self.binning_obj = HomoFeatureBinningClient()
        self.mode = consts.HOMO

    def federated_binning(self,  data_instance):

        binning_param = FeatureBinningParam(bin_num=self.bin_num, error=self.binning_error)
        self.binning_obj.bin_param = binning_param

        if self.use_missing:
            binning_result = self.binning_obj.average_run(data_instances=data_instance, abnormal_list=[NoneType()])
        else:
            binning_result = self.binning_obj.average_run(data_instances=data_instance,)

        return self.binning_obj.convert_feature_to_bin(data_instance, binning_result)

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
        self.transfer_inst.feature_number.remote(self.feature_num, role=consts.ARBITER, idx=-1, suffix=('feat_num', ))

    def fit(self, data_inst, validate_data=None):

        # binning
        data_inst = self.data_alignment(data_inst)
        self.data_bin, self.bin_split_points, self.bin_sparse_points = self.federated_binning(data_inst)

        # fid mapping
        self.feature_name_fid_mapping = self.gen_feature_fid_mapping(data_inst.schema)

        # set feature_num
        self.feature_num = self.bin_split_points.shape[0]

        # sync feature num
        self.sync_feature_num()

        # initialize validation strategy
        self.validation_strategy = self.init_validation_strategy(train_data=data_inst, validate_data=validate_data, )

        # check labels
        local_classes = self.check_label(self.data_bin)

        # sync label class and set y
        if self.task_type == consts.CLASSIFICATION:

            aligned_label, new_label_mapping = HomoLabelEncoderClient().label_alignment(local_classes)
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

        # set y_hat_val
        self.y_hat, self.init_score = self.get_init_score(self.y, self.num_classes)

        LOGGER.info('begin to fit a boosting tree')
        for epoch_idx in range(self.boosting_round):

            LOGGER.info('cur epoch idx is {}'.format(epoch_idx))

            for class_idx in range(self.booster_dim):

                # fit a booster
                model = self.fit_a_booster(epoch_idx, class_idx)

                booster_meta, booster_param = model.get_model()

                if booster_meta is not None and booster_param is not None:
                    self.booster_meta = booster_meta
                    self.boosting_model_list.append(booster_param)

                # update predict score
                cur_sample_weights = model.get_sample_weights()
                self.y_hat = self.get_new_predict_score(self.y_hat, cur_sample_weights, dim=class_idx)

            local_loss = self.compute_loss(self.y_hat, self.y)
            self.aggregator.send_local_loss(local_loss, self.data_bin.count(), suffix=(epoch_idx,))

            if self.validation_strategy:
                self.validation_strategy.validate(self, epoch_idx)

            # check stop flag if n_iter_no_change is True
            if self.n_iter_no_change:
                should_stop = self.aggregator.get_converge_status(suffix=(str(epoch_idx),))
                if should_stop:
                    LOGGER.info('n_iter_no_change stop triggered')
                    break

        self.set_summary(self.generate_summary())

    @assert_io_num_rows_equal
    def predict(self, data_inst):
        # predict is implemented in homo_secureboost
        raise NotImplementedError('predict func is not implemented')

    @abc.abstractmethod
    def fit_a_booster(self, epoch_idx: int, booster_dim: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        raise NotImplementedError()


class HomoBoostingArbiter(Boosting, ABC):

    def __init__(self):
        super(HomoBoostingArbiter, self).__init__()
        self.aggregator = HomoBoostArbiterAggregator()
        self.transfer_inst = HomoBoostingTransferVariable()
        self.check_convergence_func = None
        self.binning_obj = HomoFeatureBinningServer()

    def federated_binning(self):
        self.binning_obj.average_run()

    def sync_feature_num(self):
        feature_num_list = self.transfer_inst.feature_number.get(idx=-1, suffix=('feat_num',))
        for num in feature_num_list[1:]:
            assert feature_num_list[0] == num
        return feature_num_list[0]

    def check_label(self):
        pass

    def fit(self, data_inst, validate_data=None):

        self.federated_binning()
        # initializing
        self.feature_num = self.sync_feature_num()

        if self.task_type == consts.CLASSIFICATION:
            label_mapping = HomoLabelEncoderArbiter().label_alignment()
            LOGGER.info('label mapping is {}'.format(label_mapping))
            self.booster_dim = len(label_mapping) if len(label_mapping) > 2 else 1

        if self.n_iter_no_change:
            self.check_convergence_func = converge_func_factory("diff", self.tol)

        LOGGER.info('begin to fit a boosting tree')
        for epoch_idx in range(self.boosting_round):

            LOGGER.info('cur epoch idx is {}'.format(epoch_idx))

            for class_idx in range(self.booster_dim):
                model = self.fit_a_booster(epoch_idx, class_idx)

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

        self.set_summary(self.generate_summary())

    def predict(self, data_inst=None):
        LOGGER.debug('arbiter skip prediction')

    @abc.abstractmethod
    def fit_a_booster(self, epoch_idx: int, booster_dim: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        raise NotImplementedError()



