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
import numpy as np
import torch
from torch.utils.data import DataLoader
from fate_arch.computing._util import is_table
from fate_arch.session import computing_session as session
from federatedml.feature.instance import Instance
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.model_base import Metric
from federatedml.model_base import MetricMeta
from federatedml.nn.hetero.base import HeteroNNBase
from federatedml.nn.hetero.model import HeteroNNGuestModel
from federatedml.optim.convergence import converge_func_factory
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.param.hetero_nn_param import HeteroNNParam as NNParameter
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNMeta
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNParam
from federatedml.util import consts, LOGGER
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.nn.dataset.table import TableDataset
from federatedml.statistic.data_overview import check_with_inst_id
from federatedml.nn.backend.utils.data import add_match_id

MODELMETA = "HeteroNNGuestMeta"
MODELPARAM = "HeteroNNGuestParam"


class HeteroNNGuest(HeteroNNBase):

    def __init__(self):
        super(HeteroNNGuest, self).__init__()
        self.task_type = None
        self.converge_func = None

        self.batch_generator = batch_generator.Guest()
        self.data_keys = []

        self.label_dict = {}

        self.model = None
        self.role = consts.GUEST
        self.history_loss = []
        self.input_shape = None
        self._summary_buf = {"history_loss": [],
                             "is_converged": False,
                             "best_iteration": -1}

        self.dataset_cache_dict = {}

        self.default_table_partitions = 4

    def _init_model(self, hetero_nn_param):
        super(HeteroNNGuest, self)._init_model(hetero_nn_param)
        self.task_type = hetero_nn_param.task_type
        self.converge_func = converge_func_factory(self.early_stop, self.tol)

    def _build_model(self):
        self.model = HeteroNNGuestModel(
            self.hetero_nn_param, self.component_properties, self.flowid)
        self.model.set_transfer_variable(self.transfer_variable)
        self.model.set_partition(self.default_table_partitions)

    def _set_loss_callback_info(self):
        self.callback_meta("loss",
                           "train",
                           MetricMeta(name="train",
                                      metric_type="LOSS",
                                      extra_metas={"unit_name": "iters"}))

    @staticmethod
    def _disable_sample_weight(dataset):
        # currently not support sample weight
        if isinstance(dataset, TableDataset):
            dataset.with_sample_weight = False

    def fit(self, data_inst, validate_data=None):

        if hasattr(
                data_inst,
                'partitions') and data_inst.partitions is not None:
            self.default_table_partitions = data_inst.partitions
            LOGGER.debug(
                'reset default partitions is {}'.format(
                    self.default_table_partitions))

        train_ds = self.prepare_dataset(
            data_inst, data_type='train', check_label=True)
        train_ds.train()  # set dataset to train mode
        self._disable_sample_weight(train_ds)

        if validate_data is not None:
            val_ds = self.prepare_dataset(validate_data, data_type='validate')
            val_ds.train()  # set dataset to train mode
            self._disable_sample_weight(val_ds)
        else:
            val_ds = None

        self.callback_list.on_train_begin(train_ds, val_ds)

        # collect data from table to form data loader
        if not self.component_properties.is_warm_start:
            self._build_model()
            epoch_offset = 0
        else:
            self.callback_warm_start_init_iter(self.history_iter_epoch)
            epoch_offset = self.history_iter_epoch + 1

        # set label number
        self.model.set_label_num(self.label_num)

        if len(train_ds) == 0:
            self.model.set_empty()

        self._set_loss_callback_info()

        batch_size = len(train_ds) if self.batch_size == - \
            1 else self.batch_size
        data_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=0)

        for cur_epoch in range(epoch_offset, self.epochs + epoch_offset):

            self.iter_epoch = cur_epoch
            LOGGER.debug("cur epoch is {}".format(cur_epoch))
            self.callback_list.on_epoch_begin(cur_epoch)
            epoch_loss = 0
            acc_sample_num = 0

            for batch_idx, (batch_data, batch_label) in enumerate(data_loader):
                batch_loss = self.model.train(
                    batch_data, batch_label, cur_epoch, batch_idx)
                if acc_sample_num + batch_size > len(train_ds):
                    batch_len = len(train_ds) - acc_sample_num
                else:
                    batch_len = batch_size
                acc_sample_num += batch_size
                epoch_loss += batch_loss * batch_len

            epoch_loss = epoch_loss / len(train_ds)
            LOGGER.debug("epoch {} loss is {}".format(cur_epoch, epoch_loss))

            self.callback_metric("loss",
                                 "train",
                                 [Metric(cur_epoch, epoch_loss)])

            self.history_loss.append(epoch_loss)

            self.callback_list.on_epoch_end(cur_epoch)
            if self.callback_variables.stop_training:
                LOGGER.debug('early stopping triggered')
                break

            if self.hetero_nn_param.selector_param.method:
                # when use selective bp, loss converge will be disabled
                is_converge = False
            else:
                is_converge = self.converge_func.is_converge(epoch_loss)

            self._summary_buf["is_converged"] = is_converge
            self.transfer_variable.is_converge.remote(is_converge,
                                                      role=consts.HOST,
                                                      idx=-1,
                                                      suffix=(cur_epoch,))

            if is_converge:
                LOGGER.debug(
                    "Training process is converged in epoch {}".format(cur_epoch))
                break

        self.callback_list.on_train_end()
        self.set_summary(self._get_model_summary())

    @assert_io_num_rows_equal
    def predict(self, data_inst):

        with_match_id = False
        if is_table(data_inst):
            with_match_id = check_with_inst_id(data_inst)

        ds = self.prepare_dataset(data_inst, data_type='predict')
        ds.eval()  # set dataset to eval mode
        self._disable_sample_weight(ds)
        keys = ds.get_sample_ids()

        batch_size = len(ds) if self.batch_size == -1 else self.batch_size
        dl = DataLoader(ds, batch_size=batch_size)
        preds = []
        labels = []

        for batch_data, batch_label in dl:
            batch_pred = self.model.predict(batch_data)
            preds.append(batch_pred)
            labels.append(batch_label)

        preds = np.concatenate(preds, axis=0)
        labels = torch.concat(labels, dim=0).cpu().numpy().flatten().tolist()

        id_table = [(id_, Instance(label=l)) for id_, l in zip(keys, labels)]
        if with_match_id:
            add_match_id(id_table, ds.ds)  # ds is wrap shuffle dataset here
        data_inst = session.parallelize(
            id_table,
            partition=self.default_table_partitions,
            include_key=True)

        if self.task_type == consts.REGRESSION:
            preds = preds.flatten().tolist()
            preds = [float(pred) for pred in preds]
            predict_tb = session.parallelize(zip(keys, preds), include_key=True,
                                             partition=self.default_table_partitions)
            result = self.predict_score_to_output(data_inst, predict_tb)
        else:
            if self.label_num > 2:
                preds = preds.tolist()
                preds = [list(map(float, pred)) for pred in preds]
                predict_tb = session.parallelize(zip(keys, preds), include_key=True,
                                                 partition=self.default_table_partitions)
                result = self.predict_score_to_output(
                    data_inst, predict_tb, classes=list(range(self.label_num)))

            else:
                preds = preds.flatten().tolist()
                preds = [float(pred) for pred in preds]
                predict_tb = session.parallelize(zip(keys, preds), include_key=True,
                                                 partition=self.default_table_partitions)
                threshold = self.predict_param.threshold
                result = self.predict_score_to_output(
                    data_inst, predict_tb, classes=[
                        0, 1], threshold=threshold)

        return result

    def export_model(self):
        if self.need_cv:
            return None
        model = {MODELMETA: self._get_model_meta(),
                 MODELPARAM: self._get_model_param()}

        return model

    def load_model(self, model_dict):

        model_dict = list(model_dict["model"].values())[0]
        param = model_dict.get(MODELPARAM)
        meta = model_dict.get(MODELMETA)
        if self.hetero_nn_param is None:
            self.hetero_nn_param = NNParameter()
            self.hetero_nn_param.check()
            self.predict_param = self.hetero_nn_param.predict_param
        self._build_model()
        self._restore_model_meta(meta)
        self._restore_model_param(param)

    def _get_model_summary(self):

        self._summary_buf["history_loss"] = self.history_loss
        if self.callback_variables.validation_summary:
            self._summary_buf["validation_metrics"] = self.callback_variables.validation_summary
        """
        if self.validation_strategy:
            validation_summary = self.validation_strategy.summary()
            if validation_summary:
                self._summary_buf["validation_metrics"] = validation_summary
        """

        return self._summary_buf

    def _get_model_meta(self):
        model_meta = HeteroNNMeta()
        model_meta.task_type = self.task_type
        model_meta.module = 'HeteroNN'
        model_meta.batch_size = self.batch_size
        model_meta.epochs = self.epochs
        model_meta.early_stop = self.early_stop
        model_meta.tol = self.tol
        model_meta.hetero_nn_model_meta.CopyFrom(
            self.model.get_hetero_nn_model_meta())

        return model_meta

    def _get_model_param(self):
        model_param = HeteroNNParam()
        model_param.iter_epoch = self.iter_epoch
        model_param.hetero_nn_model_param.CopyFrom(
            self.model.get_hetero_nn_model_param())
        model_param.num_label = self.label_num
        model_param.best_iteration = self.callback_variables.best_iteration
        model_param.header.extend(self._header)

        for loss in self.history_loss:
            model_param.history_loss.append(loss)

        return model_param

    def get_metrics_param(self):
        if self.task_type == consts.CLASSIFICATION:
            if self.label_num == 2:
                return EvaluateParam(eval_type="binary",
                                     pos_label=1, metrics=self.metrics)
            else:
                return EvaluateParam(eval_type="multi", metrics=self.metrics)
        else:
            return EvaluateParam(eval_type="regression", metrics=self.metrics)

    def _restore_model_param(self, param):
        super(HeteroNNGuest, self)._restore_model_param(param)
        self.label_num = param.num_label
