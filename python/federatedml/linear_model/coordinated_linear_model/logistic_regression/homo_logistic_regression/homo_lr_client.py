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
import torch as t
from fate_arch.computing._util import is_table
from federatedml.linear_model.coordinated_linear_model.logistic_regression.\
    homo_logistic_regression.homo_lr_base import HomoLRBase
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.nn.dataset.table import TableDataset
from federatedml.nn.homo.trainer.trainer_base import ExporterBase
from federatedml.nn.homo.trainer.fedavg_trainer import FedAVGTrainer
from federatedml.callbacks.model_checkpoint import ModelCheckpoint
from federatedml.callbacks.validation_strategy import ValidationStrategy
from federatedml.protobuf.generated import lr_model_param_pb2
from federatedml.model_base import MetricMeta
from fate_arch.session import computing_session
from federatedml.nn.backend.utils.data import get_ret_predict_table, add_match_id
from federatedml.nn.loss.weighted_loss import WeightedBCE
from federatedml.statistic.data_overview import check_with_inst_id


def linear_weight_to_torch(model_weights):

    model_weights: LinearModelWeights = model_weights
    weights = model_weights.coef_
    bias = None
    use_bias = False
    if model_weights.fit_intercept:
        bias = model_weights.intercept_
        use_bias = True
    torch_linear_layer = t.nn.Linear(
        in_features=weights.shape[0], out_features=1, bias=use_bias)
    LOGGER.debug('weights are {}, biase is {}'.format(weights, bias))
    torch_linear_layer.weight.data.copy_(t.Tensor(weights))
    if use_bias:
        torch_linear_layer.bias.data.copy_(t.Tensor([bias]))
    torch_model = t.nn.Sequential(
        torch_linear_layer,
        t.nn.Sigmoid()
    )

    return torch_model


def torch_to_linear_weight(model_weights, torch_model):

    if model_weights.fit_intercept:
        model_weights._weights = np.concatenate([torch_model[0].weight.detach().numpy().flatten(),
                                                 torch_model[0].bias.detach().numpy().flatten()]).tolist()
    else:
        model_weights._weights = torch_model[0].weight.detach(
        ).numpy().flatten().tolist()


class WrappedOptAndScheduler(object):

    def __init__(self, opt, scheduler):
        self.opt = opt
        self.scheduler = scheduler

    def zero_grad(self, ):
        self.opt.zero_grad()

    def step(self, ):
        self.opt.step()
        self.scheduler.step()

    def state_dict(self):
        return self.opt.state_dict()

    def restep(self, n):
        for i in range(n):
            self.opt.zero_grad()
            self.opt.step()
            self.scheduler.step()


class HomoLRClientExporter(ExporterBase):

    def __init__(self, header, homo_lr_meta, model_weights, param_name, meta_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.header = header
        self.homo_lr_meta = homo_lr_meta
        self.model_weights = model_weights
        self.param_name = param_name
        self.meta_name = meta_name

    def export_model_dict(
            self,
            model=None,
            optimizer=None,
            model_define=None,
            optimizer_define=None,
            loss_define=None,
            epoch_idx=None,
            converge_status=None,
            loss_history=None,
            best_epoch=None,
            extra_data={}):

        torch_to_linear_weight(self.model_weights, model)
        weight_dict = {}
        for idx, header_name in enumerate(self.header):
            coef_i = self.model_weights.coef_[idx]
            weight_dict[header_name] = float(coef_i)

        result = {'iters': epoch_idx,
                  'loss_history': loss_history,
                  'is_converged': converge_status,
                  'weight': weight_dict,
                  'intercept': self.model_weights.intercept_,
                  'header': self.header,
                  'best_iteration': best_epoch
                  }

        param = lr_model_param_pb2.LRModelParam(**result)
        meta = self.homo_lr_meta

        return {self.param_name: param, self.meta_name: meta}


class HomoLRClient(HomoLRBase):

    def __init__(self):
        super(HomoLRClient, self).__init__()
        self.loss_history = []
        self.role = consts.GUEST
        self.dataset_cache = {}
        self.trainer = None
        self.best_iteration = -1
        # check point
        self.save_freq = None
        self.model_checkpoint = None

    def _init_model(self, params):
        super()._init_model(params)

    def get_dataset(self, data):

        if id(data) in self.dataset_cache:
            return self.dataset_cache[id(data)]

        if is_table(data):
            dataset = TableDataset()
            dataset.load(data)
            self.dataset_cache[id(data)] = dataset
            return dataset

        else:
            raise RuntimeError('unknown data type {}'.format(data))

    def init(self, dataset: TableDataset, partitions):

        torch_model = linear_weight_to_torch(self.model_weights)
        LOGGER.debug('torch model is {}, parameters are {} dataset {}'.format(
            torch_model, list(torch_model.parameters()), dataset))

        batch_size = len(dataset) if self.batch_size == -1 else self.batch_size

        optimizer, scheduler = self.get_torch_optimizer(
            torch_model, self.model_param)
        wrap_optimizer = WrappedOptAndScheduler(optimizer, scheduler)
        LOGGER.debug('init optimizer statedict is {}'.format(wrap_optimizer.state_dict()))

        if dataset.with_sample_weight:
            loss = WeightedBCE()
        else:
            loss = t.nn.BCELoss()

        early_stop = None
        if self.early_stop != 'weight_diff':
            early_stop = self.early_stop

        trainer = FedAVGTrainer(
            epochs=self.max_iter,
            batch_size=batch_size,
            data_loader_worker=partitions,
            secure_aggregate=True,
            aggregate_every_n_epoch=self.aggregate_iters,
            validation_freqs=self.validation_freqs,
            task_type='binary',
            checkpoint_save_freqs=self.save_freq,
            early_stop=early_stop,
            shuffle=False,
            tol=self.tol)

        if not self.callback_one_vs_rest:
            trainer.set_tracker(self.tracker)
        trainer.set_model(torch_model)
        trainer.set_model_exporter(
            HomoLRClientExporter(
                header=self.header,
                homo_lr_meta=self._get_meta(),
                model_weights=self.model_weights,
                meta_name=self.model_meta_name,
                param_name=self.model_param_name))
        trainer.set_checkpoint(self.model_checkpoint)

        return trainer, torch_model, wrap_optimizer, loss

    def get_model_summary(self, is_converged, best_iteration, loss_history, eval_summary):
        header = self.header
        if header is None:
            return {}
        weight_dict, intercept_ = self.get_weight_intercept_dict(header)
        summary = {"coef": weight_dict,
                   "intercept": intercept_,
                   "is_converged": is_converged,
                   "best_iteration": best_iteration,
                   "local_loss_history": loss_history,
                   "validation_metrics": eval_summary
                   }

        return summary

    def fit_binary(self, data_instances, validate_data=None):

        for callback_cpn in self.callback_list.callback_list:
            if isinstance(callback_cpn, ModelCheckpoint):
                self.save_freq = callback_cpn.save_freq
                self.model_checkpoint = callback_cpn
            elif isinstance(callback_cpn, ValidationStrategy):
                self.validation_freqs = callback_cpn.validation_freqs

        train_set = self.get_dataset(data_instances)
        train_set.set_type('train')
        if validate_data is not None:
            val_set = self.get_dataset(validate_data)
            val_set.set_type('validate')
        else:
            val_set = None

        if not self.component_properties.is_warm_start:
            self.model_weights = self._init_model_variables(data_instances)
        else:
            LOGGER.debug('callback warm start, iter {}'.format(self.n_iter_))
            self.callback_warm_start_init_iter(self.n_iter_ + 1)

        # fate loss callback setting
        LOGGER.debug('need one vs rest {}'.format(self.need_one_vs_rest))
        if not self.callback_one_vs_rest:  # ovr does not display loss
            self.callback_meta(
                "loss",
                "train",
                MetricMeta(
                    name="train",
                    metric_type="LOSS",
                    extra_metas={
                        "unit_name": "epochs"}))

        self.trainer, torch_model, wrap_optimizer, loss = self.init(
            train_set, data_instances.partitions)

        if self.component_properties.is_warm_start:
            wrap_optimizer.restep(self.n_iter_ + 1)

        self.trainer.train(train_set, val_set, loss=loss,
                           optimizer=wrap_optimizer)

        torch_to_linear_weight(self.model_weights, torch_model)
        eval_summary = self.trainer.get_evaluation_summary()
        summary = self.trainer.get_summary()
        self.is_converged, self.best_iteration, self.loss_history = summary[
            'need_stop'], summary['best_epoch'], summary['loss_history']
        self.n_iter_ = len(self.loss_history) - 1
        self.set_summary(self.get_model_summary(
            self.best_iteration, self.loss_history, self.is_converged, eval_summary))

    @assert_io_num_rows_equal
    def predict(self, data_instances):

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)

        data_instances = self.align_data_header(data_instances, self.header)
        with_inst_id = check_with_inst_id(data_instances)

        dataset = self.get_dataset(data_instances)

        if self.need_one_vs_rest:
            predict_result = self.one_vs_rest_obj.predict(data_instances)
            return predict_result

        dataset.set_type('predict')
        if self.trainer is None:
            self.trainer, torch_model, wrap_optimizer, loss = self.init(
                dataset, data_instances.partitions)
        trainer_ret = self.trainer.predict(dataset)
        id_table, pred_table, classes = trainer_ret()

        if with_inst_id:
            add_match_id(id_table=id_table, dataset_inst=dataset)

        id_dtable, pred_dtable = get_ret_predict_table(
            id_table, pred_table, classes, data_instances.partitions, computing_session)
        ret_table = self.predict_score_to_output(
            id_dtable, pred_dtable, classes)

        return ret_table
