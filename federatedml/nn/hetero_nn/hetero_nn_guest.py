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

from arch.api import session
from arch.api.utils import log_utils
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.nn.hetero_nn.backend.model_builder import model_builder
from federatedml.nn.hetero_nn.hetero_nn_base import HeteroNNBase
from federatedml.optim.convergence import converge_func_factory
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNMeta
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNParam
from federatedml.util import consts

LOGGER = log_utils.getLogger()
MODELMETA = "HeteroNNGuestMeta"
MODELPARAM = "HeteroNNGuestParam"


class HeteroNNGuest(HeteroNNBase):
    def __init__(self):
        super(HeteroNNGuest, self).__init__()
        self.task_type = None
        self.converge_func = None

        self.batch_generator = batch_generator.Guest()
        self.data_keys = []

        self.model_builder = None
        self.label_dict = {}

        self.model = None
        self.history_loss = []
        self.iter_epoch = 0
        self.num_label = 2

        self.input_shape = None
        self.validation_strategy = None

    def _init_model(self, hetero_nn_param):
        super(HeteroNNGuest, self)._init_model(hetero_nn_param)

        self.task_type = hetero_nn_param.task_type
        self.converge_func = converge_func_factory(self.early_stop, self.tol)

    def _build_model(self):
        self.model = model_builder("guest", self.hetero_nn_param)
        self.model.set_transfer_variable(self.transfer_variable)

    def _set_loss_callback_info(self):
        self.callback_meta("loss",
                           "train",
                           MetricMeta(name="train",
                                      metric_type="LOSS",
                                      extra_metas={"unit_name": "iters"}))

    def fit(self, data_inst, validate_data=None):
        self.validation_strategy = self.init_validation_strategy(data_inst, validate_data)
        self._build_model()
        self.prepare_batch_data(self.batch_generator, data_inst)
        if not self.input_shape:
            self.model.set_empty()

        self._set_loss_callback_info()
        cur_epoch = 0
        while cur_epoch < self.epochs:
            LOGGER.debug("cur epoch is {}".format(cur_epoch))
            epoch_loss = 0

            for batch_idx in range(len(self.data_x)):
                self.model.train(self.data_x[batch_idx], self.data_y[batch_idx], cur_epoch, batch_idx)

                self.reset_flowid()
                metrics = self.model.evaluate(self.data_x[batch_idx], self.data_y[batch_idx], cur_epoch, batch_idx)
                self.recovery_flowid()

                LOGGER.debug("metrics is {}".format(metrics))
                batch_loss = metrics["loss"]

                epoch_loss += batch_loss

            epoch_loss /= len(self.data_x)

            LOGGER.debug("epoch {}' loss is {}".format(cur_epoch, epoch_loss))

            self.callback_metric("loss",
                                 "train",
                                 [Metric(cur_epoch, epoch_loss)])

            self.history_loss.append(epoch_loss)

            if self.validation_strategy:
                self.validation_strategy.validate(self, cur_epoch)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break

            is_converge = self.converge_func.is_converge(epoch_loss)
            self.transfer_variable.is_converge.remote(is_converge,
                                                      role=consts.HOST,
                                                      idx=0,
                                                      suffix=(cur_epoch,))

            if is_converge:
                LOGGER.debug("Training process is converged in epoch {}".format(cur_epoch))
                break

            cur_epoch += 1

        if cur_epoch == self.epochs:
            LOGGER.debug("Training process reach max training epochs {} and not converged".format(self.epochs))

        if self.validation_strategy and self.validation_strategy.has_saved_best_model():
            self.load_model(self.validation_strategy.cur_best_model)

    def predict(self, data_inst):
        keys, test_x, test_y = self._load_data(data_inst)
        self.set_partition(data_inst)

        preds = self.model.predict(test_x)

        predict_tb = session.parallelize(zip(keys, preds), include_key=True)
        if self.task_type == "regression":
            result = data_inst.join(predict_tb,
                                    lambda inst, predict: [inst.label, float(predict[0]), float(predict[0]),
                                                           {"label": float(predict[0])}])
        else:
            if self.num_label > 2:
                result = data_inst.join(predict_tb,
                                        lambda inst, predict: [inst.label,
                                                               int(np.argmax(predict)),
                                                               float(np.max(predict)),
                                                               dict([(str(idx), float(predict[idx])) for idx in
                                                                     range(predict.shape[0])])])

            else:
                threshold = self.predict_param.threshold
                result = data_inst.join(predict_tb,
                                        lambda inst, predict: [inst.label,
                                                               1 if predict[0] > threshold else 0,
                                                               float(predict[0]),
                                                               {"0": 1 - float(predict[0]),
                                                                "1": float(predict[0])}])

        return result

    def export_model(self):
        if self.model is None:
            return

        return {MODELMETA: self._get_model_meta(),
                MODELPARAM: self._get_model_param()}

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        param = model_dict.get(MODELPARAM)
        meta = model_dict.get(MODELMETA)

        self._build_model()
        self._restore_model_meta(meta)
        self._restore_model_param(param)

    def _get_model_meta(self):
        model_meta = HeteroNNMeta()
        model_meta.task_type = self.task_type

        model_meta.batch_size = self.batch_size
        model_meta.epochs = self.epochs
        model_meta.early_stop = self.early_stop
        model_meta.tol = self.tol
        # model_meta.interactive_layer_lr = self.hetero_nn_param.interacitve_layer_lr

        model_meta.hetero_nn_model_meta.CopyFrom(self.model.get_hetero_nn_model_meta())

        return model_meta

    def _get_model_param(self):
        model_param = HeteroNNParam()
        model_param.iter_epoch = self.iter_epoch
        model_param.hetero_nn_model_param.CopyFrom(self.model.get_hetero_nn_model_param())
        model_param.num_label = self.num_label

        for loss in self.history_loss:
            model_param.history_loss.append(loss)

        return model_param

    def prepare_batch_data(self, batch_generator, data_inst):
        batch_generator.initialize_batch_generator(data_inst, self.batch_size)
        batch_data_generator = batch_generator.generate_batch_data()

        for batch_data in batch_data_generator:
            keys, batch_x, batch_y = self._load_data(batch_data)
            self.data_x.append(batch_x)
            self.data_y.append(batch_y)
            self.data_keys.append(keys)

        self._convert_label()
        self.set_partition(data_inst)

    def _load_data(self, data_inst):
        data = list(data_inst.collect())
        data_keys = [key for (key, val) in data]
        data_keys_map = dict(zip(sorted(data_keys), range(len(data_keys))))

        keys = [None for idx in range(len(data_keys))]
        batch_x = [None for idx in range(len(data_keys))]
        batch_y = [None for idx in range(len(data_keys))]

        for (key, inst) in data:
            idx = data_keys_map[key]
            keys[idx] = key
            batch_x[idx] = inst.features
            batch_y[idx] = inst.label

            if self.input_shape is None:
                try:
                    self.input_shape = inst.features.shape[0]
                except AttributeError:
                    self.input_shape = 0

        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)

        return keys, batch_x, batch_y

    def _convert_label(self):
        diff_label = np.unique(np.concatenate(self.data_y))
        self.label_dict = dict(zip(diff_label, range(diff_label.shape[0])))

        transform_y = []
        self.num_label = diff_label.shape[0]

        if self.task_type == "regression" or self.num_label <= 2:
            for batch_y in self.data_y:
                new_batch_y = np.zeros((batch_y.shape[0], 1))
                for idx in range(new_batch_y.shape[0]):
                    new_batch_y[idx] = batch_y[idx]

                transform_y.append(new_batch_y)

            self.data_y = transform_y
            return

        for batch_y in self.data_y:
            new_batch_y = np.zeros((batch_y.shape[0], self.num_label))
            for idx in range(new_batch_y.shape[0]):
                y = batch_y[idx]
                new_batch_y[idx][y] = 1

            transform_y.append(new_batch_y)

        self.data_y = transform_y

    def _restore_model_param(self, param):
        super(HeteroNNGuest, self)._restore_model_param(param)
        self.num_label = param.num_label
