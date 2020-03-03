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

import functools
import typing

import numpy as np
from arch.api import session as fate_session
from arch.api.utils import log_utils
from federatedml.util import consts
from federatedrec.optim.sync import user_ids_transfer_sync, average_rate_transfer_sync
from federatedrec.svdpp.hetero_svdpp.hetero_svdpp_base import HeteroSVDppBase
from federatedrec.svdpp.hetero_svdpp.backend import KerasSeqDataConverter, SVDppModel
# from federatedrec.svdpp.hetero_svdpp.backend import TFSeqDataConverter, SVDppModel

LOGGER = log_utils.getLogger()


class HeteroSVDppClient(HeteroSVDppBase):
    def __init__(self):
        super(HeteroSVDppClient, self).__init__()

        self._model = None

    def _init_model(self, param):
        super()._init_model(param)

        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = 1
        self.optimizer = param.optimizer
        self.loss = param.loss
        self.metrics = param.metrics
        self.data_converter = KerasSeqDataConverter()
        self.user_ids_sync.register_user_ids_transfer(self.transfer_variable)
        self.average_rate_sync.register_average_rate_transfer(self.transfer_variable)

    def _check_monitored_status(self, data, epoch_degree):
        metrics = self._model.evaluate(data)
        LOGGER.info(f"metrics at iter {self.aggregator_iter}: {metrics}")
        loss = metrics["loss"]
        self.aggregator.send_loss(loss=loss,
                                  degree=epoch_degree,
                                  suffix=self._iter_suffix())
        return self.aggregator.get_converge_status(suffix=self._iter_suffix())

    def send_user_ids(self, data):
        pass

    def get_user_ids(self):
        pass

    def send_average_rate(self, data):
        pass

    def get_average_rate(self):
        pass

    def join_average_rate(self, rates_table):
        # TODO: use map partition for parallelization
        rates = [float(_r) for _,_r in rates_table.collect()]
        # LOGGER.info(f"rates len {len(rates)}, data {rates}")

        average_rate = np.average(rates)
        sample_num = len(rates)

        self.send_average_rate([average_rate, sample_num])
        received_average_rate = self.get_average_rate()
        LOGGER.info(f"received average rate {received_average_rate}")

        average_rate = (average_rate + received_average_rate[0]) / (sample_num + received_average_rate[1])
        return average_rate

    def join_user_ids(self, user_ids_table):
        self.send_user_ids(user_ids_table)
        received_user_ids = self.get_user_ids()
        join_user_ids = user_ids_table.union(received_user_ids)
        user_ids = sorted([_id for (_id, _) in join_user_ids.collect()])
        LOGGER.debug(f"after join, get user ids {user_ids}")
        return user_ids

    def fit(self, data_instances, validate_data=None):
        LOGGER.debug("Start data count: {}".format(data_instances.count()))

        # self._abnormal_detection(data_instances)
        # self.init_schema(data_instances)
        # validation_strategy = self.init_validation_strategy(data_instances, validate_data)

        # parse data instances
        user_ids_table, item_ids_table, rates_table = self.extract_ids(data_instances)

        # get user ids
        user_ids = self.join_user_ids(user_ids_table)

        # get item ids
        item_ids = [_id for (_id, _) in item_ids_table.collect()]

        # get average rate
        average_rate = self.join_average_rate(rates_table)

        # Convert data_inst to keras sequence data
        data = self.data_converter.convert(data_instances, user_ids, item_ids, batch_size=self.batch_size)
        self._model = SVDppModel.build_model(user_ids, item_ids, data.rating_matrix,
                                             self.params.init_param.embed_dim,
                                             self.loss, self.optimizer, self.metrics, average_rate)

        epoch_degree = float(len(data))

        while self.aggregator_iter < self.max_iter:
            LOGGER.info(f"start {self.aggregator_iter}_th aggregation")

            # train
            self._model.train(data, aggregate_every_n_epoch=self.aggregate_every_n_epoch)

            # send model for aggregate, then set aggregated model to local
            modify_func: typing.Callable = functools.partial(self.aggregator.aggregate_then_get,
                                                             degree=epoch_degree * self.aggregate_every_n_epoch,
                                                             suffix=self._iter_suffix())
            self._model.modify(modify_func)

            # calc loss and check convergence
            if self._check_monitored_status(data, epoch_degree):
                LOGGER.info(f"early stop at iter {self.aggregator_iter}")
                break

            LOGGER.info(f"role {self.role} finish {self.aggregator_iter}_th aggregation")
            self.aggregator_iter += 1
        else:
            LOGGER.warn(f"reach max iter: {self.aggregator_iter}, not converged")

    def export_model(self):
        meta = self._get_meta()
        param = self._get_param()
        return {self.model_meta_name: meta, self.model_param_name: param}

    def _get_meta(self):
        from federatedml.protobuf.generated import svdpp_model_meta_pb2
        meta_pb = svdpp_model_meta_pb2.SVDppModelMeta()
        meta_pb.params.CopyFrom(self.model_param.generate_pb())
        meta_pb.aggregate_iter = self.aggregator_iter
        return meta_pb

    def _get_param(self):
        from federatedml.protobuf.generated import svdpp_model_param_pb2
        param_pb = svdpp_model_param_pb2.SVDppModelParam()
        param_pb.saved_model_bytes = self._model.export_model()
        param_pb.user_ids.extend(self._model.user_ids)
        param_pb.item_ids.extend(self._model.item_ids)

        # export rating matrix
        mat = svdpp_model_param_pb2.Mat()
        mat.height = self._model.rating_matrix.shape[0]
        mat.width = self._model.rating_matrix.shape[1]
        mat.values.extend(self._model.rating_matrix.flatten().astype(np.int32))
        param_pb.rating_matrix.CopyFrom(mat)

        # LOGGER.debug(f"rating matrix {param_pb.rating_matrix}")

        return param_pb

    def predict(self, data_inst):
        data = self.data_converter.convert(data_inst, self._model.user_ids, self._model.item_ids,
                                           self.batch_size, self._model.rating_matrix)
        predict = self._model.predict(data)
        threshold = self.params.predict_param.threshold

        kv = [(x[0], (0 if x[1][0] <= threshold else 1, x[1][0].item())) for x in zip(data.get_keys(), predict)]
        pred_tbl = fate_session.parallelize(kv, include_key=True)
        return data_inst.join(pred_tbl, lambda d, pred: [float(d.features.get_data(2)), pred[0], pred[1], {"label": pred[0]}])

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        model_obj = model_dict.get(self.model_param_name)
        meta_obj = model_dict.get(self.model_meta_name)
        LOGGER.debug(f"model_obj {type(model_obj)}")

        self.model_param.restore_from_pb(meta_obj.params)
        self._init_model(self.model_param)
        self.aggregator_iter = meta_obj.aggregate_iter
        self._model = SVDppModel.restore_model(model_obj.saved_model_bytes)
        self._model.set_user_ids(model_obj.user_ids)
        self._model.set_item_ids(model_obj.item_ids)
        mat = model_obj.rating_matrix
        rating_matrix = np.array(mat.values).reshape((mat.height, mat.width))
        self._model.set_rating_matrix(rating_matrix)


class HeteroSVDppHost(HeteroSVDppClient):

    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.user_ids_sync = user_ids_transfer_sync.Host()
        self.average_rate_sync = average_rate_transfer_sync.Host()

    def send_user_ids(self, data):
        self.user_ids_sync.send_host_user_ids(data)

    def get_user_ids(self):
        return self.user_ids_sync.get_guest_user_ids()

    def send_average_rate(self, data):
        self.average_rate_sync.send_host_average_rate(data)

    def get_average_rate(self):
        return self.average_rate_sync.get_guest_average_rate()

class HeteroSVDppGuest(HeteroSVDppClient):

    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
        self.user_ids_sync = user_ids_transfer_sync.Guest()
        self.average_rate_sync = average_rate_transfer_sync.Guest()

    def send_user_ids(self, data):
        self.user_ids_sync.send_guest_user_ids(data)

    def get_user_ids(self):
        return self.user_ids_sync.get_host_user_ids()

    def send_average_rate(self, data):
        self.average_rate_sync.send_guest_average_rate(data)

    def get_average_rate(self):
        return self.average_rate_sync.get_host_average_rate()

