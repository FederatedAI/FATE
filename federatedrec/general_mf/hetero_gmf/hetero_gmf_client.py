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

import typing
import functools

from federatedml.feature.instance import Instance
from federatedml.util import consts
from arch.api.utils import log_utils
from arch.api import session as fate_session
from federatedml.statistic import data_overview
from federatedrec.optim.sync import user_ids_transfer_sync
from federatedrec.general_mf.hetero_gmf.hetero_gmf_base import HeteroGMFBase
from federatedrec.general_mf.hetero_gmf.backend import GMFModel
from federatedrec.general_mf.hetero_gmf.gmf_data_convertor import GMFDataConverter

LOGGER = log_utils.getLogger()


class HeteroGMFClient(HeteroGMFBase):
    def __init__(self):
        super(HeteroGMFClient, self).__init__()

        self._model = None
        self.feature_shape = None
        self.user_num = None

    def _init_model(self, param):
        super()._init_model(param)

        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = 1
        self.optimizer = param.optimizer
        self.loss = param.loss
        self.metrics = param.metrics
        self.data_converter = GMFDataConverter()
        self.user_ids_sync.register_user_ids_transfer(self.transfer_variable)

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

    def send_user_num(self, data):
        pass

    def get_user_num(self):
        pass

    def get_features_shape(self, data_instances):
        if self.feature_shape is not None:
            return self.feature_shape
        return data_overview.get_features_shape(data_instances)

    def fit(self, data_instances, validate_data=None):
        data = self.data_converter.convert(data_instances, batch_size=self.batch_size
                                           , neg_count=self.model_param.neg_count)

        user_ids = data.unique_user_ids
        item_ids = data.unique_items_ids
        user_num = data.user_count

        LOGGER.info(f'send user_num')
        self.send_user_num(user_num)
        LOGGER.info(f'get remote user_num')
        remote_user_num = self.get_user_num()
        LOGGER.info(f'local user num: {user_num}, remote user num: {remote_user_num}')
        self.user_num = max(remote_user_num, user_num)

        self._model = GMFModel.build_model(user_ids, item_ids, self.params.init_param.embed_dim,
                                           self.loss, self.optimizer, self.metrics, user_num=self.user_num)

        epoch_degree = float(len(data))

        while self.aggregator_iter < self.max_iter:
            LOGGER.info(f"start {self.aggregator_iter}_th aggregation")

            # train
            LOGGER.debug(f"begin train")
            self._model.train(data, aggregate_every_n_epoch=self.aggregate_every_n_epoch)
            LOGGER.debug(f"after train")

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
        model_dict = {self.model_meta_name: meta, self.model_param_name: param}
        LOGGER.info(f"model_dict keys: {model_dict.keys()}")
        return model_dict

    def _get_meta(self):
        from federatedrec.protobuf.generated import gmf_model_meta_pb2
        LOGGER.info(f"_get_meta")
        meta_pb = gmf_model_meta_pb2.GMFModelMeta()
        meta_pb.params.CopyFrom(self.model_param.generate_pb())
        meta_pb.aggregate_iter = self.aggregator_iter
        return meta_pb

    def _get_param(self):
        from federatedrec.protobuf.generated import gmf_model_param_pb2
        LOGGER.info(f"_get_param")
        param_pb = gmf_model_param_pb2.GMFModelParam()
        param_pb.saved_model_bytes = self._model.export_model()
        param_pb.user_ids.extend(self._model.user_ids)
        param_pb.item_ids.extend(self._model.item_ids)

        return param_pb

    def predict(self, data_inst):
        LOGGER.info(f"data_inst type: {type(data_inst)}, size: {data_inst.count()}, table name: {data_inst.get_name()}")
        LOGGER.info(f"current flowid: {self.flowid}")
        if self.flowid == 'validate':
            # use GMFSequenceData
            data = self.data_converter.convert(data_inst, batch_size=self.batch_size, neg_count=self.model_param.neg_count
                                               , training=True, flow_id=self.flowid)
            keys = data.get_keys()
            labels = data.get_validate_labels()
            label_data = fate_session.parallelize(zip(keys, labels), include_key=True)
        else:
            # use GMFSequencePredictData
            data = self.data_converter.convert(data_inst, batch_size=self.batch_size, training=False)
            label_data = data_inst.map(lambda k, v: (k, v.features.astype(int).tolist()[2]))
        LOGGER.info(f"label_data example: {label_data.take(10)}")
        LOGGER.info(f"data example: {data_inst.first()[1].features.astype(int)}")
        LOGGER.info(f"converted data, size :{data.size}")
        predict = self._model.predict(data)
        LOGGER.info(f"predict shape: {predict.shape}")
        threshold = self.params.predict_param.threshold

        kv = [(x[0], (0 if x[1] <= threshold else 1, x[1].item())) for x in zip(data.get_keys(), predict)]
        pred_tbl = fate_session.parallelize(kv, include_key=True)
        pred_data = label_data.join(pred_tbl, lambda d, pred: [d, pred[0], pred[1], {"label": pred[0]}])
        LOGGER.info(f"pred_data sample: {pred_data.take(20)}")
        return pred_data

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        model_obj = model_dict.get(self.model_param_name)
        meta_obj = model_dict.get(self.model_meta_name)
        self.model_param.restore_from_pb(meta_obj.params)
        self._init_model(self.model_param)
        self.aggregator_iter = meta_obj.aggregate_iter
        self._model = GMFModel.restore_model(model_obj.saved_model_bytes)
        self._model.set_user_ids(model_obj.user_ids)
        self._model.set_item_ids(model_obj.item_ids)


class HeteroGMFHost(HeteroGMFClient):

    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.user_ids_sync = user_ids_transfer_sync.Host()

    def send_user_ids(self, data):
        self.user_ids_sync.send_host_user_ids(data)

    def get_user_ids(self):
        return self.user_ids_sync.get_guest_user_ids()

    def send_user_num(self, data):
        self.user_ids_sync.send_host_user_num(data)

    def get_user_num(self):
        return self.user_ids_sync.get_guest_user_num()


class HeteroGMFGuest(HeteroGMFClient):

    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
        self.user_ids_sync = user_ids_transfer_sync.Guest()

    def send_user_ids(self, data):
        self.user_ids_sync.send_guest_user_ids(data)

    def get_user_ids(self):
        return self.user_ids_sync.get_host_user_ids()

    def send_user_num(self, data):
        self.user_ids_sync.send_guest_user_num(data)

    def get_user_num(self):
        return self.user_ids_sync.get_host_user_num()
