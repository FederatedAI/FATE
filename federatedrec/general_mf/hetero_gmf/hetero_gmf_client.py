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

from federatedml.util import consts
from arch.api.utils import log_utils
from arch.api import session as fate_session
from federatedml.statistic import data_overview
from federatedrec.optim.sync import user_num_transfer_sync
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
        self.item_num = None
        self.aggregator_iter = None

    def _init_model(self, param):
        super()._init_model(param)

        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = 1
        self.optimizer = param.optimizer
        self.loss = param.loss
        self.metrics = param.metrics
        self.data_converter = GMFDataConverter()
        self.user_num_sync.register_user_num_transfer(self.transfer_variable)

    def _check_monitored_status(self, data, epoch_degree):
        """
        check the model whether is converged or not
        :param data:
        :param epoch_degree:
        :return:
        """
        metrics = self._model.evaluate(data)
        LOGGER.info(f"metrics at iter {self.aggregator_iter}: {metrics}")
        loss = metrics["loss"]
        self.aggregator.send_loss(loss=loss,
                                  degree=epoch_degree,
                                  suffix=self._iter_suffix())
        return self.aggregator.get_converge_status(suffix=self._iter_suffix())

    def send_user_num(self, data):
        pass

    def get_user_num(self):
        pass

    def get_features_shape(self, data_instances):
        if self.feature_shape is not None:
            return self.feature_shape
        return data_overview.get_features_shape(data_instances)

    def fit(self, data_instances, validate_data=None):
        """
        train model
        :param data_instances: training data
        :param validate_data:  validation data
        :return:
        """
        data = self.data_converter.convert(data_instances, batch_size=self.batch_size,
                                           neg_count=self.model_param.neg_count)

        user_num = data.user_count
        item_num = data.item_count

        LOGGER.info(f'send user_num')
        self.send_user_num(user_num)
        LOGGER.info(f'get remote user_num')
        remote_user_num = self.get_user_num()
        LOGGER.info(f'local user num: {user_num}, remote user num: {remote_user_num}')
        self.user_num = max(remote_user_num, user_num)
        self.item_num = item_num

        self._model = GMFModel.build_model(self.user_num, item_num, self.params.init_param.embed_dim,
                                           self.loss, self.optimizer, self.metrics)

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
        """
        export model
        :return: saved model dict
        """
        meta = self._get_meta()
        param = self._get_param()
        model_dict = {self.model_meta_name: meta, self.model_param_name: param}
        LOGGER.info(f"model_dict keys: {model_dict.keys()}")
        return model_dict

    def _get_meta(self):
        """
        get meta data for saving model
        :return:
        """
        from federatedml.protobuf.generated import gmf_model_meta_pb2
        LOGGER.info(f"_get_meta")
        meta_pb = gmf_model_meta_pb2.GMFModelMeta()
        meta_pb.params.CopyFrom(self.model_param.generate_pb())
        meta_pb.aggregate_iter = self.aggregator_iter
        return meta_pb

    def _get_param(self):
        """
        get model param for saving model
        :return:
        """
        from federatedml.protobuf.generated import gmf_model_param_pb2
        LOGGER.info(f"_get_param")
        param_pb = gmf_model_param_pb2.GMFModelParam()
        param_pb.saved_model_bytes = self._model.export_model()
        param_pb.user_num = self.user_num
        param_pb.item_num = self.item_num
        return param_pb

    def predict(self, data_inst):
        """
        predicton function. Note that: GMF model use different DataConverter in evaluation and prediction procedure.
        :param data_inst: data instance
        :return: the prediction results
        """
        LOGGER.info(f"data_inst type: {type(data_inst)}, size: {data_inst.count()}, table name: {data_inst.get_name()}")
        LOGGER.info(f"current flowid: {self.flowid}")
        if self.flowid == 'validate':
            # use GMFSequenceData in evaluation procedure (after training procedure)
            data = self.data_converter.convert(data_inst, batch_size=self.batch_size, neg_count=self.model_param.neg_count,
                                               training=True, flow_id=self.flowid)
            keys = data.get_keys()
            labels = data.get_validate_labels()
            label_data = fate_session.parallelize(zip(keys, labels), include_key=True)
        else:
            # use GMFSequencePredictData in prediction procedure
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
        """
        load model from saved model, and initialize the model params
        :param model_dict:
        :return:
        """
        model_dict = list(model_dict["model"].values())[0]
        model_obj = model_dict.get(self.model_param_name)
        meta_obj = model_dict.get(self.model_meta_name)
        self.user_num = model_obj.user_num
        self.item_num = model_obj.item_num
        self.model_param.restore_from_pb(meta_obj.params)
        self._init_model(self.model_param)
        self.aggregator_iter = meta_obj.aggregate_iter
        self._model = GMFModel.restore_model(model_obj.saved_model_bytes, model_obj.user_num, model_obj.item_num, self.model_param.init_param.embed_dim)
        self._model.set_user_num(model_obj.user_num)
        self._model.set_item_num(model_obj.item_num)


class HeteroGMFHost(HeteroGMFClient):
    """
    Host HeteroGMF Class, implement the get_user_num and send_user_num function
    """

    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.user_num_sync = user_num_transfer_sync.Host()

    def send_user_num(self, data):
        self.user_num_sync.send_host_user_num(data)

    def get_user_num(self):
        return self.user_num_sync.get_guest_user_num()


class HeteroGMFGuest(HeteroGMFClient):
    """
    Guest HeteroGMF Class, implement the get_user_num and send_user_num function
    """

    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
        self.user_num_sync = user_num_transfer_sync.Guest()

    def send_user_num(self, data):
        self.user_num_sync.send_guest_user_num(data)

    def get_user_num(self):
        return self.user_num_sync.get_host_user_num()
