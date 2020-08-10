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

from arch.api import session
from arch.api.utils.log_utils import LoggerFactory
from fate_flow.entity.metric import MetricType, MetricMeta, Metric
from federatedml.framework.homo.blocks import secure_mean_aggregator, loss_scatter, has_converged
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.has_converged import HasConvergedTransVar
from federatedml.framework.homo.blocks.loss_scatter import LossScatterTransVar
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.model_base import ModelBase
from federatedml.nn.homo_nn import nn_model
from federatedml.nn.homo_nn.nn_model import restore_nn_model
from federatedml.optim.convergence import converge_func_factory
from federatedml.param.homo_nn_param import HomoNNParam
from federatedml.util import consts
from federatedml.util.io_check import assert_io_num_rows_equal

Logger = LoggerFactory.get_logger()
MODEL_META_NAME = "HomoNNModelMeta"
MODEL_PARAM_NAME = "HomoNNModelParam"


def _build_model_dict(meta, param):
    return {MODEL_META_NAME: meta, MODEL_PARAM_NAME: param}


def _extract_param(model_dict: dict):
    return model_dict.get(MODEL_PARAM_NAME, None)


def _extract_meta(model_dict: dict):
    return model_dict.get(MODEL_META_NAME, None)


class HomoNNBase(ModelBase):
    def __init__(self, trans_var):
        super().__init__()
        self.model_param = HomoNNParam()
        self.aggregate_iteration_num = 0
        self.transfer_variable = trans_var

    def _suffix(self):
        return self.aggregate_iteration_num,

    def _init_model(self, param: HomoNNParam):
        self.param = param
        self.enable_secure_aggregate = param.secure_aggregate
        self.max_aggregate_iteration_num = param.max_iter


class HomoNNServer(HomoNNBase):

    def __init__(self, trans_var):
        super().__init__(trans_var=trans_var)
        self.model = None

        self.aggregator = secure_mean_aggregator.Server(self.transfer_variable.secure_aggregator_trans_var)
        self.loss_scatter = loss_scatter.Server(self.transfer_variable.loss_scatter_trans_var)
        self.has_converged = has_converged.Server(self.transfer_variable.has_converged_trans_var)

    def _init_model(self, param: HomoNNParam):
        super()._init_model(param=param)
        early_stop = self.model_param.early_stop
        self.converge_func = converge_func_factory(early_stop.converge_func, early_stop.eps).is_converge
        self.loss_consumed = early_stop.converge_func != "weight_diff"

    def callback_loss(self, iter_num, loss):
        metric_meta = MetricMeta(name='train',
                                 metric_type="LOSS",
                                 extra_metas={
                                     "unit_name": "iters",
                                 })

        self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
        self.callback_metric(metric_name='loss',
                             metric_namespace='train',
                             metric_data=[Metric(iter_num, loss)])

    def _is_converged(self):
        loss = self.loss_scatter.weighted_loss_mean(suffix=self._suffix())
        Logger.info(f"loss at iter {self.aggregate_iteration_num}: {loss}")
        self.callback_loss(self.aggregate_iteration_num, loss)
        if self.loss_consumed:
            is_converged = self.converge_func(loss)
        else:
            is_converged = self.converge_func(self.model)
        self.has_converged.remote_converge_status(is_converge=is_converged, suffix=self._suffix())
        return is_converged

    def fit(self, data_inst):
        while self.aggregate_iteration_num < self.max_aggregate_iteration_num:
            self.model = self.aggregator.weighted_mean_model(suffix=self._suffix())
            self.aggregator.send_aggregated_model(model=self.model, suffix=self._suffix())

            if self._is_converged():
                Logger.info(f"early stop at iter {self.aggregate_iteration_num}")
                break
            self.aggregate_iteration_num += 1
        else:
            Logger.warn(f"reach max iter: {self.aggregate_iteration_num}, not converged")

    def save_model(self):
        return self.model


class HomoNNClient(HomoNNBase):

    def __init__(self, trans_var):
        super().__init__(trans_var=trans_var)
        self.aggregator = secure_mean_aggregator.Client(self.transfer_variable.secure_aggregator_trans_var)
        self.loss_scatter = loss_scatter.Client(self.transfer_variable.loss_scatter_trans_var)
        self.has_converged = has_converged.Client(self.transfer_variable.has_converged_trans_var)

        self.nn_model = None

    def _init_model(self, param: HomoNNParam):
        super()._init_model(param=param)
        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = param.aggregate_every_n_epoch
        self.nn_define = param.nn_define
        self.config_type = param.config_type
        self.optimizer = param.optimizer
        self.loss = param.loss
        self.metrics = param.metrics
        self.encode_label = param.encode_label

        self.data_converter = nn_model.get_data_converter(self.config_type)
        self.model_builder = nn_model.get_nn_builder(config_type=self.config_type)

    def _is_converged(self, data, epoch_degree):
        metrics = self.nn_model.evaluate(data)
        Logger.info(f"metrics at iter {self.aggregate_iteration_num}: {metrics}")
        loss = metrics["loss"]
        self.loss_scatter.send_loss(loss=(loss, epoch_degree), suffix=self._suffix())
        is_converged = self.has_converged.get_converge_status(suffix=self._suffix())
        return is_converged

    def __build_nn_model(self, input_shape):
        self.nn_model = self.model_builder(input_shape=input_shape,
                                           nn_define=self.nn_define,
                                           optimizer=self.optimizer,
                                           loss=self.loss,
                                           metrics=self.metrics)

    def __build_pytorch_model(self, nn_define):
        self.nn_model = self.model_builder(nn_define=nn_define,
                                           optimizer=self.optimizer,
                                           loss=self.loss,
                                           metrics=self.metrics)

    def fit(self, data_inst, *args):
        data = self.data_converter.convert(data_inst, batch_size=self.batch_size, encode_label=self.encode_label)
        if self.config_type == "pytorch":
            self.__build_pytorch_model(self.nn_define)
        else:
            self.__build_nn_model(data.get_shape()[0])

        epoch_degree = float(len(data)) * self.aggregate_every_n_epoch

        while self.aggregate_iteration_num < self.max_aggregate_iteration_num:
            Logger.info(f"start {self.aggregate_iteration_num}_th aggregation")

            # train
            self.nn_model.train(data, aggregate_every_n_epoch=self.aggregate_every_n_epoch)

            # send model for aggregate, then set aggregated model to local
            self.aggregator.send_weighted_model(weighted_model=self.nn_model.get_model_weights(),
                                                weight=epoch_degree * self.aggregate_every_n_epoch,
                                                suffix=self._suffix())
            weights = self.aggregator.get_aggregated_model(suffix=self._suffix())
            self.nn_model.set_model_weights(weights=weights)

            # calc loss and check convergence
            if self._is_converged(data, epoch_degree):
                Logger.info(f"early stop at iter {self.aggregate_iteration_num}")
                break

            Logger.info(f"role {self.role} finish {self.aggregate_iteration_num}_th aggregation")
            self.aggregate_iteration_num += 1
        else:
            Logger.warn(f"reach max iter: {self.aggregate_iteration_num}, not converged")

    def export_model(self):
        return _build_model_dict(meta=self._get_meta(), param=self._get_param())

    def _get_meta(self):
        from federatedml.protobuf.generated import nn_model_meta_pb2
        meta_pb = nn_model_meta_pb2.NNModelMeta()
        meta_pb.params.CopyFrom(self.model_param.generate_pb())
        meta_pb.aggregate_iter = self.aggregate_iteration_num
        return meta_pb

    def _get_param(self):
        from federatedml.protobuf.generated import nn_model_param_pb2
        param_pb = nn_model_param_pb2.NNModelParam()
        param_pb.saved_model_bytes = self.nn_model.export_model()
        return param_pb

    @assert_io_num_rows_equal
    def predict(self, data_inst):

        data = self.data_converter.convert(data_inst, batch_size=self.batch_size, encode_label=self.encode_label)
        predict = self.nn_model.predict(data)
        num_output_units = predict.shape[1]
        threshold = self.param.predict_param.threshold

        if num_output_units == 1:
            kv = [(x[0], (0 if x[1][0] <= threshold else 1, x[1][0].item())) for x in zip(data.get_keys(), predict)]
            pred_tbl = session.parallelize(kv, include_key=True, partition=data_inst.get_partitions())
            return data_inst.join(pred_tbl,
                                  lambda d, pred: [d.label, pred[0], pred[1], {"0": 1 - pred[1], "1": pred[1]}])
        else:
            kv = [(x[0], (x[1].argmax(), [float(e) for e in x[1]])) for x in zip(data.get_keys(), predict)]
            pred_tbl = session.parallelize(kv, include_key=True, partition=data_inst.get_partitions())
            return data_inst.join(pred_tbl,
                                  lambda d, pred: [d.label, pred[0].item(),
                                                   pred[1][pred[0]],
                                                   {str(v): pred[1][v] for v in range(len(pred[1]))}])
    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        model_obj = _extract_param(model_dict)
        meta_obj = _extract_meta(model_dict)
        self.model_param.restore_from_pb(meta_obj.params)
        self._init_model(self.model_param)
        self.aggregate_iteration_num = meta_obj.aggregate_iter
        self.nn_model = restore_nn_model(self.config_type, model_obj.saved_model_bytes)


# server: Arbiter, clients: Guest and Hosts
class HomoNNDefaultTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.secure_aggregator_trans_var = SecureAggregatorTransVar(server=server, clients=clients, prefix=self.prefix)
        self.loss_scatter_trans_var = LossScatterTransVar(server=server, clients=clients, prefix=self.prefix)
        self.has_converged_trans_var = HasConvergedTransVar(server=server, clients=clients, prefix=self.prefix)


class HomoNNDefaultClient(HomoNNClient):

    def __init__(self):
        super().__init__(trans_var=HomoNNDefaultTransVar())


class HomoNNDefaultServer(HomoNNServer):
    def __init__(self):
        super().__init__(trans_var=HomoNNDefaultTransVar())


# server: Arbiter, clients: Guest and Hosts
class HomoNNGuestServerTransVar(HomoNNDefaultTransVar):
    def __init__(self, server=(consts.GUEST,), clients=(consts.HOST,), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)


class HomoNNGuestServerClient(HomoNNClient):
    def __init__(self):
        super().__init__(trans_var=HomoNNGuestServerTransVar())


class HomoNNGuestServerServer(HomoNNServer):

    def __init__(self):
        super().__init__(trans_var=HomoNNGuestServerTransVar())


# server: Arbiter, clients: Hosts
class HomoNNArbiterSubmitTransVar(HomoNNDefaultTransVar):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.HOST,), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)


class HomoNNArbiterSubmitClient(HomoNNClient):
    def __init__(self):
        super().__init__(trans_var=HomoNNArbiterSubmitTransVar())


class HomoNNArbiterSubmitServer(HomoNNServer):

    def __init__(self):
        super().__init__(trans_var=HomoNNArbiterSubmitTransVar())
