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
import json

from fate_arch.abc import CTableABC
from fate_arch.session import computing_session
from fate_flow.entity.metric import MetricMeta, Metric
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
from federatedml.util import consts, LOGGER
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter, HomoLabelEncoderClient

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

        self._summary = dict(loss_history=[], is_converged=False)

    def _init_model(self, param: HomoNNParam):
        super()._init_model(param=param)
        early_stop = self.model_param.early_stop
        self.converge_func = converge_func_factory(early_stop.converge_func, early_stop.eps).is_converge
        self.loss_consumed = early_stop.converge_func != "weight_diff"

    def callback_loss(self, iter_num, loss):
        # noinspection PyTypeChecker
        metric_meta = MetricMeta(name='train',
                                 metric_type="LOSS",
                                 extra_metas={
                                     "unit_name": "iters",
                                 })

        self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
        self.callback_metric(metric_name='loss',
                             metric_namespace='train',
                             metric_data=[Metric(iter_num, loss)])

        self._summary["loss_history"].append(loss)

    def _is_converged(self):
        loss = self.loss_scatter.weighted_loss_mean(suffix=self._suffix())
        LOGGER.info(f"loss at iter {self.aggregate_iteration_num}: {loss}")
        self.callback_loss(self.aggregate_iteration_num, loss)
        if self.loss_consumed:
            is_converged = self.converge_func(loss)
        else:
            is_converged = self.converge_func(self.model)
        self.has_converged.remote_converge_status(is_converge=is_converged, suffix=self._suffix())
        self._summary["is_converged"] = is_converged
        return is_converged

    def fit(self, data_inst):
        label_mapping = HomoLabelEncoderArbiter().label_alignment()
        LOGGER.info(f'label mapping: {label_mapping}')
        while self.aggregate_iteration_num < self.max_aggregate_iteration_num:
            self.model = self.aggregator.weighted_mean_model(suffix=self._suffix())
            self.aggregator.send_aggregated_model(model=self.model, suffix=self._suffix())

            if self._is_converged():
                LOGGER.info(f"early stop at iter {self.aggregate_iteration_num}")
                break
            self.aggregate_iteration_num += 1
        else:
            LOGGER.warn(f"reach max iter: {self.aggregate_iteration_num}, not converged")
        self.set_summary(self._summary)

    def save_model(self):
        return self.model


class HomoNNClient(HomoNNBase):

    def __init__(self, trans_var):
        super().__init__(trans_var=trans_var)
        self.aggregator = secure_mean_aggregator.Client(self.transfer_variable.secure_aggregator_trans_var)
        self.loss_scatter = loss_scatter.Client(self.transfer_variable.loss_scatter_trans_var)
        self.has_converged = has_converged.Client(self.transfer_variable.has_converged_trans_var)

        self.nn_model = None
        self._summary = dict(loss_history=[], is_converged=False)
        self._header = []
        self._label_align_mapping = None

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
        LOGGER.info(f"metrics at iter {self.aggregate_iteration_num}: {metrics}")
        loss = metrics["loss"]
        self.loss_scatter.send_loss(loss=(loss, epoch_degree), suffix=self._suffix())
        is_converged = self.has_converged.get_converge_status(suffix=self._suffix())
        self._summary["is_converged"] = is_converged
        self._summary["loss_history"].append(loss)
        return is_converged

    def _align_labels(self, data_inst):
        local_labels = data_inst.map(lambda k, v: [k, {v.label}]).reduce(lambda x, y: x | y)
        _, self._label_align_mapping = HomoLabelEncoderClient().label_alignment(local_labels)
        num_classes = len(self._label_align_mapping)

        if self.config_type == "pytorch":
            for layer in reversed(self.nn_define):
                if layer['layer'] == "Linear":
                    output_dim = layer['config'][1]
                    if output_dim == 1 and num_classes == 2:
                        return
                    layer['config'][1] = num_classes
                    return

        if self.config_type == "nn":
            for layer in reversed(self.nn_define):
                if layer['layer'] == "Dense":
                    output_dim = layer.get('units', None)
                    if output_dim == 1 and num_classes == 2:
                        return
                    layer['units'] = num_classes
                    return

        if self.config_type == "keras":
            layers = self.nn_define['config']['layers']
            for layer in reversed(layers):
                if layer['class_name'] == 'Dense':
                    output_dim = layer['config'].get('units', None)
                    if output_dim == 1 and num_classes == 2:
                        return
                    layer['config']['units'] = num_classes
                    return

    def fit(self, data_inst: CTableABC, *args):
        self._header = data_inst.schema["header"]
        self._align_labels(data_inst)
        data = self.data_converter.convert(data_inst, batch_size=self.batch_size, encode_label=self.encode_label,
                                           label_mapping=self._label_align_mapping)
        self.nn_model = self.model_builder(input_shape=data.get_shape()[0],
                                           nn_define=self.nn_define,
                                           optimizer=self.optimizer,
                                           loss=self.loss,
                                           metrics=self.metrics)

        epoch_degree = float(len(data)) * self.aggregate_every_n_epoch

        while self.aggregate_iteration_num < self.max_aggregate_iteration_num:
            LOGGER.info(f"start {self.aggregate_iteration_num}_th aggregation")

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
                LOGGER.info(f"early stop at iter {self.aggregate_iteration_num}")
                break

            LOGGER.info(f"role {self.role} finish {self.aggregate_iteration_num}_th aggregation")
            self.aggregate_iteration_num += 1
        else:
            LOGGER.warn(f"reach max iter: {self.aggregate_iteration_num}, not converged")

        self.set_summary(self._summary)

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
        param_pb.header.extend(self._header)
        for label, mapped in self._label_align_mapping.items():
            param_pb.label_mapping.add(label=json.dumps(label), mapped=json.dumps(mapped))
        return param_pb

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        model_obj = _extract_param(model_dict)
        meta_obj = _extract_meta(model_dict)
        self.model_param.restore_from_pb(meta_obj.params)
        self._init_model(self.model_param)
        self.aggregate_iteration_num = meta_obj.aggregate_iter
        self.nn_model = restore_nn_model(self.config_type, model_obj.saved_model_bytes)
        self._header = list(model_obj.header)
        self._label_align_mapping = {}
        for item in model_obj.label_mapping:
            label = json.loads(item.label)
            mapped = json.loads(item.mapped)
            self._label_align_mapping[label] = mapped

    @assert_io_num_rows_equal
    def predict(self, data_inst: CTableABC):
        self.align_data_header(data_instances=data_inst, pre_header=self._header)
        data = self.data_converter.convert(data_inst, batch_size=self.batch_size, encode_label=self.encode_label,
                                           label_mapping=self._label_align_mapping)
        predict = self.nn_model.predict(data)
        num_output_units = predict.shape[1]
        if num_output_units == 1:
            kv = zip(data.get_keys(), map(lambda x: x.tolist()[0], predict))
        else:
            kv = zip(data.get_keys(), predict.tolist())
        pred_tbl = computing_session.parallelize(kv, include_key=True, partition=data_inst.partitions)
        classes = [0, 1] if num_output_units == 1 else [i for i in range(num_output_units)]
        return self.predict_score_to_output(data_inst, pred_tbl, classes=classes,
                                            threshold=self.param.predict_param.threshold)


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


# server: Guest, clients: Hosts
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
