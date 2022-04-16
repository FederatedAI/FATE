from re import X
from federatedml.nn.homo_nn.enter_point import HomoNNDefaultTransVar
from federatedml.util import LOGGER
from fate_arch.computing import is_table
from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.has_converged import HasConvergedTransVar
from federatedml.framework.homo.blocks.loss_scatter import LossScatterTransVar
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.model_base import ModelBase
from federatedml.nn.homo_nn._consts import _extract_meta, _extract_param
from federatedml.param.graphNN_param import GraphNNParam
from federatedml.util import LOGGER
from federatedml.nn.backend.pytorch.nn_model import (layers, PytorchGNNModel)
from federatedml.framework.homo.blocks import (
    secure_mean_aggregator,
    loss_scatter,
    has_converged,
)
import torch
from federatedml.util.homo_label_encoder import (
    HomoLabelEncoderArbiter,
    HomoLabelEncoderClient,
)
from torch.nn import *
import json
from fate_arch.session import computing_session
import math
from federatedml.nn.backend.pytorch.layer import GCNLayer
from federatedml.nn.backend.pytorch.nn_model import PytorchData, PytorchGraphData


class GraphNNServer(ModelBase):
    def __init__(self):
        LOGGER.info("*********GraphNNServer*********")
        super().__init__()
        self.model_param = GraphNNParam()
        self.transfer_variable = HomoNNDefaultTransVar()
        self._api_version = 0

    def _init_model(self, param: GraphNNParam):
        super()._init_model(param)
        from federatedml.nn.homo_nn import _version_0
        _version_0.server_init_model(self, param)

    def callback_loss(self, iter_num, loss):
        # noinspection PyTypeChecker
        metric_meta = MetricMeta(
            name="train",
            metric_type="LOSS",
            extra_metas={
                "unit_name": "iters",
            },
        )

        self.callback_meta(
            metric_name="loss", metric_namespace="train", metric_meta=metric_meta
        )
        self.callback_metric(
            metric_name="loss",
            metric_namespace="train",
            metric_data=[Metric(iter_num, loss)],
        )

    def fit(self, data_inst):
        LOGGER.info("******GraphNNServer.fit()**********")
        if True:
            from federatedml.nn.homo_nn import _version_0
            _version_0.server_fit(self=self, data_inst=data_inst)



class GraphNNClient(ModelBase):
    def __init__(self):
        LOGGER.info("*********GraphNNClient*********")
        super().__init__()
        self.model_param = GraphNNParam()
        self.transfer_variable = HomoNNDefaultTransVar()
        self._api_version = 0
        self._trainer = ...

    def _build_pytorch(self, nn_define, optimizer, loss, metrics, **kwargs):
        LOGGER.info(loss)
        model = torch.nn.ModuleList()
        nn_define = json.loads(nn_define)
 
        for config in nn_define:
            if config['type'] == 'Linear':
                layer = Linear(in_features = config['in_features'], out_features=config['out_features'], bias=config['bias'])
            elif config['type'] == 'ReLU':
                layer = ReLU()
            elif config['type'] == 'Sigmoid':
                layer = Sigmoid()
            elif config['type'] == 'GCNLayer':
                layer = GCNLayer(in_features = config['in_features'], out_features=config['out_features'], bias=config['bias'])
            elif config['type'] == 'LogSoftmax':
                layer = LogSoftmax()
            else:
                raise NotImplementedError
            model.append(layer)
        return PytorchGNNModel(model, optimizer, loss, metrics)

    def convert_feats(self, data, *args, **kwargs):
        return PytorchData(data, *args, **kwargs)

    def convert_adj(self, data):
        return PytorchGraphData(data).edges

    def _client_set_params(self, param):
        self.nn_model = None
        self._summary = dict(loss_history=[], is_converged=False)
        self._header = []
        self._label_align_mapping = None

        self.param = param
        self.enable_secure_aggregate = param.secure_aggregate
        self.max_aggregate_iteration_num = param.max_iter
        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = param.aggregate_every_n_epoch
        self.nn_define = param.nn_define
        self.config_type = param.config_type
        self.optimizer = param.optimizer
        self.loss = param.loss
        self.metrics = param.metrics
        self.encode_label = param.encode_label

    def _suffix(self):
        return (self.aggregate_iteration_num,)

    def client_is_converged(self, feats, adj, epoch_degree):
        metrics = self.nn_model.evaluate(feats, adj)
        LOGGER.info(f"metrics at iter {self.aggregate_iteration_num}: {metrics}")
        loss = metrics["loss"]
        self.loss_scatter.send_loss(loss=(loss, epoch_degree), suffix=self._suffix())
        is_converged = self.has_converged.get_converge_status(suffix=self._suffix())
        self._summary["is_converged"] = is_converged
        self._summary["loss_history"].append(loss)
        return is_converged        

    def _init_model(self, param: GraphNNParam):
        LOGGER.info("**********GraphNNClient.init_model()********")
        self.aggregate_iteration_num = 0
        self.aggregator = secure_mean_aggregator.Client(
            self.transfer_variable.secure_aggregator_trans_var
        )
        self.loss_scatter = loss_scatter.Client(
            self.transfer_variable.loss_scatter_trans_var
        )
        self.has_converged = has_converged.Client(
            self.transfer_variable.has_converged_trans_var
        )
        self._client_set_params(param)

    def _client_align_labels(self, data_inst):
        local_labels = data_inst.map(lambda k, v: [k, {v.label}]).reduce(lambda x, y: x | y)
        _, self._label_align_mapping = HomoLabelEncoderClient().label_alignment(
            local_labels
        )

    def fit(self, data_raw):
        LOGGER.info("*******GraphNNClient.fit()*******")
        data_feats, data_adj = data_raw[0], data_raw[1]
        self._header = data_feats.schema["header"]
        self._client_align_labels(data_inst=data_feats)
        LOGGER.info("num_label="+str(len(self._label_align_mapping)))
        LOGGER.info("batch_size="+str(self.batch_size))
        LOGGER.info("encode_lable="+str(self.encode_label))
        feats = self.convert_feats(
            data_feats,
            batch_size=self.batch_size,
            encode_label=self.encode_label,
            label_mapping=self._label_align_mapping,
        )

        adj = self.convert_adj(data_adj)

        self.nn_model = self._build_pytorch(
            input_shape=feats.get_shape()[0],
            nn_define=self.nn_define,
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        epoch_degree = float(len(feats)) * self.aggregate_every_n_epoch

        while self.aggregate_iteration_num < self.max_aggregate_iteration_num:
            LOGGER.info(f"start {self.aggregate_iteration_num}_th aggregation")

            # train
            self.nn_model.train(feats, adj, aggregate_every_n_epoch=self.aggregate_every_n_epoch)

            # send model for aggregate, then set aggregated model to local
            self.aggregator.send_weighted_model(
                weighted_model=self.nn_model.get_model_weights(),
                weight=epoch_degree * self.aggregate_every_n_epoch,
                suffix=self._suffix(),
            )
            weights = self.aggregator.get_aggregated_model(suffix=self._suffix())
            self.nn_model.set_model_weights(weights=weights)

            # calc loss and check convergence
            if self.client_is_converged(feats, adj, epoch_degree):
                LOGGER.info(f"early stop at iter {self.aggregate_iteration_num}")
                break

            LOGGER.info(
                f"role {self.role} finish {self.aggregate_iteration_num}_th aggregation"
            )
            self.aggregate_iteration_num += 1
        else:
            LOGGER.warn(f"reach max iter: {self.aggregate_iteration_num}, not converged")

        self.set_summary(self._summary)        

    def predict(self, data_raw):
        LOGGER.info("***********GraphNNClient.predict()**********")
        data_feats, data_adj = data_raw
        self.align_data_header(data_instances=data_feats, pre_header=self._header)
        feats = self.convert_feats(
            data_feats,
            batch_size=self.batch_size,
            encode_label=self.encode_label,
            label_mapping=self._label_align_mapping,
        )
        adj = self.convert_adj(data_adj)
        predict = self.nn_model.predict(feats, adj)

        num_output_units = predict.shape[1]
        if num_output_units == 1:
            kv = zip(feats.get_keys(), map(lambda x: x.tolist()[0], predict))
        else:
            kv = zip(feats.get_keys(), predict.tolist())
        pred_tbl = computing_session.parallelize(
            kv, include_key=True, partition=data_feats.partitions
        )
        classes = [0, 1] if num_output_units == 1 else [i for i in range(num_output_units)]
        return self.predict_score_to_output(
            data_feats,
            pred_tbl,
            classes=classes,
            threshold=self.param.predict_param.threshold,
        )

    def export_model(self):
        LOGGER.info("***********GraphNNClient.export_model()**********")
        from federatedml.nn.homo_nn import _version_0
        return _version_0.client_export_model(self=self)

    def client_set_params(self, param):
        self.nn_model = None
        self._summary = dict(loss_history=[], is_converged=False)
        self._header = []
        self._label_align_mapping = None

        self.param = param
        self.enable_secure_aggregate = param.secure_aggregate
        self.max_aggregate_iteration_num = param.max_iter
        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = param.aggregate_every_n_epoch
        self.nn_define = param.nn_define
        self.config_type = param.config_type
        self.optimizer = param.optimizer
        self.loss = param.loss
        self.metrics = param.metrics
        self.encode_label = param.encode_label

    def client_load_model(self, meta_obj, model_obj):
        self.model_param.restore_from_pb(meta_obj.params)
        self.client_set_params(self.model_param)
        self.aggregate_iteration_num = meta_obj.aggregate_iter
        self.nn_model = PytorchGNNModel.restore_model(model_obj.saved_model_bytes)
        self._header = list(model_obj.header)
        self._label_align_mapping = {}
        for item in model_obj.label_mapping:
            label = json.loads(item.label)
            mapped = json.loads(item.mapped)
            self._label_align_mapping[label] = mapped

    def load_model(self, model_dict):
        LOGGER.info("***********GraphNNClient.load_model()**********")
        # LOGGER.info(model_dict)
        model_dict = list(model_dict["model"].values())[0]
        model_obj = _extract_param(model_dict)
        meta_obj = _extract_meta(model_dict)
        self.client_load_model(
            meta_obj=meta_obj, model_obj=model_obj
        )





    
