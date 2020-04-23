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
from cv_task import dataloader_detector
import torch
from torch.autograd import Variable
from federatedml.nn.backend.pytorch.nn_model import PytorchNNModel
from cv_task import net
from torch.utils.data import DataLoader
import numpy as np
import time


Logger = LoggerFactory.get_logger()
MODEL_META_NAME = "HomoNNModelMeta"
MODEL_PARAM_NAME = "HomoNNModelParam"


def _build_model_dict(meta, param):
    return {MODEL_META_NAME: meta, MODEL_PARAM_NAME: param}


def _extract_param(model_dict: dict):
    return model_dict.get(MODEL_PARAM_NAME, None)


def _extract_meta(model_dict: dict):
    return model_dict.get(MODEL_META_NAME, None)


class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

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

        self.data_converter = nn_model.get_data_converter(self.config_type)
        self.model_builder = nn_model.get_nn_builder(config_type=self.config_type)

    def _is_converged(self, data, epoch_degree):
        if self.config_type=="cv":
            loss = data
        else:
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

        if self.config_type == "pytorch":
            data = self.data_converter.convert(data_inst, batch_size=self.batch_size)
            self.__build_pytorch_model(self.nn_define)
            epoch_degree = float(len(data)) * self.aggregate_every_n_epoch
        elif self.config_type == "cv":
            config_default = ObjDict(self.nn_define[0])
            config, model, loss, get_pbb = net.get_model()
            dataset_train = dataloader_detector.get_trainloader("train", config, config_default)
            optimizer = torch.optim.SGD(
                model.parameters(),
                config_default.lr,
                momentum=config_default.momentum,
                weight_decay=config_default.weight_decay)
            self.nn_model = PytorchNNModel(model=model,
                                           optimizer=optimizer,
                                           loss=loss)
            epoch_degree = float(len(dataset_train))*self.aggregate_every_n_epoch

        else:
            data = self.data_converter.convert(data_inst, batch_size=self.batch_size)
            self.__build_nn_model(data.get_shape()[0])
            epoch_degree = float(len(data)) * self.aggregate_every_n_epoch

        while self.aggregate_iteration_num < self.max_aggregate_iteration_num:
            Logger.info(f"start {self.aggregate_iteration_num}_th aggregation")
            #train
            if self.config_type == "cv":
                trainloader = DataLoader(dataset_train,
                                         batch_size=config_default.batch_size,
                                         shuffle=True,
                                         pin_memory=False)
                epoch_degree = float(len(trainloader))
                metrics = []
                self.nn_model._model.train()
                for i, (data, target, coord) in enumerate(trainloader):
                    # print('shape of data:', data.shape)
                    # print('shape of coord:', coord.shape)
                    # print('*******************', i, "/", str(len(trainloader)))
                    Logger.info(f"{i}:shape of data: {data.shape}, shape of coord:{coord.shape}")
                    output = self.nn_model._model(data, coord)
                    loss_output = loss(output, target)
                    optimizer.zero_grad()
                    loss_output[0].backward()
                    optimizer.step()
                    metrics.append(loss_output)
                    Logger.info(f"finish{i}th data")
                metrics = np.asarray(metrics, np.float32)
                #metrics
                acc = (np.sum(metrics[:, 6]) + np.sum(metrics[:, 8])) / (np.sum(metrics[:, 7]) + np.sum(metrics[:, 9]))
                tpr = np.sum(metrics[:, 6]) / np.sum(metrics[:, 7])
                tnr = np.sum(metrics[:, 8]) / np.sum(metrics[:, 9])
                tp = np.sum(metrics[:, 6])
                p = np.sum(metrics[:, 7])
                tn = np.sum(metrics[:, 8])
                n = np.sum(metrics[:, 9])
                total_loss = np.mean(metrics[:, 0])
                #设定需要用于聚合的loss用于后面判断
                data = total_loss
                classification_loss = np.mean(metrics[:, 1])
                bbox_regressiong_loss_1 = np.mean(metrics[:, 2])
                bbox_regressiong_loss_2 = np.mean(metrics[:, 3])
                bbox_regressiong_loss_3 = np.mean(metrics[:, 4])
                bbox_regressiong_loss_4 = np.mean(metrics[:, 5])
                Logger.info('EPOCH {}, acc: {:.2f}, tpr: {:.2f} ({}/{}), tnr: {:.1f} ({}/{}), total_loss: {:.3f}, classification loss: {:.3f}, bbox regression loss: {:.2f}, {:.2f}, {:.2f}, {:.2f}\
                        '.format(self.aggregate_iteration_num, acc, tpr, tp, p, tnr, tn, n,
                                 total_loss, classification_loss,
                                 bbox_regressiong_loss_1, bbox_regressiong_loss_2,
                                 bbox_regressiong_loss_3, bbox_regressiong_loss_4))
            else:
                self.nn_model.train(data, aggregate_every_n_epoch=self.aggregate_every_n_epoch)

            # send model for aggregate, then set aggregated model to local
            self.aggregator.send_weighted_model(weighted_model=self.nn_model.get_model_weights(),
                                                weight=epoch_degree * self.aggregate_every_n_epoch,
                                                suffix=self._suffix())

            weights = self.aggregator.get_aggregated_model(suffix=self._suffix())
            self.nn_model.set_model_weights(weights=weights)
            #calc loss and check convergence
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

    def predict(self, data_inst):
        if self.config_type == "cv":
            config_default = ObjDict(self.nn_define[0])
            Logger.info(f"{self.nn_define}")
            #这里只有的model不会用于预测，使用的模型为self.nn_model
            config, model, loss, get_pbb = net.get_model()
            dataset_validation = dataloader_detector.get_trainloader("validation", config, config_default)
            validateloader = DataLoader(dataset_validation,
                                        batch_size=config_default.batch_size,
                                        shuffle=False,
                                        pin_memory=False)
            Logger.info("validate begin.")
            #模型在这里
            self.nn_model._model.eval()
            metrics = []
            start_time = time.time()
            for i, (data, target, coord) in enumerate(validateloader):
                print('*******************', i, "/", str(len(validateloader)))
                # data = Variable(data.cuda())
                data = Variable(data)
                # target = Variable(target.cuda())
                target = Variable(target)
                # coord = Variable(coord.cuda())
                coord = Variable(coord)
                with torch.no_grad():
                    output = self.nn_model._model(data, coord)
                loss_output = loss(output, target, train=False)
                metrics.append(loss_output)
            end_time = time.time()
            metrics = np.asarray(metrics, np.float32)
            epoch = 1
            msg = 'EPOCH {} '.format(
                epoch) + 'Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
                      100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
                      100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
                      np.sum(metrics[:, 7]),
                      np.sum(metrics[:, 9]),
                      end_time - start_time) + 'loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
                      np.mean(metrics[:, 0]),
                      np.mean(metrics[:, 1]),
                      np.mean(metrics[:, 2]),
                      np.mean(metrics[:, 3]),
                      np.mean(metrics[:, 4]),
                      np.mean(metrics[:, 5]))
            print(msg)
            Logger.info(msg)
        else:
            data = self.data_converter.convert(data_inst, batch_size=self.batch_size)
            predict = self.nn_model.predict(data)
            num_output_units = data.get_shape()[1]
            threshold = self.param.predict_param.threshold

            if num_output_units[0] == 1:
                kv = [(x[0], (0 if x[1][0] <= threshold else 1, x[1][0].item())) for x in zip(data.get_keys(), predict)]
                pred_tbl = session.parallelize(kv, include_key=True)
                return data_inst.join(pred_tbl, lambda d, pred: [d.label, pred[0], pred[1], {"label": pred[0]}])
            else:
                kv = [(x[0], (x[1].argmax(), [float(e) for e in x[1]])) for x in zip(data.get_keys(), predict)]
                pred_tbl = session.parallelize(kv, include_key=True)
                return data_inst.join(pred_tbl,
                                      lambda d, pred: [d.label, pred[0].item(),
                                                       pred[1][pred[0]] / (sum(pred[1])),
                                                       {"raw_predict": pred[1]}])

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
