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
#
import json
import math
import os
import tempfile
import types
import typing

import numpy
import pytorch_lightning as pl
import torch
import torch.optim
from fate_arch.computing import is_table
from fate_arch.computing.non_distributed import LocalData
from fate_arch.session import computing_session
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.nn.backend.pytorch.data import TableDataSet, VisionDataSet
from federatedml.nn.backend.pytorch.layer import get_layer_fn
from federatedml.nn.backend.pytorch.loss import get_loss_fn
from federatedml.nn.backend.pytorch.optimizer import get_optimizer
from federatedml.nn.homo_nn import _consts
from federatedml.param import HomoNNParam
from federatedml.protobuf.generated import nn_model_meta_pb2, nn_model_param_pb2
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter
from pytorch_lightning.callbacks import Callback
from torch import nn


class _PyTorchSAContext(object):
    def __init__(self, max_num_aggregation, name):
        self.max_num_aggregation = max_num_aggregation
        self._name = name
        self._aggregation_iteration = 0
        self._early_stopped = False

    def _suffix(self, group: str = "model"):
        return (
            self._name,
            group,
            f"{self._aggregation_iteration}",
        )

    def set_stopped(self):
        self._early_stopped = True

    def increase_aggregation_iteration(self):
        self._aggregation_iteration += 1

    @property
    def aggregation_iteration(self):
        return self._aggregation_iteration

    def finished(self):
        if (
            self._early_stopped
            or self._aggregation_iteration >= self.max_num_aggregation
        ):
            return True
        return False


class PyTorchSAClientContext(_PyTorchSAContext):
    def __init__(self, max_num_aggregation, aggregate_every_n_epoch, name="default"):
        super(PyTorchSAClientContext, self).__init__(
            max_num_aggregation=max_num_aggregation, name=name
        )
        self.transfer_variable = SecureAggregatorTransVar()
        self.aggregator = aggregator.Client(self.transfer_variable.aggregator_trans_var)
        self.random_padding_cipher = random_padding_cipher.Client(
            self.transfer_variable.random_padding_cipher_trans_var
        )
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self._params: list = None

        self._should_stop = False
        self.loss_summary = []

    def init(self):
        self.random_padding_cipher.create_cipher()

    def encrypt(self, tensor: torch.Tensor, weight):
        return self.random_padding_cipher.encrypt(
            torch.clone(tensor).detach().mul_(weight)
        ).numpy()

    def send_model(self, tensors, weight):
        self.aggregator.send_model(
            ([self.encrypt(p.data, weight) for p in tensors], weight),
            suffix=self._suffix(),
        )

    def recv_model(self):
        return [
            torch.from_numpy(arr)
            for arr in self.aggregator.get_aggregated_model(suffix=self._suffix())
        ]

    def send_loss(self, loss, weight):
        self.aggregator.send_model((loss, weight), suffix=self._suffix(group="loss"))

    def recv_loss(self):
        return self.aggregator.get_aggregated_model(
            suffix=self._suffix(group="convergence")
        )

    def do_aggregation(self, weight):
        """
        do secure aggregation step

        Args:
            weight: Aggregate weight.

        Returns:

        """
        self.send_model(self._params, weight)
        agg_tensors: typing.List[torch.Tensor] = self.recv_model()
        for param, agg_tensor in zip(self._params, agg_tensors):
            if param.grad is None:
                continue
            param.data.copy_(agg_tensor)

    def do_convergence_check(self, weight, loss):
        loss_value = loss.detach().numpy().tolist()
        self.loss_summary.append(loss_value)

        # send loss to server
        self.send_loss(loss_value, weight)

        # recv convergence status
        status = self.recv_loss()
        return status

    def configure_aggregation_params(
        self,
        optimizer,
    ):
        if optimizer is not None:
            self._params = [
                param
                for param_group in optimizer.param_groups
                for param in param_group["params"]
            ]
            return
        raise TypeError(f"params and optimizer can't be both none")

    def should_aggregate_on_epoch(self, epoch_index):
        return (epoch_index + 1) % self.aggregate_every_n_epoch == 0

    def should_stop(self):
        return self._should_stop

    def set_converged(self):
        self._should_stop = True


class PyTorchSAServerContext(_PyTorchSAContext):
    def __init__(self, max_num_aggregation, eps=0.0, name="default"):
        super(PyTorchSAServerContext, self).__init__(
            max_num_aggregation=max_num_aggregation, name=name
        )
        self.transfer_variable = SecureAggregatorTransVar()
        self.aggregator = aggregator.Server(self.transfer_variable.aggregator_trans_var)
        self.random_padding_cipher = random_padding_cipher.Server(
            self.transfer_variable.random_padding_cipher_trans_var
        )

        self._eps = eps
        self._loss = math.inf

    def init(self, init_aggregation_iteration=0):
        self.random_padding_cipher.exchange_secret_keys()
        self._aggregation_iteration = init_aggregation_iteration

    def send_model(self, aggregated_tensors):
        return self.aggregator.send_aggregated_model(
            aggregated_tensors, suffix=self._suffix()
        )

    def recv_model(self):
        return self.aggregator.get_models(suffix=self._suffix())

    def send_convergence_status(self, status):
        self.aggregator.send_aggregated_model(
            status, suffix=self._suffix(group="convergence")
        )

    def recv_losses(self):
        return self.aggregator.get_models(suffix=self._suffix(group="loss"))

    def do_convergence_check(self):
        # recieve losses and weights of parties
        loss_weight_pairs = self.recv_losses()
        total_loss = 0.0
        total_weight = 0.0

        for loss, weight in loss_weight_pairs:
            total_loss += loss * weight
            total_weight += weight
        mean_loss = total_loss / total_weight

        is_converged = abs(mean_loss - self._loss) < self._eps
        # send convergen status
        self.send_convergence_status(is_converged)

        self._loss = mean_loss

        LOGGER.info(f"convergence check: loss={mean_loss}, is_converged={is_converged}")
        return is_converged, mean_loss


class EarlyStopCallback(Callback):
    def __init__(self, context: PyTorchSAClientContext):
        self.context = context

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.context.should_stop():
            trainer.should_stop = True


class FedLightModule(pl.LightningModule):
    def __init__(
        self,
        context: PyTorchSAClientContext,
        layers_config: typing.List[typing.Mapping],
        optimizer_config: types.SimpleNamespace,
        loss_config: typing.Mapping,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.context = context

        # model
        layers = []
        for layer_config in layers_config:
            layer_name = layer_config["layer"]
            layer_kwargs = {k: v for k, v in layer_config.items() if k != "layer"}
            layers.append(get_layer_fn(layer_name, layer_kwargs))
        self.model = nn.Sequential(*layers)

        # loss
        loss_name = loss_config["loss"]
        loss_kwargs = {k: v for k, v in loss_config.items() if k != "loss"}
        self.loss_fn, self.expected_label_type = get_loss_fn(loss_name, loss_kwargs)

        # optimizer
        self._optimizer_name = optimizer_config.optimizer
        self._optimizer_kwargs = optimizer_config.kwargs

        self._num_data_consumed = 0
        self._all_consumed_data_aggregated = True

        self._should_early_stop = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        if y_hat.shape[1] > 1:
            accuracy = (y_hat.argmax(dim=1) == y).sum().float() / float(y.size(0))
        else:
            y_prob = y_hat[:, 0] > 0.5
            accuracy = (y == y_prob).sum().float() / y.size(0)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([x["val_loss"] for x in outputs]))
        accuracy = torch.mean(torch.stack([x["val_accuracy"] for x in outputs]))
        convergence_status = self.context.do_convergence_check(
            self._num_data_consumed, loss
        )
        LOGGER.info(
            f"validation epoch end, local loss: {loss}, local accuracy: {accuracy}, convergence statu: {convergence_status}"
        )

        # aggregation end
        self._num_data_consumed = 0
        self.context.increase_aggregation_iteration()
        if convergence_status:
            self.context.set_converged()

    def training_epoch_end(self, outputs) -> None:
        ...

    def on_train_epoch_start(self) -> None:
        if self._all_consumed_data_aggregated:
            self._num_data_consumed = 0
            self._all_consumed_data_aggregated = False

    def on_train_batch_start(
        self, batch: typing.Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if self._all_consumed_data_aggregated:
            self._all_consumed_data_aggregated = False
            self._num_data_consumed = len(batch)
        else:
            self._num_data_consumed += len(batch)

    def on_train_epoch_end(self, outputs) -> None:
        if self.context.should_aggregate_on_epoch(self.current_epoch):
            self.context.do_aggregation(float(self._num_data_consumed))
            self._all_consumed_data_aggregated = True
            # self._num_data_consumed = 0

    def configure_optimizers(self):
        optimizer = get_optimizer(
            parameters=self.parameters(),
            optimizer_name=self._optimizer_name,
            optimizer_kwargs=self._optimizer_kwargs,
        )
        self.context.configure_aggregation_params(optimizer=optimizer)
        return optimizer


class PyTorchFederatedTrainer(object):
    def __init__(
        self,
        pl_trainer: pl.Trainer = None,
        header=None,
        label_mapping=None,
        pl_model: FedLightModule = None,
        context: PyTorchSAClientContext = None,
    ):
        self.pl_trainer = pl_trainer
        self.pl_model = pl_model
        self.context = context
        self.header = header
        self.label_mapping = label_mapping

    def get_label_mapping(self):
        return self.label_mapping

    def fit(self, dataloader):
        self.pl_trainer.fit(
            self.pl_model,
            train_dataloader=dataloader,
            val_dataloaders=dataloader,
        )

    def summary(self):
        return {
            "loss": self.context.loss_summary,
            "is_converged": self.context.should_stop(),
        }

    def predict(self, dataset, batch_size):
        if batch_size < 0:
            batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=1
        )
        results = []
        for x, y in dataloader:
            results.append(self.pl_model(x).detach().numpy())

        predict = numpy.vstack(results)
        num_output_units = predict.shape[1]
        if num_output_units == 1:
            kv = zip(dataset.get_keys(), (x.tolist()[0] for x in predict))
        else:
            kv = zip(dataset.get_keys(), predict.tolist())

        partitions = getattr(dataset, "partitions", 10)

        pred_tbl = computing_session.parallelize(
            kv, include_key=True, partition=partitions
        )
        classes = (
            [0, 1] if num_output_units == 1 else [i for i in range(num_output_units)]
        )

        return pred_tbl, classes

    def export_model(self, param):

        param_pb = nn_model_param_pb2.NNModelParam()

        # save api_version
        param_pb.api_version = param.api_version

        # save pl model bytes
        with tempfile.TemporaryDirectory() as d:
            filepath = os.path.join(d, "model.ckpt")
            self.pl_trainer.save_checkpoint(filepath)
            with open(filepath, "rb") as f:
                param_pb.saved_model_bytes = f.read()

        # save header
        param_pb.header.extend(self.header)

        # save label mapping
        if self.label_mapping is not None:
            for label, mapped in self.label_mapping.items():
                param_pb.label_mapping.add(
                    label=json.dumps(label), mapped=json.dumps(mapped)
                )

        # meta
        meta_pb = nn_model_meta_pb2.NNModelMeta()
        meta_pb.params.CopyFrom(param.generate_pb())
        meta_pb.aggregate_iter = self.context.aggregation_iteration

        return {_consts.MODEL_META_NAME: meta_pb, _consts.MODEL_PARAM_NAME: param_pb}

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):

        # restore pl model
        with tempfile.TemporaryDirectory() as d:
            filepath = os.path.join(d, "model.ckpt")
            with open(filepath, "wb") as f:
                f.write(model_obj.saved_model_bytes)
            pl_model = FedLightModule.load_from_checkpoint(filepath)

        # restore context
        context = pl_model.context

        # restore pl trainer
        total_epoch = context.max_num_aggregation * context.aggregate_every_n_epoch
        pl_trainer = pl.Trainer(
            max_epochs=total_epoch,
            min_epochs=total_epoch,
            callbacks=[EarlyStopCallback(context)],
            num_sanity_val_steps=0,
        )
        pl_trainer.model = pl_model

        # restore data header
        header = list(model_obj.header)

        # restore label mapping
        label_mapping = {}
        for item in model_obj.label_mapping:
            label = json.loads(item.label)
            mapped = json.loads(item.mapped)
            label_mapping[label] = mapped
        if not label_mapping:
            label_mapping = None

        # restore trainer
        trainer = PyTorchFederatedTrainer(
            pl_trainer=pl_trainer,
            header=header,
            label_mapping=label_mapping,
            pl_model=pl_model,
            context=context,
        )

        # restore model param
        param.restore_from_pb(meta_obj.params)
        return trainer

    def save_checkpoint(self, filepath="./model.ckpt"):
        self.pl_trainer.save_checkpoint(filepath)


def make_dataset(data, **kwargs):
    if is_table(data):
        dataset = TableDataSet(data_instances=data, **kwargs)
    elif isinstance(data, LocalData):
        dataset = VisionDataSet(data.path, **kwargs)
    else:
        raise TypeError(f"data type {data} not supported")

    return dataset


def make_predict_dataset(data, trainer: PyTorchFederatedTrainer):
    return make_dataset(
        data,
        is_train=False,
        label_align_mapping=trainer.get_label_mapping(),
        expected_label_type=trainer.pl_model.expected_label_type,
    )


def build_trainer(param: HomoNNParam, data, should_label_align=True, trainer=None):
    header = data.schema["header"]
    if trainer is None:
        total_epoch = param.aggregate_every_n_epoch * param.max_iter
        context = PyTorchSAClientContext(
            max_num_aggregation=param.max_iter,
            aggregate_every_n_epoch=param.aggregate_every_n_epoch,
        )
        pl_trainer = pl.Trainer(
            max_epochs=total_epoch,
            min_epochs=total_epoch,
            callbacks=[EarlyStopCallback(context)],
            num_sanity_val_steps=0,
        )
        context.init()
        pl_model = FedLightModule(
            context,
            layers_config=param.nn_define,
            optimizer_config=param.optimizer,
            loss_config={"loss": param.loss},
        )
        expected_label_type = pl_model.expected_label_type
        dataset = make_dataset(
            data=data,
            is_train=should_label_align,
            expected_label_type=expected_label_type,
        )

        batch_size = param.batch_size
        if batch_size < 0:
            batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=1
        )
        trainer = PyTorchFederatedTrainer(
            pl_trainer=pl_trainer,
            header=header,
            label_mapping=dataset.get_label_align_mapping(),
            pl_model=pl_model,
            context=context,
        )
    else:
        trainer.context.init()
        expected_label_type = trainer.pl_model.expected_label_type

        dataset = make_dataset(
            data=data,
            is_train=should_label_align,
            expected_label_type=expected_label_type,
        )

        batch_size = param.batch_size
        if batch_size < 0:
            batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=1
        )
    return trainer, dataloader


class PytorchFederatedAggregator(object):
    def __init__(self, context: PyTorchSAServerContext):
        self.context = context

    def fit(self, loss_callback):
        while not self.context.finished():
            recv_elements: typing.List[
                typing.Tuple[typing.List[numpy.ndarray], float]
            ] = self.context.recv_model()
            degrees = [party_tuple[1] for party_tuple in recv_elements]
            tensors = [party_tuple[0] for party_tuple in recv_elements]
            total_degree = sum(degrees)
            for i in range(len(tensors)):
                for j, tensor in enumerate(tensors[i]):
                    tensor /= total_degree
                    if i != 0:
                        tensors[0][j] += tensor

            self.context.send_model(tensors[0])
            is_converged, loss = self.context.do_convergence_check()
            loss_callback(self.context.aggregation_iteration, float(loss))

            # increse aggregation iteration number at iteration end
            self.context.increase_aggregation_iteration()
            if is_converged:
                break

    def export_model(self, param):

        param_pb = nn_model_param_pb2.NNModelParam()

        # save api_version
        param_pb.api_version = param.api_version

        meta_pb = nn_model_meta_pb2.NNModelMeta()
        meta_pb.params.CopyFrom(param.generate_pb())
        meta_pb.aggregate_iter = self.context.aggregation_iteration

        return {_consts.MODEL_META_NAME: meta_pb, _consts.MODEL_PARAM_NAME: param_pb}

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        param.restore_from_pb(meta_obj.params)

    @staticmethod
    def dataset_align():
        LOGGER.info("start label alignment")
        label_mapping = HomoLabelEncoderArbiter().label_alignment()
        LOGGER.info(f"label aligned, mapping: {label_mapping}")


def build_aggregator(param: HomoNNParam, init_iteration=0):
    context = PyTorchSAServerContext(
        max_num_aggregation=param.max_iter, eps=param.early_stop.eps
    )
    context.init(init_aggregation_iteration=init_iteration)
    fed_aggregator = PytorchFederatedAggregator(context)
    return fed_aggregator
