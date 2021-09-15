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
from fate_arch.computing import is_table
from federatedml.model_base import Metric, MetricMeta
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.has_converged import HasConvergedTransVar
from federatedml.framework.homo.blocks.loss_scatter import LossScatterTransVar
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.model_base import ModelBase
from federatedml.nn.homo_nn._consts import _extract_meta, _extract_param
from federatedml.param.homo_nn_param import HomoNNParam
from federatedml.util import component_properties, consts
from federatedml.util import LOGGER


class HomoNNBase(ModelBase):
    def __init__(self, trans_var):
        super().__init__()
        self.model_param = HomoNNParam()
        self.transfer_variable = trans_var
        self._api_version = 0

    def _init_model(self, param):
        self.param = param
        self.set_version(param.api_version)

    def is_version_0(self):
        return self._api_version == 0

    def set_version(self, version):
        self._api_version = version


class HomoNNServer(HomoNNBase):
    def __init__(self, trans_var):
        super().__init__(trans_var=trans_var)
        self._init_iteration = 0

    def _init_model(self, param: HomoNNParam):
        super()._init_model(param)
        if self.is_version_0():
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
        self.callback_list.on_train_begin(data_inst, None)
        if self.is_version_0():
            from federatedml.nn.homo_nn import _version_0

            _version_0.server_fit(self=self, data_inst=data_inst)

        else:
            from federatedml.nn.homo_nn._torch import build_aggregator

            self.aggregator = build_aggregator(
                self.param, init_iteration=self._init_iteration
            )

            if not self.component_properties.is_warm_start:
                self.aggregator.dataset_align()
            self.aggregator.fit(self.callback_loss)

        self.callback_list.on_train_end()

    def export_model(self):
        if self.is_version_0():
            from federatedml.nn.homo_nn import _version_0

            return _version_0.arbiter_export_model(self=self)

        else:
            return self.aggregator.export_model(param=self.param)

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        model_obj = _extract_param(model_dict)
        meta_obj = _extract_meta(model_dict)

        # compatibility
        if not hasattr(model_obj, "api_version"):
            self.set_version(0)
        else:
            self.set_version(model_obj.api_version)

        if self.is_version_0():
            from federatedml.nn.homo_nn import _version_0

            _version_0.arbiter_load_model(
                self=self,
                meta_obj=meta_obj,
                model_obj=model_obj,
                is_warm_start_mode=self.component_properties.is_warm_start,
            )
        else:
            self._init_iteration = meta_obj.aggregate_iter
            self.param.restore_from_pb(
                meta_obj.params,
                is_warm_start_mode=self.component_properties.is_warm_start,
            )


class HomoNNClient(HomoNNBase):
    def __init__(self, trans_var):
        super().__init__(trans_var=trans_var)
        self._trainer = ...

    def _init_model(self, param: HomoNNParam):
        super()._init_model(param)
        if self.is_version_0():
            from federatedml.nn.homo_nn import _version_0

            _version_0.client_init_model(self, param)

    def fit(self, data, *args):
        self.callback_list.on_train_begin(data, None)
        if self.is_version_0():
            from federatedml.nn.homo_nn import _version_0

            _version_0.client_fit(self=self, data_inst=data)
        else:
            from federatedml.nn.homo_nn._torch import build_trainer

            if not self.component_properties.is_warm_start:
                self._trainer = None
            self._trainer, dataloader = build_trainer(
                param=self.param,
                data=data,
                should_label_align=not self.component_properties.is_warm_start,
                trainer=self._trainer,
            )
            self._trainer.fit(dataloader)
            self.set_summary(self._trainer.summary())
            # save model to local filesystem
            self._trainer.save_checkpoint()
        self.callback_list.on_train_end()

    def predict(self, data):

        if self.is_version_0():
            from federatedml.nn.homo_nn import _version_0

            results = _version_0.client_predict(self=self, data_inst=data)
            return results

        else:
            from federatedml.nn.homo_nn._torch import make_predict_dataset

            dataset = make_predict_dataset(data=data, trainer=self._trainer)
            predict_tbl, classes = self._trainer.predict(
                dataset=dataset,
                batch_size=self.param.batch_size,
            )
            data_instances = data if is_table(data) else dataset.as_data_instance()
            results = self.predict_score_to_output(
                data_instances,
                predict_tbl,
                classes=classes,
                threshold=self.param.predict_param.threshold,
            )
            return results

    def export_model(self):
        if self.is_version_0():
            from federatedml.nn.homo_nn import _version_0

            return _version_0.client_export_model(self=self)

        else:
            return self._trainer.export_model(self.param)

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        model_obj = _extract_param(model_dict)
        meta_obj = _extract_meta(model_dict)

        # compatibility
        if not hasattr(model_obj, "api_version"):
            self.set_version(0)
        else:
            self.set_version(model_obj.api_version)

        if self.is_version_0():
            from federatedml.nn.homo_nn import _version_0

            _version_0.client_load_model(
                self=self,
                meta_obj=meta_obj,
                model_obj=model_obj,
                is_warm_start_mode=self.component_properties.is_warm_start,
            )
        else:
            from federatedml.nn.homo_nn._torch import PyTorchFederatedTrainer

            self._trainer = PyTorchFederatedTrainer.load_model(
                model_obj=model_obj, meta_obj=meta_obj, param=self.param
            )


# server: Arbiter, clients: Guest and Hosts
class HomoNNDefaultTransVar(HomoTransferBase):
    def __init__(
        self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None
    ):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.secure_aggregator_trans_var = SecureAggregatorTransVar(
            server=server, clients=clients, prefix=self.prefix
        )
        self.loss_scatter_trans_var = LossScatterTransVar(
            server=server, clients=clients, prefix=self.prefix
        )
        self.has_converged_trans_var = HasConvergedTransVar(
            server=server, clients=clients, prefix=self.prefix
        )


class HomoNNDefaultClient(HomoNNClient):
    def __init__(self):
        super().__init__(trans_var=HomoNNDefaultTransVar())


class HomoNNDefaultServer(HomoNNServer):
    def __init__(self):
        super().__init__(trans_var=HomoNNDefaultTransVar())
