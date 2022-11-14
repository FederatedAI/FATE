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

from pipeline.component.component_base import FateComponent
from pipeline.component.nn.models.sequantial import Sequential
from pipeline.component.nn.backend.torch.interactive import InteractiveLayer
from pipeline.interface import Input
from pipeline.interface import Output
from pipeline.utils.tools import extract_explicit_parameter
from pipeline.component.nn.interface import DatasetParam


class HeteroNN(FateComponent):

    @extract_explicit_parameter
    def __init__(self, task_type="classification", epochs=None, batch_size=-1, early_stop="diff",
                 tol=1e-5, encrypt_param=None, predict_param=None, cv_param=None, interactive_layer_lr=0.1,
                 validation_freqs=None, early_stopping_rounds=None, use_first_metric_only=None,
                 floating_point_precision=23, selector_param=None, seed=100,
                 dataset: DatasetParam = DatasetParam(dataset_name='table'), **kwargs
                 ):

        explicit_parameters = kwargs["explict_parameters"]
        explicit_parameters["optimizer"] = None
        explicit_parameters["bottom_nn_define"] = None
        explicit_parameters["top_nn_define"] = None
        explicit_parameters["interactive_layer_define"] = None
        explicit_parameters["loss"] = None
        FateComponent.__init__(self, **explicit_parameters)

        if "name" in explicit_parameters:
            del explicit_parameters["name"]
        for param_key, param_value in explicit_parameters.items():
            setattr(self, param_key, param_value)

        self.input = Input(self.name, data_type="multi")
        self.output = Output(self.name, data_type='single')
        self._module_name = "HeteroNN"
        self.optimizer = None
        self.bottom_nn_define = None
        self.top_nn_define = None
        self.interactive_layer_define = None

        # model holder
        self._bottom_nn_model = Sequential()
        self._interactive_layer = Sequential()
        self._top_nn_model = Sequential()

        if hasattr(self, 'dataset'):
            assert isinstance(self.dataset, DatasetParam), 'dataset must be a DatasetParam class'
            self.dataset.check()
            self.dataset: DatasetParam = self.dataset.to_dict()

    def add_dataset(self, dataset_param: DatasetParam):

        assert isinstance(dataset_param, DatasetParam), 'dataset must be a DatasetParam class'
        dataset_param.check()
        self.dataset: DatasetParam = dataset_param.to_dict()
        self._component_parameter_keywords.add("dataset")
        self._component_param["dataset"] = self.dataset

    def add_bottom_model(self, model):
        if not hasattr(self, "_bottom_nn_model"):
            setattr(self, "_bottom_nn_model", Sequential())

        self._bottom_nn_model.add(model)

    def set_interactve_layer(self, layer):

        if not hasattr(self, "_interactive_layer"):
            setattr(self, "_interactive_layer", Sequential())
        assert isinstance(layer, InteractiveLayer), 'You need to add an interactive layer instance, \n' \
                                                    'you can access InteractiveLayer by:\n' \
                                                    't.nn.InteractiveLayer after fate_torch_hook(t)\n' \
                                                    'or from pipeline.component.nn.backend.torch.interactive ' \
                                                    'import InteractiveLayer'
        self._interactive_layer.add(layer)

    def add_top_model(self, model):
        if not hasattr(self, "_top_nn_model"):
            setattr(self, "_top_nn_model", Sequential())

        self._top_nn_model.add(model)

    def _set_optimizer(self, opt):
        assert hasattr(opt, 'to_dict'), 'opt does not have function to_dict(), remember to call fate_torch_hook(t)'
        self.optimizer = opt.to_dict()

    def _set_loss(self, loss):
        assert hasattr(loss, 'to_dict'), 'loss does not have function to_dict(), remember to call fate_torch_hook(t)'
        loss_conf = loss.to_dict()
        setattr(self, "loss", loss_conf)

    def compile(self, optimizer, loss):

        self._set_optimizer(optimizer)
        self._set_loss(loss)
        self._compile_common_network_config()
        self._compile_role_network_config()
        self._compile_interactive_layer()

    def _compile_interactive_layer(self):
        if hasattr(self, "_interactive_layer") and not self._interactive_layer.is_empty():
            self.interactive_layer_define = self._interactive_layer.get_network_config()
            self._component_param["interactive_layer_define"] = self.interactive_layer_define

    def _compile_common_network_config(self):
        if hasattr(self, "_bottom_nn_model") and not self._bottom_nn_model.is_empty():
            self.bottom_nn_define = self._bottom_nn_model.get_network_config()
            self._component_param["bottom_nn_define"] = self.bottom_nn_define

        if hasattr(self, "_top_nn_model") and not self._top_nn_model.is_empty():
            self.top_nn_define = self._top_nn_model.get_network_config()
            self._component_param["top_nn_define"] = self.top_nn_define

    def _compile_role_network_config(self):
        all_party_instance = self._get_all_party_instance()
        for role in all_party_instance:
            for party in all_party_instance[role]["party"].keys():
                all_party_instance[role]["party"][party]._compile_common_network_config()

    def get_bottom_model(self):

        if hasattr(self, "_bottom_nn_model") and not getattr(self, "_bottom_nn_model").is_empty():
            return getattr(self, "_bottom_nn_model").get_model()

        bottom_models = {}
        all_party_instance = self._get_all_party_instance()
        for role in all_party_instance.keys():
            for party in all_party_instance[role]["party"].keys():
                party_inst = all_party_instance[role]["party"][party]
                if party_inst is not None:
                    btn_model = all_party_instance[role]["party"][party].get_bottom_model()
                    if btn_model is not None:
                        bottom_models[party] = btn_model

        return bottom_models if len(bottom_models) > 0 else None

    def get_top_model(self):
        if hasattr(self, "_top_nn_model") and not getattr(self, "_top_nn_model").is_empty():
            return getattr(self, "_top_nn_model").get_model()

        models = {}
        all_party_instance = self._get_all_party_instance()
        for role in all_party_instance.keys():
            for party in all_party_instance[role]["party"].keys():
                party_inst = all_party_instance[role]["party"][party]
                if party_inst is not None:
                    top_model = all_party_instance[role]["party"][party].get_top_model()
                    if top_model is not None:
                        models[party] = top_model

        return models if len(models) > 0 else None

    def __getstate__(self):
        state = dict(self.__dict__)
        if "_bottom_nn_model" in state:
            del state["_bottom_nn_model"]

        if "_interactive_layer" in state:
            del state["_interactive_layer"]

        if "_top_nn_model" in state:
            del state["_top_nn_model"]

        return state
