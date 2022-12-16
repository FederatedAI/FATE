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
import copy
import torch as t
from torch.optim import Adam
from pipeline.component.component_base import FateComponent
from pipeline.component.nn.backend.torch.base import Sequential
from pipeline.component.nn.backend.torch import base
from pipeline.interface import Input
from pipeline.interface import Output
from pipeline.utils.tools import extract_explicit_parameter
from pipeline.component.nn.interface import TrainerParam, DatasetParam
from pipeline.component.nn.backend.torch.cust import CustModel
from pipeline.utils.logger import LOGGER

# default parameter dict
DEFAULT_PARAM_DICT = {
    'trainer': TrainerParam(trainer_name='fedavg_trainer'),
    'dataset': DatasetParam(dataset_name='table'),
    'torch_seed': 100,
    'loss': None,
    'optimizer': None,
    'nn_define': None
}


class HomoNN(FateComponent):
    """

    Parameters
    ----------
    name, name of this component
    trainer, trainer param
    dataset, dataset param
    torch_seed, global random seed
    loss, loss function from fate_torch
    optimizer, optimizer from fate_torch
    model, a fate torch sequential defining the model structure
    """

    @extract_explicit_parameter
    def __init__(self,
                 name=None,
                 trainer: TrainerParam = TrainerParam(trainer_name='fedavg_trainer', epochs=10, batch_size=512,  # training parameter
                                                      early_stop=None, tol=0.0001,  # early stop parameters
                                                      secure_aggregate=True, weighted_aggregation=True,
                                                      aggregate_every_n_epoch=None,  # federation
                                                      cuda=False, pin_memory=True, shuffle=True, data_loader_worker=0,  # GPU dataloader
                                                      validation_freqs=None),
                 dataset: DatasetParam = DatasetParam(dataset_name='table'),
                 torch_seed: int = 100,
                 loss=None,
                 optimizer: t.optim.Optimizer = None,
                 model: Sequential = None, **kwargs):

        explicit_parameters = copy.deepcopy(DEFAULT_PARAM_DICT)
        if 'name' not in kwargs["explict_parameters"]:
            raise RuntimeError('moduel name is not set')
        explicit_parameters["name"] = kwargs["explict_parameters"]['name']
        FateComponent.__init__(self, **explicit_parameters)
        kwargs["explict_parameters"].pop('name')

        self.input = Input(self.name, data_type="multi")
        self.output = Output(self.name, data_type='single')
        self._module_name = "HomoNN"
        self._updated = {'trainer': False, 'dataset': False,
                         'torch_seed': False, 'loss': False, 'optimizer': False, 'model': False}
        self._set_param(kwargs["explict_parameters"])
        self._check_parameters()

    def _set_updated(self, attr, status=True):

        if attr in self._updated:
            self._updated[attr] = status
        else:
            raise ValueError('attr {} not in update status {}'.format(attr, self._updated))

    def _set_param(self, params):
        if "name" in params:
            del params["name"]
        for param_key, param_value in params.items():
            setattr(self, param_key, param_value)

    def _check_parameters(self):

        if hasattr(self, 'trainer') and self.trainer is not None and not self._updated['trainer']:
            assert isinstance(
                self.trainer, TrainerParam), 'trainer must be a TrainerPram class'
            self.trainer.check()
            self.trainer: TrainerParam = self.trainer.to_dict()
            self._set_updated('trainer', True)

        if hasattr(self, 'dataset') and self.dataset is not None and not self._updated['dataset']:
            assert isinstance(
                self.dataset, DatasetParam), 'dataset must be a DatasetParam class'
            self.dataset.check()
            self.dataset: DatasetParam = self.dataset.to_dict()
            self._set_updated('dataset', True)

        if hasattr(self, 'model') and self.model is not None and not self._updated['model']:
            if isinstance(self.model, Sequential):
                self.nn_define = self.model.get_network_config()
            elif isinstance(self.model, CustModel):
                self.model = Sequential(self.model)
                self.nn_define = self.model.get_network_config()
            else:
                raise RuntimeError('Model must be a fate-torch Sequential, but got {} '
                                   '\n do remember to call fate_torch_hook():'
                                   '\n    import torch as t'
                                   '\n    fate_torch_hook(t)'.format(
                                       type(self.model)))
            self._set_updated('model', True)

        if hasattr(self, 'optimizer') and self.optimizer is not None and not self._updated['optimizer']:
            if not isinstance(self.optimizer, base.FateTorchOptimizer):
                raise ValueError('please pass FateTorchOptimizer instances to Homo-nn components, got {}.'
                                 'do remember to use fate_torch_hook():\n'
                                 '    import torch as t\n'
                                 '    fate_torch_hook(t)'.format(type(self.optimizer)))
            optimizer_config = self.optimizer.to_dict()
            self.optimizer = optimizer_config
            self._set_updated('optimizer', True)

        if hasattr(self, 'loss') and self.loss is not None and not self._updated['loss']:
            if isinstance(self.loss, base.FateTorchLoss):
                loss_config = self.loss.to_dict()
            elif issubclass(self.loss, base.FateTorchLoss):
                loss_config = self.loss().to_dict()
            else:
                raise ValueError('unable to parse loss function {}, loss must be an instance'
                                 'of FateTorchLoss subclass or a subclass of FateTorchLoss, '
                                 'do remember to use fate_torch_hook()'.format(self.loss))
            self.loss = loss_config
            self._set_updated('loss', True)

    def component_param(self, **kwargs):

        # reset paramerters
        used_attr = set()
        setattr(self, 'model', None)
        if 'model' in kwargs:
            self.model = kwargs['model']
            kwargs.pop('model')
            self._set_updated('model', False)

        for attr in self._component_parameter_keywords:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
                self._set_updated(attr, False)
                used_attr.add(attr)
        self._check_parameters()  # check and convert homo-nn paramters
        not_use_attr = set(kwargs.keys()).difference(used_attr)
        for attr in not_use_attr:
            LOGGER.warning(f"key {attr}, value {kwargs[attr]} not use")
        self._role_parameter_keywords |= used_attr
        for attr in self.__dict__:
            if attr not in self._component_parameter_keywords:
                continue
            else:
                self._component_param[attr] = getattr(self, attr)

    def __getstate__(self):
        state = dict(self.__dict__)
        if "model" in state:
            del state["model"]
        return state
