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
import torch as t
import sys
from pipeline.param.base_param import BaseParam
from pipeline.component.component_base import FateComponent
from pipeline.component.nn.backend.torch.base import Sequential
from pipeline.component.nn.backend.torch.cust_model import CustModel
from pipeline.component.nn.backend.torch import base
from pipeline.interface import Input
from pipeline.interface import Output
from pipeline.utils.tools import extract_explicit_parameter

try:
    import IPython as ipy
    from IPython.core.magic import register_cell_magic
except ImportError as e:
    print('IPython is not installed, to use save_to_fate function, you need to install IPython')
    register_cell_magic = None


if register_cell_magic is not None:
    if ipy.get_ipython():
        @register_cell_magic
        def save_to_fate(line, cell):

            # search for federatedml path
            base_path = None
            for p in sys.path:
                if p.endswith('/fate/python'):
                    base_path = p
                    break

            if base_path is None:
                raise ValueError('cannot find fate/python in system path, please check your configuration')

            base_path = base_path + '/federatedml/'

            model_pth = 'nn/model_zoo/'
            dataset_pth = 'nn/dataset/'
            trainer_pth = 'nn/homo/trainer/'
            aggregator_pth = 'framework/homo/aggregator/'

            mode_map = {
                'model': model_pth,
                'trainer': trainer_pth,
                'aggregator': aggregator_pth,
                'dataset': dataset_pth
            }

            args = line.split()
            assert len(args) == 2, "input args len is not 2, got {} \n expect format: %%save_to_fate SAVE_MODE FILENAME \n SAVE_MODE in ['model', 'dataset', 'trainer', 'aggregator']   FILE_NAME xxx.py".format(args)
            modes_avail = ['model', 'dataset', 'trainer', 'aggregator']
            save_mode = args[0]
            file_name = args[1]
            assert save_mode in modes_avail, 'avail modes are {}, got {}'.format(modes_avail, save_mode)
            assert file_name.endswith('.py'), 'save file should be a .py'
            with open(base_path+mode_map[save_mode]+file_name, 'w') as f:
                f.write(cell)
            ipy.get_ipython().run_cell(cell)
    else:
        save_to_fate = None
else:
    save_to_fate = None

    
class TrainerParam(BaseParam):

    def __init__(self, trainer_name=None, **kwargs):
        super(TrainerParam, self).__init__()
        self.trainer_name = trainer_name
        self.param = kwargs

    def check(self):
        if self.trainer_name is None:
            raise ValueError('You did not specify the trainer name, please set the trainer name')
        self.check_string(self.trainer_name, 'trainer_name')

    def to_dict(self):
        ret = {'trainer_name': self.trainer_name, 'param': self.param}
        return ret


class DatasetParam(BaseParam):

    def __init__(self, dataset_name=None, **kwargs):
        super(DatasetParam, self).__init__()
        self.dataset_name = dataset_name
        self.param = kwargs

    def check(self):
        self.check_string(self.dataset_name, 'dataset_name')

    def to_dict(self):
        ret = {'dataset_name': self.dataset_name, 'param': self.param}
        return ret


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
                 trainer: TrainerParam = TrainerParam(epochs=10, batch_size=512,  # training parameter
                                                      early_stop=None, eps=0.0001,  # early stop parameters
                                                      secure_aggregate=True, weighted_aggregation=True,
                                                      aggregate_every_n_epoch=None,  # federation
                                                      cuda=False, pin_memory=True, shuffle=True, data_loader_worker=0,  # GPU dataloader
                                                      validation_freq=None),
                 dataset: DatasetParam = DatasetParam(dataset_name='table'),
                 torch_seed: int = 100,
                 loss=None,
                 optimizer: t.optim.Optimizer = None,
                 model: Sequential = None
                 , **kwargs):

        explicit_parameters = kwargs["explict_parameters"]
        explicit_parameters["nn_define"] = None
        FateComponent.__init__(self, **explicit_parameters)

        if "name" in explicit_parameters:
            del explicit_parameters["name"]

        for param_key, param_value in explicit_parameters.items():
            setattr(self, param_key, param_value)

        self.input = Input(self.name, data_type="multi")
        self.output = Output(self.name, data_type='single')
        self._module_name = "HomoNN"
        self.nn_define = None

        if hasattr(self, 'trainer'):
            assert isinstance(self.trainer, TrainerParam), 'trainer must be a TrainerPram class'
            self.trainer.check()
            self.trainer: TrainerParam = self.trainer.to_dict()

        if hasattr(self, 'dataset'):
            assert isinstance(self.dataset, DatasetParam), 'dataset must be a DatasetParam class'
            self.dataset.check()
            self.dataset: DatasetParam = self.dataset.to_dict()

        if hasattr(self, 'model'):
            assert isinstance(self.model, Sequential), 'Model must be a fate-torch Sequential, but got {} ' \
                                                       '\n do remember to call fate_torch_hook():' \
                                                       '\n    import torch as t' \
                                                       '\n    fate_torch_hook(t)'.format(type(self.model))
            self.nn_define = self.model.get_network_config()
            del self.model

        if hasattr(self, 'optimizer'):
            if self.optimizer is not None:
                if not isinstance(self.optimizer, base.FateTorchOptimizer):
                    raise ValueError('please pass FateTorchOptimizer instances to Homo-nn components, got {}.'
                                     'do remember to use fate_torch_hook():\n'
                                     '    import torch as t\n'
                                     '    fate_torch_hook(t)')
                optimizer_config = self.optimizer.to_dict()
                self.optimizer = optimizer_config

        if hasattr(self, 'loss'):
            if self.loss is not None:
                if isinstance(self.loss, base.FateTorchLoss):
                    loss_config = self.loss.to_dict()
                elif issubclass(self.loss, base.FateTorchLoss):
                    loss_config = self.loss().to_dict()
                else:
                    raise ValueError('unable to parse loss function {}, loss must be an instance'
                                     'of FateTorchLoss subclass or a subclass of FateTorchLoss, '
                                     'do remember to use fate_torch_hook()'.format(self.loss))
                self.loss = loss_config

    def __getstate__(self):
        state = dict(self.__dict__)
        if "model" in state:
            del state["model"]
        return state
