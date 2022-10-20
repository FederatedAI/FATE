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
from pipeline.param.base_param import BaseParam
from pipeline.component.component_base import FateComponent
from pipeline.component.nn.backend.torch.base import Sequential, FateTorchOptimizer, FateTorchLoss
from pipeline.component.nn.backend.torch import nn, base
from pipeline.interface import Input
from pipeline.interface import Output
from pipeline.utils.tools import extract_explicit_parameter
try:
    from IPython.core.magic import register_cell_magic
except ImportError as e:
    print('IPython is not installed, to use save_to_fate function, you need to install IPython')
    register_cell_magic = None

base_path = '/data/projects/cwj/standalone_fate_install_1.9.0_release/fate/python/federatedml/'
model_pth = 'nn_/model_zoo/'
dataset_pth = 'nn_/dataset'
trainer_pth = 'nn_/homo/trainer/'
aggregator_pth = 'framework/homo/aggregator/'

mode_map = {
    'model': model_pth,
    'trainer': trainer_pth,
    'aggregator': aggregator_pth,
    'dataset': dataset_pth
}

if register_cell_magic is not None:
    @register_cell_magic
    def save_to_fate(line, cell):
        args = line.split()
        assert len(args) == 2, "input args len is not 2, got {} \n expect format: %%save_to_fate SAVE_MODE FILENAME \n SAVE_MODE in ['model', 'dataset', 'trainer', 'aggregator']   FILE_NAME xxx.py".format(args)
        modes_avail = ['model', 'dataset', 'trainer', 'aggregator']
        save_mode = args[0]
        file_name = args[1]
        assert save_mode in modes_avail, 'avail modes are {}, got {}'.format(modes_avail, save_mode)
        assert file_name.endswith('.py'), 'save file should be a .py'
        with open(base_path+mode_map[save_mode]+file_name, 'w') as f:
            f.write(cell)
        get_ipython().run_cell(cell)
else:
    save_to_fate = None

    
class TrainerParam(BaseParam):

    def __init__(self, trainer_name='', **kwargs):
        super(TrainerParam, self).__init__()
        self.trainer_name = trainer_name
        self.param = kwargs

    def check(self):
        self.check_string(self.trainer_name, 'trainer_name')

    def to_dict(self):
        ret = {'trainer_name': self.trainer_name, 'param': self.param}
        return ret


class DatasetParam(BaseParam):

    def __init__(self, dataset_name='', **kwargs):
        super(DatasetParam, self).__init__()
        self.dataset_name = dataset_name
        self.param = kwargs

    def check(self):
        self.check_string(self.dataset_name, 'dataset_name')

    def to_dict(self):
        ret = {'dataset_name': self.dataset_name, 'param': self.param}
        return ret


class HomoCustNN(FateComponent):

    @extract_explicit_parameter
    def __init__(self,
                 name=None,
                 trainer: TrainerParam = TrainerParam(trainer_name='fedavg_trainer', epochs=100,
                                                      batch_size=-1, early_stop=None, tol=0.0001,
                                                      secure_aggregate=True,
                                                      aggregate_every_n_epochs=None),
                 dataset: DatasetParam = DatasetParam(dataset_name='table'),
                 torch_seed: int = 100,
                 validation_freq: int = None,
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
        self._module_name = "HomoCustNN"
        self.nn_define = None

        if hasattr(self, 'trainer'):
            self.trainer: TrainerParam = self.trainer.to_dict()

        if hasattr(self, 'dataset'):
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
