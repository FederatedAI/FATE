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
from torch import nn
from transformers import AutoModel
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_dropout_emulator_and_adapters(
        transformer_layers: nn.ModuleList,
        emulator_layer_num: int,
        adapter_top_layer_num: int,
        adapter_bottom_layer_num: int):

    assert adapter_bottom_layer_num > 0 and adapter_top_layer_num > 0, "adapter layer num must be greater than 0"
    assert emulator_layer_num < len(
        transformer_layers), "emulator layer num must be less than the number of transformer layers"
    assert adapter_bottom_layer_num + adapter_top_layer_num < len(
        transformer_layers), "adapter layer num must be less than the number of transformer layers"
    assert emulator_layer_num < len(
        transformer_layers) and emulator_layer_num > 0, "emulator layer num must be less than the number of transformer layers"

    bottom_idx = adapter_bottom_layer_num
    top_idx = len(transformer_layers) - adapter_top_layer_num
    bottom_layers = transformer_layers[:bottom_idx]
    top_layers = transformer_layers[top_idx:]
    kept_layers = transformer_layers[bottom_idx:top_idx]
    emulator = nn.ModuleList()
    stride = (len(kept_layers) - 1) / (emulator_layer_num - 1)

    layer_idx = []
    for i in range(emulator_layer_num):
        idx = int(round(i * stride))
        layer_idx.append(idx)
        emulator.append(kept_layers[idx])
    logger.info(
        'take layer {} of the original model as the emulator'.format(
            t.Tensor(layer_idx) +
            bottom_idx))
    return nn.ModuleList(emulator), nn.ModuleList(
        bottom_layers), nn.ModuleList(top_layers)



def split_numpy_array(embedding_matrix, n, suffix):
    # Calculate the indices where the splits should occur
    embedding_matrix = embedding_matrix['weight']
    indices = np.linspace(0, embedding_matrix.shape[0], n+1, dtype=int)

    # Split the embedding matrix at the calculated indices
    slices = [embedding_matrix[indices[i]:indices[i+1]] for i in range(n)]

    # Create a dictionary with the slices
    result_dict = {suffix+str(i): slice for i, slice in enumerate(slices)}
    return result_dict


def recover_numpy_array(slices_dict, suffix=""):
    # Get the slices from the dictionary and concatenate them
    slices = [slices_dict[suffix + str(i)] for i in range(len(slices_dict))]
    complete_array = np.concatenate(slices, axis=0)
    return {'weight': complete_array}


class OffsiteTuningBaseModel(t.nn.Module):

    def __init__(self, emulator_layer_num: int, adapter_top_layer_num: int = 2,
                 adapter_bottom_layer_num: int = 2, fp16_mix_precision=False):
        super().__init__()
        self.fp16_mix_precision = fp16_mix_precision
        self.model = self.get_base_model()
        self.initialize_model()
        self.emulator, self.adapter_bottom, self.adapter_top = get_dropout_emulator_and_adapters(
            transformer_layers=self.get_model_transformer_blocks(self.model),
            emulator_layer_num=emulator_layer_num,
            adapter_top_layer_num=adapter_top_layer_num,
            adapter_bottom_layer_num=adapter_bottom_layer_num
        )
        self.post_initialization()

    def initialize_model(self):
        if self.fp16_mix_precision:
            self.model.half()
        for param in self.model.parameters():
            param.requires_grad = False

    def post_initialization(self):
        pass

    def get_adapter_top(self):
        return self.adapter_top

    def get_adapter_bottom(self):
        return self.adapter_bottom

    def get_emulator(self):
        return self.emulator

    def get_additional_param_state_dict(self):
        # get parameter of additional parameter
        return {}

    def load_additional_param_state_dict(self, submodel_weights: dict):
        # load additional weights:
        pass

    def _get_numpy_arr(self, v):
        if v.dtype == t.bfloat16:
            # float 32
            v = v.detach().cpu().float().numpy()
        else:
            v = v.detach().cpu().numpy()

        return v


    def load_numpy_state_dict(self, module_dict, state_dict):
        param_dict = module_dict

        for k, v in param_dict.items():
            if k not in state_dict:
                continue
            addition_weights = {
                k: t.tensor(v) for k,
                v in state_dict[k].items()}
            v.load_state_dict(addition_weights)

    def get_numpy_state_dict(self, module_dict):

        weight_dict = {}
        for k, v in module_dict.items():
            weight_dict[k] = {
                k: self._get_numpy_arr(v) for k,
                v in v.state_dict().items()}
        return weight_dict

    def get_submodel_weights(self) -> dict:
        submodel_weights = {
            "emulator": {
                k: self._get_numpy_arr(v) for k,
                v in self.get_emulator().state_dict().items()},
            "adapter_top": {
                k: self._get_numpy_arr(v) for k,
                v in self.get_adapter_top().state_dict().items()},
            "adapter_bottom": {
                k: self._get_numpy_arr(v) for k,
                v in self.get_adapter_bottom().state_dict().items()}}
        addition_weights = self.get_additional_param_state_dict()
        submodel_weights.update(addition_weights)
        return submodel_weights

    def load_submodel_weights(self, submodel_weights: dict):

        emulator_weights = {
            k: t.tensor(v) for k,
            v in submodel_weights['emulator'].items()}
        adapter_top_weights = {
            k: t.tensor(v) for k,
            v in submodel_weights['adapter_top'].items()}
        adapter_bottom_weights = {
            k: t.tensor(v) for k,
            v in submodel_weights['adapter_bottom'].items()}

        emulator = self.get_emulator()
        adapter_top = self.get_adapter_top()
        adapter_bottom = self.get_adapter_bottom()

        emulator.load_state_dict(emulator_weights)
        adapter_top.load_state_dict(adapter_top_weights)
        adapter_bottom.load_state_dict(adapter_bottom_weights)
        self.load_additional_param_state_dict(submodel_weights)

    def forward(self, **kwargs):
        raise NotImplementedError()

    def get_base_model(self):
        raise NotImplementedError()

    def get_model_transformer_blocks(self, model: t.nn.Module):
        raise NotImplementedError()


class OffsiteTuningMainModel(OffsiteTuningBaseModel):

    def post_initialization(self):
        pass


class OffsiteTuningSubModel(OffsiteTuningBaseModel):

    def post_initialization(self):
        # mix precision model training
        for param in self.adapter_top.parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in self.adapter_bottom.parameters():
            param.data = param.data.float()
            param.requires_grad = True