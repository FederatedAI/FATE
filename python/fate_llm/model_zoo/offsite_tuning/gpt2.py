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
from fate_llm.model_zoo.offsite_tuning.offsite_tuning_model import OffsiteTuningSubModel, OffsiteTuningMainModel, get_dropout_emulator_and_adapters, split_numpy_array, recover_numpy_array
from transformers import GPT2LMHeadModel, GPT2Config
import torch
from typing import Optional, Tuple


class GPT2LMHeadMainModel(OffsiteTuningMainModel):

    def __init__(
            self,
            model_name_or_path,
            emulator_layer_num: int,
            adapter_top_layer_num: int = 2,
            adapter_bottom_layer_num: int = 2):

        self.model_name_or_path = model_name_or_path
        super().__init__(
            emulator_layer_num,
            adapter_top_layer_num,
            adapter_bottom_layer_num)

    def get_base_model(self):
        return GPT2LMHeadModel.from_pretrained(self.model_name_or_path)

    def get_model_transformer_blocks(self, model: GPT2LMHeadModel):
        return model.transformer.h

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):

        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

    def get_additional_param_state_dict(self):
        # get parameter of additional parameter
        model = self.model
        param_dict = {
            'wte': model.transformer.wte,
            'wpe': model.transformer.wpe,
            'last_ln_f': model.transformer.ln_f
        }

        addition_weights = self.get_numpy_state_dict(param_dict)

        wte = addition_weights.pop('wte')
        wte_dict = split_numpy_array(wte, 10, 'wte')
        wpe = addition_weights.pop('wpe')
        wpe_dict = split_numpy_array(wpe, 10, 'wpe')
        addition_weights.update(wte_dict)
        addition_weights.update(wpe_dict)
        return addition_weights

    def load_additional_param_state_dict(self, submodel_weights: dict):
        # load additional weights:
        model = self.model
        param_dict = {
            'wte': model.transformer.wte,
            'wpe': model.transformer.wpe,
            'last_ln_f': model.transformer.ln_f
        }

        new_submodel_weight = {}
        new_submodel_weight['last_ln_f'] = submodel_weights['last_ln_f']
        wte_dict, wpe_dict = {}, {}
        for k, v in submodel_weights.items():
            if 'wte' in k:
                wte_dict[k] = v
            if 'wpe' in k:
                wpe_dict[k] = v
        wte = recover_numpy_array(wte_dict, 'wte')
        wpe = recover_numpy_array(wpe_dict, 'wpe')
        new_submodel_weight['wte'] = wte
        new_submodel_weight['wpe'] = wpe

        self.load_numpy_state_dict(param_dict, new_submodel_weight)


class GPT2LMHeadSubModel(OffsiteTuningSubModel):

    def __init__(
            self,
            model_name_or_path,
            emulator_layer_num: int,
            adapter_top_layer_num: int = 2,
            adapter_bottom_layer_num: int = 2,
            fp16_mix_precision=False,
            partial_weight_decay=None):

        self.model_name_or_path = model_name_or_path
        self.emulator_layer_num = emulator_layer_num
        self.adapter_top_layer_num = adapter_top_layer_num
        self.adapter_bottom_layer_num = adapter_bottom_layer_num
        super().__init__(
            emulator_layer_num,
            adapter_top_layer_num,
            adapter_bottom_layer_num,
            fp16_mix_precision)
        self.partial_weight_decay = partial_weight_decay

    def get_base_model(self):
        total_layer_num = self.emulator_layer_num + \
            self.adapter_top_layer_num + self.adapter_bottom_layer_num
        config = GPT2Config.from_pretrained(self.model_name_or_path)
        config.num_hidden_layers = total_layer_num
        # initialize a model without pretrained weights
        return GPT2LMHeadModel(config)

    def get_model_transformer_blocks(self, model: GPT2LMHeadModel):
        return model.transformer.h

    def get_additional_param_state_dict(self):
        # get parameter of additional parameter
        model = self.model
        param_dict = {
            'wte': model.transformer.wte,
            'wpe': model.transformer.wpe,
            'last_ln_f': model.transformer.ln_f
        }

        addition_weights = self.get_numpy_state_dict(param_dict)

        wte = addition_weights.pop('wte')
        wte_dict = split_numpy_array(wte, 10, 'wte')
        wpe = addition_weights.pop('wpe')
        wpe_dict = split_numpy_array(wpe, 10, 'wpe')
        addition_weights.update(wte_dict)
        addition_weights.update(wpe_dict)
        return addition_weights

    def load_additional_param_state_dict(self, submodel_weights: dict):
        # load additional weights:
        model = self.model
        param_dict = {
            'wte': model.transformer.wte,
            'wpe': model.transformer.wpe,
            'last_ln_f': model.transformer.ln_f
        }

        new_submodel_weight = {}
        new_submodel_weight['last_ln_f'] = submodel_weights['last_ln_f']
        wte_dict, wpe_dict = {}, {}
        for k, v in submodel_weights.items():
            if 'wte' in k:
                wte_dict[k] = v
            if 'wpe' in k:
                wpe_dict[k] = v
        wte = recover_numpy_array(wte_dict, 'wte')
        wpe = recover_numpy_array(wpe_dict, 'wpe')
        new_submodel_weight['wte'] = wte
        new_submodel_weight['wpe'] = wpe

        self.load_numpy_state_dict(param_dict, new_submodel_weight)

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):

        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

    def parameters(self, recurse=True):
        if self.partial_weight_decay is None:
            return super().parameters(recurse)
        elif isinstance(self.partial_weight_decay, float):
            no_decay = ["bias", "layer_norm.weight"]
            return [
                {
                    "params": [
                        p for n, p in self.named_parameters() if not any(
                            nd in n for nd in no_decay)], "weight_decay": self.partial_weight_decay}, {
                    "params": [
                        p for n, p in self.named_parameters() if any(
                            nd in n for nd in no_decay)], "weight_decay": 0.0}]
        else:
            raise ValueError(
                f"partial_weight_decay should be None or float, but got {self.partial_weight_decay}")