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
import deepspeed


def deepspeed_init(model, ds_config):
    deepspeed.init_distributed()
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model, optimizer, _, _ = deepspeed.initialize(model=model,
                                                  model_parameters=model_parameters,
                                                  config=ds_config)

    return model, optimizer


def is_zero3(ds_config):
    return ds_config.get("zero_optimization", {}).get("stage", -1)


def init_deepspeed_env(ds_config):
    """
    to enabled deepspeed stage3, these should be call first
    """
    if is_zero3(ds_config):
        from transformers.deepspeed import HfDeepSpeedConfig

        HfDeepSpeedConfig(ds_config)
