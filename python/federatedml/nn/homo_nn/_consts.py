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

MODEL_META_NAME = "HomoNNModelMeta"
MODEL_PARAM_NAME = "HomoNNModelParam"


def _build_model_dict(meta, param):
    return {MODEL_META_NAME: meta, MODEL_PARAM_NAME: param}


def _extract_param(model_dict: dict):
    return model_dict.get(MODEL_PARAM_NAME, None)


def _extract_meta(model_dict: dict):
    return model_dict.get(MODEL_META_NAME, None)
