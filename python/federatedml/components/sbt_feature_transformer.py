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

from .components import ComponentMeta

sbt_feature_transformer_cpn_meta = ComponentMeta("SBTFeatureTransformer")


@sbt_feature_transformer_cpn_meta.bind_param
def sbt_feature_transformer_param():
    from federatedml.param.sbt_feature_transformer_param import SBTTransformerParam

    return SBTTransformerParam


@sbt_feature_transformer_cpn_meta.bind_runner.on_guest
def sbt_feature_transformer_guest_runner():
    from federatedml.feature.sbt_feature_transformer.sbt_feature_transformer import (
        HeteroSBTFeatureTransformerGuest,
    )

    return HeteroSBTFeatureTransformerGuest


@sbt_feature_transformer_cpn_meta.bind_runner.on_host
def sbt_feature_transformer_host_runner():
    from federatedml.feature.sbt_feature_transformer.sbt_feature_transformer import (
         HeteroSBTFeatureTransformerHost,
    )

    return HeteroSBTFeatureTransformerHost
