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

homo_data_split_cpn_meta = ComponentMeta("HomoDataSplit")


@homo_data_split_cpn_meta.bind_param
def homo_data_split_param():
    from federatedml.param.data_split_param import DataSplitParam

    return DataSplitParam


@homo_data_split_cpn_meta.bind_runner.on_guest
def homo_data_split_guest_runner():
    from federatedml.model_selection.data_split.homo_data_split import (
        HomoDataSplitGuest,
    )

    return HomoDataSplitGuest


@homo_data_split_cpn_meta.bind_runner.on_host
def homo_data_split_host_runner():
    from federatedml.model_selection.data_split.homo_data_split import HomoDataSplitHost

    return HomoDataSplitHost
