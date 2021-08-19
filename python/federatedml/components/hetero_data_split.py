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

hetero_data_split_cpn_meta = ComponentMeta("HeteroDataSplit")


@hetero_data_split_cpn_meta.bind_param
def hetero_data_split_param():
    from federatedml.param.data_split_param import DataSplitParam

    return DataSplitParam


@hetero_data_split_cpn_meta.bind_runner.on_guest
def hetero_data_split_guest_runner():
    from federatedml.model_selection.data_split.hetero_data_split import (
        HeteroDataSplitGuest,
    )

    return HeteroDataSplitGuest


@hetero_data_split_cpn_meta.bind_runner.on_host
def hetero_data_split_host_runner():
    from federatedml.model_selection.data_split.hetero_data_split import (
        HeteroDataSplitHost,
    )

    return HeteroDataSplitHost
