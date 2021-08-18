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

hetero_nn_cpn_meta = ComponentMeta("HeteroNN")


@hetero_nn_cpn_meta.bind_param
def hetero_nn_param():
    from federatedml.param.hetero_nn_param import HeteroNNParam

    return HeteroNNParam


@hetero_nn_cpn_meta.bind_runner.on_guest
def hetero_nn_guest_runner():
    from federatedml.nn.hetero_nn.hetero_nn_guest import HeteroNNGuest

    return HeteroNNGuest


@hetero_nn_cpn_meta.bind_runner.on_host
def hetero_nn_host_runner():
    from federatedml.nn.hetero_nn.hetero_nn_host import HeteroNNHost

    return HeteroNNHost
