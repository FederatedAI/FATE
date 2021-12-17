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

hetero_linr_cpn_meta = ComponentMeta("HeteroLinR")


@hetero_linr_cpn_meta.bind_param
def hetero_linr_param():
    from federatedml.param.linear_regression_param import LinearParam

    return LinearParam


@hetero_linr_cpn_meta.bind_runner.on_guest
def hetero_linr_runner_guest():
    from federatedml.linear_model.coordinated_linear_model.linear_regression.hetero_linear_regression.hetero_linr_guest import (
        HeteroLinRGuest, )

    return HeteroLinRGuest


@hetero_linr_cpn_meta.bind_runner.on_host
def hetero_linr_runner_host():
    from federatedml.linear_model.coordinated_linear_model.linear_regression.hetero_linear_regression.hetero_linr_host import (
        HeteroLinRHost, )

    return HeteroLinRHost


@hetero_linr_cpn_meta.bind_runner.on_arbiter
def hetero_linr_runner_arbiter():
    from federatedml.linear_model.coordinated_linear_model.linear_regression.hetero_linear_regression.hetero_linr_arbiter import (
        HeteroLinRArbiter, )

    return HeteroLinRArbiter
