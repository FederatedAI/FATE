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

hetero_sshe_lr_cpn_meta = ComponentMeta("HeteroSSHELR")


@hetero_sshe_lr_cpn_meta.bind_param
def hetero_sshe_lr_param():
    from federatedml.param.hetero_sshe_lr_param import HeteroSSHELRParam

    return HeteroSSHELRParam


@hetero_sshe_lr_cpn_meta.bind_runner.on_guest
def hetero_sshe_lr_runner_guest():
    from federatedml.linear_model.bilateral_linear_model.hetero_sshe_logistic_regression.hetero_lr_guest import (
        HeteroLRGuest,
    )

    return HeteroLRGuest


@hetero_sshe_lr_cpn_meta.bind_runner.on_host
def hetero_sshe_lr_runner_host():
    from federatedml.linear_model.bilateral_linear_model.hetero_sshe_logistic_regression.hetero_lr_host import (
        HeteroLRHost,
    )

    return HeteroLRHost
