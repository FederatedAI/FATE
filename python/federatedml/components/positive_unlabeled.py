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

positive_unlabeled_cpn_meta = ComponentMeta("PositiveUnlabeled")


@positive_unlabeled_cpn_meta.bind_param
def positive_unlabeled_param():
    from federatedml.param.positive_unlabeled_param import PositiveUnlabeledParam

    return PositiveUnlabeledParam


@positive_unlabeled_cpn_meta.bind_runner.on_guest.on_host
def positive_unlabeled_client_runner():
    from federatedml.semi_supervised_learning.positive_unlabeled.positive_unlabeled_transformer import PositiveUnlabeled

    return PositiveUnlabeled
