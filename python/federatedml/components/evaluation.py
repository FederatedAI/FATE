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

evaluation_cpn_meta = ComponentMeta("Evaluation")


@evaluation_cpn_meta.bind_param
def evaluation_param():
    from federatedml.param.evaluation_param import EvaluateParam

    return EvaluateParam


@evaluation_cpn_meta.bind_runner.on_guest.on_host.on_arbiter
def evaluation_runner():
    from federatedml.evaluation.evaluation import Evaluation

    return Evaluation
