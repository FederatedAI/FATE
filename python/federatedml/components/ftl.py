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

ftl_cpn_meta = ComponentMeta("FTL")


@ftl_cpn_meta.bind_param
def ftl_param():
    from federatedml.param.ftl_param import FTLParam

    return FTLParam


@ftl_cpn_meta.bind_runner.on_guest
def ftl_guest_runner():
    from federatedml.transfer_learning.hetero_ftl.ftl_guest import FTLGuest

    return FTLGuest


@ftl_cpn_meta.bind_runner.on_host
def ftl_host_runner():
    from federatedml.transfer_learning.hetero_ftl.ftl_host import FTLHost

    return FTLHost
