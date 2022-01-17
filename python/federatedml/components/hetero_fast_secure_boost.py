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

hetero_secure_boost_cpn_meta = ComponentMeta("HeteroFastSecureBoost")


@hetero_secure_boost_cpn_meta.bind_param
def hetero_secure_boost_param():
    from federatedml.param.boosting_param import HeteroSecureBoostParam

    return HeteroSecureBoostParam


@hetero_secure_boost_cpn_meta.bind_runner.on_guest
def hetero_secure_boost_guest_runner():
    from federatedml.ensemble import (HeteroSecureBoostingTreeGuest)

    return HeteroSecureBoostingTreeGuest


@hetero_secure_boost_cpn_meta.bind_runner.on_host
def hetero_secure_boost_host_runner():
    from federatedml.ensemble import (HeteroSecureBoostingTreeHost)

    return HeteroSecureBoostingTreeHost
