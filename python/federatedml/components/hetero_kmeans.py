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

hetero_kmeans_cpn_meta = ComponentMeta("HeteroKmeans")


@hetero_kmeans_cpn_meta.bind_param
def hetero_kmeans_param():
    from federatedml.param.hetero_kmeans_param import KmeansParam

    return KmeansParam


@hetero_kmeans_cpn_meta.bind_runner.on_guest
def hetero_kmeans_runner_guest():
    from federatedml.unsupervised_learning.kmeans.hetero_kmeans.hetero_kmeans_client import (
        HeteroKmeansGuest,
    )

    return HeteroKmeansGuest


@hetero_kmeans_cpn_meta.bind_runner.on_host
def hetero_kmeans_runner_host():
    from federatedml.unsupervised_learning.kmeans.hetero_kmeans.hetero_kmeans_client import (
        HeteroKmeansHost,
    )

    return HeteroKmeansHost


@hetero_kmeans_cpn_meta.bind_runner.on_arbiter
def hetero_kmeans_runner_arbiter():
    from federatedml.unsupervised_learning.kmeans.hetero_kmeans.hetero_kmeans_arbiter import (
        HeteroKmeansArbiter,
    )

    return HeteroKmeansArbiter
