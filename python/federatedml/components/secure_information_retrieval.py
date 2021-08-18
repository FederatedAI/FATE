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

secure_information_retrieval_cpn_meta = ComponentMeta("SecureInformationRetrieval")


@secure_information_retrieval_cpn_meta.bind_param
def secure_information_retrieval_param():
    from federatedml.param.sir_param import SecureInformationRetrievalParam

    return SecureInformationRetrievalParam


@secure_information_retrieval_cpn_meta.bind_runner.on_guest
def secure_information_retrieval_guest_runner():
    from federatedml.secure_information_retrieval.secure_information_retrieval_guest import (
        SecureInformationRetrievalGuest,
    )

    return SecureInformationRetrievalGuest


@secure_information_retrieval_cpn_meta.bind_runner.on_host
def secure_information_retrieval_host_runner():
    from federatedml.secure_information_retrieval.secure_information_retrieval_host import (
        SecureInformationRetrievalHost,
    )

    return SecureInformationRetrievalHost
