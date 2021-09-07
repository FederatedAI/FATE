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

secure_add_example_cpn_meta = ComponentMeta("SecureAddExample")


@secure_add_example_cpn_meta.bind_param
def secure_add_example_param():
    from federatedml.param.secure_add_example_param import SecureAddExampleParam

    return SecureAddExampleParam


@secure_add_example_cpn_meta.bind_runner.on_guest
def secure_add_example_guest_runner():
    from federatedml.toy_example.secure_add_guest import SecureAddGuest

    return SecureAddGuest


@secure_add_example_cpn_meta.bind_runner.on_host
def secure_add_example_host_runner():
    from federatedml.toy_example.secure_add_host import SecureAddHost

    return SecureAddHost
