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


from federatedml.util.transfer_variable.base_transfer_variable import Variable
import types


def scatter(host_variable: Variable, guest_variable: Variable, suffix=tuple()) -> types.GeneratorType:
    """
    scatter values from guest and hosts

    Args:
        host_variable: a variable represents `Host -> Arbiter`
        guest_variable: a variable represent `Guest -> Arbiter`
        suffix: additional suffix appended to transfer tag

    Returns:
        return a generator of scatted values
    """
    yield guest_variable.get(idx=0, suffix=suffix)
    for ret in host_variable.get(idx=-1, suffix=suffix):
        yield ret
