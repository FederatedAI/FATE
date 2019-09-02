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


class Scatter(object):

    def __init__(self, host_variable: Variable, guest_variable: Variable):
        """
            scatter values from guest and hosts

            Args:
                host_variable: a variable represents `Host -> Arbiter`
                guest_variable: a variable represent `Guest -> Arbiter`

            Returns:
                return a generator of scatted values
        """
        self._host_variable = host_variable
        self._guest_variable = guest_variable

    def get(self, suffix=tuple(), host_ids=None):
        """
        create a generator of values from guest and hosts.
        Args:
            suffix: tag suffix
            host_ids: ids of hosts to get value from.
                If None provided, get values from all hosts.
                If a list of int provided, get values from all hosts listed.
                Otherwise, ValueError raised
        """
        yield self._guest_variable.get(idx=0, suffix=suffix)
        if host_ids is None:
            host_ids = -1
        for ret in self._host_variable.get(idx=host_ids, suffix=suffix):
            yield ret
