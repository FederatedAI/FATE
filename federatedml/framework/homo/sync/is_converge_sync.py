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

from federatedml.transfer_variable.transfer_class.base_transfer_variable import Variable
import types


class Arbiter(object):

    # noinspection PyAttributeOutsideInit
    def register_is_converge(self, is_converge_variable: Variable):
        self._is_converge_variable = is_converge_variable
        return self

    def check_converge_status(self, converge_func: types.FunctionType, converge_args, suffix=tuple()):
        is_converge = converge_func(*converge_args)
        self._is_converge_variable.remote(is_converge, role=None, idx=-1, suffix=suffix)
        return is_converge


class _Client(object):

    # noinspection PyAttributeOutsideInit
    def register_is_converge(self, is_converge_variable: Variable):
        self._is_converge_variable = is_converge_variable
        return self

    def get_converge_status(self, suffix=tuple()):
        return self._is_converge_variable.get(idx=0, suffix=suffix)


Guest = _Client
Host = _Client
