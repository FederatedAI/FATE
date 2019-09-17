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

from federatedml.framework.weights import Weights
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.base_transfer_variable import Variable


class Arbiter(object):

    # noinspection PyAttributeOutsideInit
    def register_model_broadcaster(self, model_transfer: Variable):
        self._models_broadcast = model_transfer
        return self

    def send_model(self, model: Weights, ciphers_dict=None, suffix=tuple()):
        if ciphers_dict:
            self._models_broadcast.remote(obj=model.for_remote(),
                                          role=consts.GUEST,
                                          idx=0,
                                          suffix=suffix)
            for i, cipher in ciphers_dict.items():
                if cipher:
                    self._models_broadcast.remote(obj=model.encrypted(cipher, inplace=False).for_remote(),
                                                  role=consts.HOST,
                                                  idx=i,
                                                  suffix=suffix)
                else:
                    self._models_broadcast.remote(obj=model.for_remote(),
                                                  role=consts.HOST,
                                                  idx=i,
                                                  suffix=suffix)
        else:
            self._models_broadcast.remote(obj=model.for_remote(),
                                          role=None,
                                          idx=-1,
                                          suffix=suffix)


class Client(object):
    # noinspection PyAttributeOutsideInit
    def register_model_broadcaster(self, model_transfer: Variable):
        self._models_broadcast = model_transfer
        return self

    def get_model(self, suffix=tuple()):
        return self._models_broadcast.get(idx=0, suffix=suffix)


Guest = Client
Host = Client
