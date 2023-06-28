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

from ..unify import device


class CipherKit:
    def __init__(self, device: device) -> None:
        self.device = device

    @property
    def phe(self):
        return PHECipher(self.device)


class PHECipher:
    def __init__(self, _device) -> None:
        self.device = _device

    def keygen(self, **kwargs):
        from fate.arch.tensor import keygen

        return keygen(self.device, **kwargs)
