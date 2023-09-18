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

import logging
import typing

from ..unify import device

logger = logging.getLogger(__name__)


class CipherKit:
    def __init__(self, device: device, cipher_mapping: typing.Optional[dict] = None) -> None:
        self._device = device
        if cipher_mapping is None:
            self._cipher_mapping = {}
        else:
            self._cipher_mapping = cipher_mapping
        self._allow_custom_random_seed = False
        self._custom_random_seed = 42

    def set_phe(self, device: device, options: typing.Optional[dict]):
        if "phe" not in self._cipher_mapping:
            self._cipher_mapping["phe"] = {}
        self._cipher_mapping["phe"][device] = options

    def _set_default_phe(self):
        if "phe" not in self._cipher_mapping:
            self._cipher_mapping["phe"] = {}
        if self._device not in self._cipher_mapping["phe"]:
            if self._device == device.CPU:
                self._cipher_mapping["phe"][device.CPU] = {"kind": "paillier", "key_length": 1024}
            else:
                logger.warning(f"no impl exists for device {self._device}, fallback to CPU")
                self._cipher_mapping["phe"][device.CPU] = self._cipher_mapping["phe"].get(
                    device.CPU, {"kind": "paillier", "key_length": 1024}
                )

    @property
    def phe(self):
        self._set_default_phe()
        if self._device not in self._cipher_mapping["phe"]:
            raise ValueError(f"no impl exists for device {self._device}")
        return PHECipherBuilder(**self._cipher_mapping["phe"][self._device])

    @property
    def allow_custom_random_seed(self):
        return self._allow_custom_random_seed

    def set_allow_custom_random_seed(self, allow_custom_random_seed):
        self._allow_custom_random_seed = allow_custom_random_seed

    def set_custom_random_seed(self, custom_random_seed):
        self._custom_random_seed = custom_random_seed

    def get_custom_random_seed(self):
        return self._custom_random_seed


class PHECipherBuilder:
    def __init__(self, kind, key_length) -> None:
        self.kind = kind
        self.key_length = key_length

    def setup(self, options: typing.Optional[dict] = None):
        if options is None:
            kind = self.kind
            key_size = self.key_length
        else:
            kind = options.get("kind", self.kind)
            key_size = options.get("key_length", 1024)

        if kind == "paillier":
            from fate.arch.protocol.phe.paillier import evaluator, keygen
            from fate.arch.tensor.phe import PHETensorCipher

            sk, pk, coder = keygen(key_size)
            tensor_cipher = PHETensorCipher.from_raw_cipher(pk, coder, sk, evaluator)
            return PHECipher(kind, key_size, pk, sk, evaluator, coder, tensor_cipher, True, True, True)

        if kind == "ou":
            from fate.arch.protocol.phe.ou import evaluator, keygen
            from fate.arch.tensor.phe import PHETensorCipher

            sk, pk, coder = keygen(key_size)
            tensor_cipher = PHETensorCipher.from_raw_cipher(pk, coder, sk, evaluator)
            return PHECipher(kind, key_size, pk, sk, evaluator, coder, tensor_cipher, False, False, True)

        elif kind == "mock":
            from fate.arch.protocol.phe.mock import evaluator, keygen
            from fate.arch.tensor.phe import PHETensorCipher

            sk, pk, coder = keygen(key_size)
            tensor_cipher = PHETensorCipher.from_raw_cipher(pk, coder, sk, evaluator)
            return PHECipher(kind, key_size, pk, sk, evaluator, coder, tensor_cipher, True, False, False)

        else:
            raise ValueError(f"Unknown PHE keygen kind: {self.kind}")


class PHECipher:
    def __init__(
        self,
        kind,
        key_size,
        pk,
        sk,
        evaluator,
        coder,
        tensor_cipher,
        can_support_negative_number,
        can_support_squeeze,
        can_support_pack,
    ) -> None:
        self._kind = kind
        self._key_size = key_size
        self._pk = pk
        self._sk = sk
        self._coder = coder
        self._evaluator = evaluator
        self._tensor_cipher = tensor_cipher
        self._can_support_negative_number = can_support_negative_number
        self._can_support_squeeze = can_support_squeeze
        self._can_support_pack = can_support_pack

    @property
    def kind(self):
        return self._kind

    @property
    def can_support_negative_number(self):
        return self._can_support_negative_number

    @property
    def can_support_squeeze(self):
        return self._can_support_squeeze

    @property
    def can_support_pack(self):
        return self._can_support_pack

    @property
    def key_size(self):
        return self._key_size

    def get_tensor_encryptor(self):
        return self._tensor_cipher.pk

    def get_tensor_coder(self):
        return self._tensor_cipher.coder

    def get_tensor_decryptor(self):
        return self._tensor_cipher.sk

    @property
    def pk(self):
        return self._pk

    @property
    def coder(self):
        return self._coder

    @property
    def sk(self):
        return self._sk

    @property
    def evaluator(self):
        return self._evaluator
