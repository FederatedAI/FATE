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

from ..unify import device

logger = logging.getLogger(__name__)


class CipherKit:
    def __init__(self, device: device, cipher_mapping=None) -> None:
        self._device = device
        self._cipher_mapping = cipher_mapping

    @property
    def phe(self):
        if self._cipher_mapping is None:
            if self._device == device.CPU:
                return PHECipherBuilder("paillier")
            else:
                logger.warning(f"no impl exists for device {self._device}, fallback to CPU")
                return PHECipherBuilder("paillier")

        if "phe" not in self._cipher_mapping:
            raise ValueError("phe is not set")

        if self._device not in self._cipher_mapping["phe"]:
            raise ValueError(f"phe is not set for device {self._device}")

        return PHECipherBuilder(self._cipher_mapping["phe"][self._device])


class PHECipherBuilder:
    def __init__(self, kind) -> None:
        self.kind = kind

    def setup(self, options):
        kind = options.get("kind", self.kind)
        key_length = options.get("key_length", 1024)

        if kind == "paillier_old":
            import fate_utils
            from fate.arch.tensor.paillier import PaillierTensorCipher

            pk, sk = fate_utils.tensor.keygen(key_length)
            tensor_cipher = PaillierTensorCipher.from_raw_cipher(pk, None, sk)
            return PHECipher(pk, sk, None, None, tensor_cipher)

        if kind == "paillier":
            from fate.arch.protocol.phe.paillier import evaluator, keygen, Coder
            from fate.arch.tensor.phe import PHETensorCipher

            sk, pk = keygen(key_length)
            coder = Coder.from_pk(pk)
            tensor_cipher = PHETensorCipher.from_raw_cipher(pk, coder, sk, evaluator)
            return PHECipher(pk, sk, evaluator, coder, tensor_cipher)

        if kind == "heu":
            from fate.arch.protocol.phe.heu import evaluator, keygen
            from fate.arch.tensor.phe import PHETensorCipher

            sk, pk, coder = keygen(key_length)
            tensor_cipher = PHETensorCipher.from_raw_cipher(pk, coder, sk, evaluator)
            return PHECipher(pk, sk, evaluator, coder, tensor_cipher)

        elif kind == "mock":
            from fate.arch.tensor.mock import PaillierTensorCipher

            tensor_cipher = PaillierTensorCipher(**options)
            return PHECipher(None, None, None, None, tensor_cipher)

        else:
            raise ValueError(f"Unknown PHE keygen kind: {self.kind}")


class PHECipher:
    def __init__(self, pk, sk, evaluator, coder, tensor_cipher) -> None:
        self._pk = pk
        self._sk = sk
        self._coder = coder
        self._evaluator = evaluator
        self._tensor_cipher = tensor_cipher

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
