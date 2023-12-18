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

from fate.arch.config import cfg
from ..unify import device as device_type

if typing.TYPE_CHECKING:
    from ._context import Context
    from ..tensor.phe import PHETensorCipher, PHETensorCipherPublic
logger = logging.getLogger(__name__)


class CipherKit:
    def __init__(self, device: device_type, cipher_mapping: typing.Optional[dict] = None) -> None:
        self._device = device
        if cipher_mapping is None:
            self._cipher_mapping = {}
        else:
            self._cipher_mapping = cipher_mapping
        self._allow_custom_random_seed = False
        self._custom_random_seed = 42

        self.ctx = None

    def set_ctx(self, ctx: "Context"):
        self.ctx = ctx

    def set_phe(self, device: device_type, options: typing.Optional[dict]):
        if "phe" not in self._cipher_mapping:
            self._cipher_mapping["phe"] = {}
        self._cipher_mapping["phe"][device] = options

    def _set_default_phe(self):
        if "phe" not in self._cipher_mapping:
            self._cipher_mapping["phe"] = {}
        if self._device not in self._cipher_mapping["phe"]:
            if self._device == device_type.CPU:
                self._cipher_mapping["phe"][device_type.CPU] = {
                    "kind": "paillier",
                    "key_length": cfg.safety.phe.paillier.minimum_key_size,
                }
            else:
                logger.warning(f"no impl exists for device {self._device}, fallback to CPU")
                self._cipher_mapping["phe"][device_type.CPU] = self._cipher_mapping["phe"].get(
                    device_type.CPU, {"kind": "paillier", "key_length": cfg.safety.phe.paillier.minimum_key_size}
                )

    @property
    def phe(self):
        if self.ctx is None:
            raise ValueError("context not set")
        self._set_default_phe()
        if self._device not in self._cipher_mapping["phe"]:
            raise ValueError(f"no impl exists for device {self._device}")
        return PHECipherBuilder(self.ctx, **self._cipher_mapping["phe"][self._device])

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
    def __init__(self, ctx: "Context", kind, key_length) -> None:
        self.ctx = ctx
        self.kind = kind
        self.key_length = key_length

    def broadcast(self, src: int = 0, options: typing.Optional[dict] = None, tag: str = "phe_cipher"):
        if src == self.ctx.rank:
            cipher = self.setup(options)
            cipher_public = cipher.to_public()
            for p in self.ctx.parties:
                if p.rank != src:
                    p.put(tag, cipher_public)
            return cipher
        else:
            return self.ctx.parties[src].get(name=tag)

    def setup(self, options: typing.Optional[dict] = None):
        if options is None:
            kind = self.kind
            key_size = self.key_length
        else:
            kind = options.get("kind", self.kind)
            key_size = options.get("key_length", self.key_length)

        if kind == "paillier":
            if not cfg.safety.phe.paillier.allow:
                raise ValueError("paillier is not allowed in config")
            if key_size < cfg.safety.phe.paillier.minimum_key_size:
                raise ValueError(
                    f"key size {key_size} is too small, minimum is {cfg.safety.phe.paillier.minimum_key_size}"
                )
            from fate.arch.protocol.phe.paillier import evaluator, keygen
            from fate.arch.tensor.phe import PHETensorCipher

            sk, pk, coder = keygen(key_size)
            tensor_cipher = PHETensorCipher.from_raw_cipher(pk, coder, sk, evaluator)

            return PHECipher(kind, key_size, pk, sk, evaluator, coder, tensor_cipher, True, True, True)

        if kind == "ou":
            if not cfg.safety.phe.ou.allow:
                raise ValueError("ou is not allowed in config")
            if key_size < cfg.safety.phe.ou.minimum_key_size:
                raise ValueError(f"key size {key_size} is too small, minimum is {cfg.safety.phe.ou.minimum_key_size}")
            from fate.arch.protocol.phe.ou import evaluator, keygen
            from fate.arch.tensor.phe import PHETensorCipher

            sk, pk, coder = keygen(key_size)
            tensor_cipher = PHETensorCipher.from_raw_cipher(pk, coder, sk, evaluator)
            return PHECipher(kind, key_size, pk, sk, evaluator, coder, tensor_cipher, False, False, True)

        elif kind == "mock":
            if not cfg.safety.phe.mock.allow:
                raise ValueError("mock is not allowed in config")
            from fate.arch.protocol.phe.mock import evaluator, keygen
            from fate.arch.tensor.phe import PHETensorCipher

            sk, pk, coder = keygen(key_size)
            tensor_cipher = PHETensorCipher.from_raw_cipher(pk, coder, sk, evaluator)
            return PHECipher(kind, key_size, pk, sk, evaluator, coder, tensor_cipher, True, False, False)

        else:
            raise ValueError(f"Unknown PHE keygen kind: {self.kind}")


class PHECipherPublic:
    def __init__(
        self,
        kind,
        key_size,
        pk,
        evaluator,
        coder,
        tensor_cipher: "PHETensorCipherPublic",
        can_support_negative_number,
        can_support_squeeze,
        can_support_pack,
    ) -> None:
        self._kind = kind
        self._key_size = key_size
        self._pk = pk
        self._coder = coder
        self._evaluator = evaluator
        self._tensor_cipher: "PHETensorCipherPublic" = tensor_cipher
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

    @property
    def pk(self):
        return self._pk

    @property
    def coder(self):
        return self._coder

    @property
    def evaluator(self):
        return self._evaluator


class PHECipher(PHECipherPublic):
    def __init__(
        self,
        kind,
        key_size,
        pk,
        sk,
        evaluator,
        coder,
        tensor_cipher: "PHETensorCipher",
        can_support_negative_number,
        can_support_squeeze,
        can_support_pack,
    ) -> None:
        super().__init__(
            kind,
            key_size,
            pk,
            evaluator,
            coder,
            tensor_cipher,
            can_support_negative_number,
            can_support_squeeze,
            can_support_pack,
        )
        self._sk = sk
        self._tensor_cipher = tensor_cipher

    def get_tensor_decryptor(self):
        return self._tensor_cipher.sk

    @property
    def sk(self):
        return self._sk

    def to_public(self):
        return PHECipherPublic(
            self.kind,
            self.key_size,
            self.pk,
            self.evaluator,
            self.coder,
            self._tensor_cipher.to_public(),
            self.can_support_negative_number,
            self.can_support_squeeze,
            self.can_support_pack,
        )
