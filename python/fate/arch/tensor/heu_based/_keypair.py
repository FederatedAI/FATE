import typing

import torch
from heu import numpy as hnp
from heu import phe

if typing.TYPE_CHECKING:
    from ._tensor import PaillierTensor, PaillierTensorEncoded


class HeuTensorCipher:
    def __init__(self, pk, sk, coder) -> None:
        self._pk = pk
        self._coder = coder
        self._sk = sk

    @classmethod
    def from_raw_cipher(cls, phe_kit):
        kit = hnp.HeKit(phe_kit)
        pub_kit = hnp.setup(kit.public_key())
        return cls(pk=PHETensorEncryptor(pub_kit), coder=PHETensorCoder(pub_kit), sk=PHETensorDecryptor(kit))

    @property
    def pk(self):
        return self._pk

    @property
    def coder(self):
        return self._coder

    @property
    def sk(self):
        return self._sk


class PHETensorCoder:
    def __init__(self, pub_kit: hnp.HeKit) -> None:
        self._pub_kit = pub_kit
        self._float_encoder = pub_kit.float_encoder()
        self._int_encoder = pub_kit.int_encoder()

    def encode(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            from ._tensor import PaillierTensorEncoded

            shape = tensor.shape
            tensor = tensor.flatten()
            if tensor.dtype == torch.float64:
                data = self._pub_kit.array(tensor.detach().numpy(), self._float_encoder)
            elif tensor.dtype == torch.float32:
                data = self._pub_kit.array(tensor.detach().numpy(), self._float_encoder)
            elif tensor.dtype == torch.int64:
                data = self._pub_kit.array(tensor.detach().numpy(), self._int_encoder)
            elif tensor.dtype == torch.int32:
                data = self._pub_kit.array(tensor.detach().numpy(), self._int_encoder)
            else:
                raise NotImplementedError(f"{tensor.dtype} not supported")
            return PaillierTensorEncoded(shape, data, tensor.dtype)
        elif hasattr(tensor, "encode"):
            return tensor.encode(self)
        else:
            raise NotImplementedError(f"`{tensor}` not supported")

    # def decode(self, tensor: "PaillierTensorEncoded"):
    #     from ._tensor import PaillierTensorEncoded
    #
    #     if isinstance(tensor, PaillierTensorEncoded):
    #         return self._pub_kit.decode_vec(tensor.data, tensor.dtype).reshape(tensor.shape)
    #     elif hasattr(tensor, "decode"):
    #         return tensor.decode(self)
    #     else:
    #         raise NotImplementedError(f"`{tensor}` not supported")


class PHETensorEncryptor:
    def __init__(self, pub_kit: hnp.HeKit) -> None:
        self._pub_kit = pub_kit

    def encrypt_encoded(self, tensor: "PaillierTensorEncoded", obfuscate=False):
        from ._tensor import PaillierTensor, PaillierTensorEncoded

        if isinstance(tensor, PaillierTensorEncoded):
            data = self._pub_kit.encryptor().encrypt(tensor.data)
            return PaillierTensor(self._pub_kit, tensor.coder, tensor.shape, data, tensor.dtype)
        elif hasattr(tensor, "encrypt_encoded"):
            return tensor.encrypt_encoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def encrypt_tensor(self, tensor: torch.Tensor, obfuscate=False):
        return self.encrypt_encoded(self._coder.encode(tensor), obfuscate)


class PHETensorDecryptor:
    def __init__(self, kit) -> None:
        self._kit = kit

    def decrypt_encoded(self, tensor: "PaillierTensor"):
        from ._tensor import PaillierTensor, PaillierTensorEncoded

        if isinstance(tensor, PaillierTensor):
            data = self._sk.decrypt_to_encoded(tensor.data)
            return PaillierTensorEncoded(tensor.coder, tensor.shape, data, tensor.dtype)

        elif hasattr(tensor, "decrypt_encoded"):
            return tensor.decrypt_encoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def decrypt_tensor(self, tensor: "PaillierTensor"):
        return self._coder.decode(self.decrypt_encoded(tensor))
