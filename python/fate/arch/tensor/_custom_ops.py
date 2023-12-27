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

import torch


def encrypt_f(tensor, encryptor):
    if isinstance(tensor, torch.Tensor):
        return encryptor.encrypt_tensor(tensor.detach())
    else:
        # torch tensor-like
        if hasattr(tensor, "__torch_function__"):
            return tensor.__torch_function__(encrypt_f, (type(tensor),), (tensor, encryptor), None)
    raise NotImplementedError("")


def encrypt_encoded_f(tensor, encryptor):
    if isinstance(tensor, torch.Tensor):
        return encryptor.encrypt_encoded(tensor.detach())
    else:
        # torch tensor-like
        if hasattr(tensor, "__torch_function__"):
            return tensor.__torch_function__(encrypt_encoded_f, (type(tensor),), (tensor, encryptor), None)
    raise NotImplementedError("")


def decrypt_encoded_f(tensor, decryptor):
    # torch tensor-like
    if hasattr(tensor, "__torch_function__"):
        return tensor.__torch_function__(decrypt_encoded_f, (type(tensor),), (tensor, decryptor), None)
    raise NotImplementedError("")


def encode_f(tensor, coder):
    if isinstance(tensor, torch.Tensor):
        return coder.encode(tensor.detach())
    else:
        # torch tensor-like
        if hasattr(tensor, "__torch_function__"):
            return tensor.__torch_function__(encode_f, (type(tensor),), (tensor, coder), None)
    raise NotImplementedError("")


def decrypt_f(tensor, decryptor):
    # torch tensor-like
    if hasattr(tensor, "__torch_function__"):
        return tensor.__torch_function__(decrypt_f, (type(tensor),), (tensor, decryptor), None)
    raise NotImplementedError(f"{type(tensor)}")


def decode_f(tensor, coder):
    if hasattr(tensor, "__torch_function__"):
        return tensor.__torch_function__(decode_f, (type(tensor),), (tensor, coder), None)
    raise NotImplementedError(f"{type(tensor)}")


def rmatmul_f(input, other):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        return torch.matmul(other, input)
    else:
        # torch tensor-like
        if isinstance(input, torch.Tensor):
            return torch.matmul(other, input)

        else:
            if hasattr(input, "__torch_function__"):
                return input.__torch_function__(rmatmul_f, (type(input), type(other)), (input, other), None)
    raise NotImplementedError("")


def to_local_f(input):
    if isinstance(input, torch.Tensor):
        return input

    else:
        # torch tensor-like
        if hasattr(input, "__torch_function__"):
            return input.__torch_function__(to_local_f, (type(input),), (input,), None)
    raise NotImplementedError("")


def slice_f(input, arg):
    if isinstance(input, torch.Tensor):
        return input[arg]

    else:
        # torch tensor-like
        if hasattr(input, "__torch_function__"):
            out = input.__torch_function__(slice_f, (type(input),), (input, arg), None)
            if out == NotImplemented:
                raise NotImplementedError(f"slice_f: {input}")
            return out

    raise NotImplementedError(f"slice_f: {input}")


def encode_as_int_f(input, precision: int):
    if isinstance(input, torch.Tensor):
        return (input * 2**precision).astype(torch.int64)
    else:
        # torch tensor-like
        if hasattr(input, "__torch_function__"):
            return input.__torch_function__(encode_as_int_f, (type(input),), (input, precision), None)
    raise NotImplementedError("")


# hook custom ops to torch
torch.encrypt_f = encrypt_f
torch.encrypt_encoded_f = encrypt_encoded_f
torch.decrypt_encoded_f = decrypt_encoded_f
torch.decrypt_f = decrypt_f
torch.encode_f = encode_f
torch.decode_f = decode_f
torch.rmatmul_f = rmatmul_f
torch.to_local_f = to_local_f
torch.slice_f = slice_f
torch.encode_as_int_f = encode_as_int_f
