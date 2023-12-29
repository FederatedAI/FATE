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

from fate.arch.tensor import _custom_ops

from ._tensor import DTensor, implements


@implements(_custom_ops.encrypt_encoded_f)
def encrypt_encoded_f(input: DTensor, encryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.encrypt_encoded_f(x, encryptor), type="encrypted"))


@implements(_custom_ops.decrypt_encoded_f)
def decrypt_encoded_f(input: DTensor, decryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.decrypt_encoded_f(x, decryptor), type="encoded"))


@implements(_custom_ops.encrypt_f)
def encrypt_f(input: DTensor, encryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.encrypt_f(x, encryptor), type="encrypted"))


@implements(_custom_ops.decrypt_f)
def decrypt_f(input: DTensor, decryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.decrypt_f(x, decryptor), type="plain"))


@implements(_custom_ops.decode_f)
def decode_f(input: DTensor, coder):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.decode_f(x, coder), type="plain"))


@implements(_custom_ops.encode_f)
def encode_f(input: DTensor, coder):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.encode_f(x, coder), type="encoded"))
