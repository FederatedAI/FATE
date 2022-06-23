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

from fate_arch.tensor import (
    ARBITER,
    GUEST,
    HOST,
    FedIter,
    FPTensor,
    LabeledDataloaderWrapper,
    PHECipher,
    PHECipherKind,
    PHEEncryptor,
    PHETensor,
    UnlabeledDataloaderWrapper,
)
from fate_arch.tensor import functional as F
from federatedml.model_base import ModelBase

fediter = FedIter()


class FEDKEY:
    cipher = fediter.declare_key("cipher")
    wx_g = fediter.declare_key("wxg")
    wx_h = fediter.declare_key("wxh")
    g_h = fediter.declare_key("g_h")
    g_g = fediter.declare_key("g_g")
    dw_g = fediter.declare_key("dw_g")
    dw_h = fediter.declare_key("dw_h")


# d_i = 0.25 * \sum_{j}(w_j * x_{ij}) - 0.5 * y_i
# g_{ij} = d_i * x_{ij}
# g_j = 1/m * \sum_{i}(g_{ij})


class LRArbiter(ModelBase):
    def __init__(self):
        self.phe_cipher_kind = PHECipherKind.PAILLIER

    def fit(self, *args):
        # sync encryptor
        cipher = PHECipher.keygen(kind=self.phe_cipher_kind)
        cipher.encryptor.push(GUEST + HOST, FEDKEY.cipher)

        for iternum in fediter:
            g_g = PHETensor.pull_one(GUEST, FEDKEY.g_g)
            g_h = PHETensor.pull_one(HOST, FEDKEY.g_h)

            # update
            dw_g = g_g.decrypt_phe(cipher.decryptor)
            dw_h = g_h.decrypt_phe(cipher.decryptor)

            # push
            dw_g.push(GUEST, FEDKEY.dw_g)
            dw_h.push(HOST, FEDKEY.dw_h)


class LRGuest(ModelBase):
    def __init__(self):
        self.max_iter = 10
        self.alpha = 0.1

    def fit(self, data_instance):
        # sync encryptor
        dataloader = LabeledDataloaderWrapper(
            data_instance, max_iter=self.max_iter, with_intercept=True
        )
        encryptor = PHEEncryptor.pull_one(ARBITER, FEDKEY.cipher)

        w_g = FPTensor.zeors(...)
        for iternum in fediter:
            X, Y = dataloader.__next__()
            wx_g = 0.25 * X @ w_g - 0.5 * Y

            # calculate d
            wx_g.encrypted_phe(encryptor).push(HOST, FEDKEY.wx_h)
            wx_h = PHETensor.pull_one(GUEST, FEDKEY.wx_g)
            wx = wx_g + wx_h

            # gradian
            g_g = F.weighted_mean(X, wx)  # g = (d^T * X) / m

            # push g_h(local gradian) to arbiter and pull dw_h(suggested weights diff) from arbiter
            g_g.push(ARBITER, FEDKEY.g_g)
            dw_g = FPTensor.pull_one(ARBITER, FEDKEY.dw_g)

            # update weights
            w_g -= dw_g


class LRHost(ModelBase):
    def __init__(self):
        self.max_iter = 10

    def fit(self, data_instance):
        # sync encryptor
        dataloader = UnlabeledDataloaderWrapper(data_instance, max_iter=self.max_iter)
        encryptor = PHEEncryptor.pull_one(ARBITER, FEDKEY.cipher)

        w_h = FPTensor.zeors(...)
        fediter.start()
        for X in dataloader:
            fediter.increse_iter()
            wx_h = 0.25 * X @ w_h

            # calculate d
            wx_h.encrypted_phe(encryptor).push(GUEST, FEDKEY.wx_h)
            wx_g = PHETensor.pull_one(HOST, FEDKEY.wx_g)
            wx = wx_g + wx_h

            # gradian
            g_h = F.weighted_mean(X, wx)  # g = (d^T * X) / m

            # push g_h(local gradian) to arbiter and pull dw_h(suggested weights diff) from arbiter
            g_h.push(ARBITER, FEDKEY.g_h)
            dw_h = FPTensor.pull_one(ARBITER, FEDKEY.dw_h)

            # update weights
            w_h -= dw_h
