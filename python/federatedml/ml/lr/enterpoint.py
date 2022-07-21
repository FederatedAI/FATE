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
    Context,
    LabeledDataloaderWrapper,
    UnlabeledDataloaderWrapper,
    CipherKind,
)
from fate_arch.tensor import functional as F
from fate_arch.tensor._bridge import MLBase


# d_i = 0.25 * \sum_{j}(w_j * x_{ij}) - 0.5 * y_i
# g_{ij} = d_i * x_{ij}
# g_j = 1/m * \sum_{i}(g_{ij})


class LRArbiter(MLBase):
    def __init__(self):
        super().__init__()
        self.max_iter = 10

    def _fit(self, ctx: Context, *args):
        # sync encryptor
        encryptor, decryptor = ctx.cypher_utils.keygen(CipherKind.PHE, key_length=1024)
        ctx.remote(target=GUEST + HOST, key="cipher", value=encryptor)

        with ctx.create_iter(self.max_iter) as iterations:
            for i in iterations:
                g_g = ctx.tensor_utils.phe_get(source=GUEST, key="g_g")
                g_h = ctx.tensor_utils.phe_get(source=HOST, key="g_h")

                # update
                dw_g = decryptor.decrypt(g_g)
                dw_h = decryptor.decrypt(g_h)

                # push
                dw_g.remote(GUEST, "dw_g")
                dw_h.remote(HOST, "dw_h")


class LRGuest(MLBase):
    def __init__(self):
        self.max_iter = 10
        self.alpha = 0.1

    def _fit(self, ctx: Context, data):
        # sync encryptor
        dataloader = LabeledDataloaderWrapper(
            data, max_iter=self.max_iter, with_intercept=True
        )
        encryptor = ctx.cypher_utils.phe_get_encryptor(ARBITER, key="cipher")

        num_feature = dataloader.shape[1]
        w_g = ctx.tensor_utils.zeros(shape=(num_feature, 1))

        with ctx.create_iter(self.max_iter) as iterations:
            for i in iterations:
                X, Y = dataloader.next_batch()
                wx_g = 0.25 * X @ w_g - 0.5 * Y

                # calculate
                encryptor.encrypt(wx_g).remote(HOST, "wx_g")
                wx_h = ctx.tensor_utils.phe_get(HOST, "wx_h")
                wx = wx_g + wx_h

                # gradian
                g_g = F.weighted_mean(X, wx)  # g = (d^T * X) / m

                g_g.remote(ARBITER, "g_g")
                dw_g = ctx.tensor_utils.get(ARBITER, "dw_g")

                # update weights
                w_g -= dw_g


class LRHost(MLBase):
    def __init__(self):
        self.max_iter = 10

    def _fit(self, ctx: Context, data):
        # sync encryptor
        dataloader = UnlabeledDataloaderWrapper(data, max_iter=self.max_iter)
        encryptor = ctx.cypher_utils.phe_get_encryptor(ARBITER, key="cipher")
        num_feature = dataloader.shape[1]
        w_h = ctx.tensor_utils.zeros(shape=(num_feature, 1))

        with ctx.create_iter(self.max_iter) as iterations:
            for i in iterations:
                X = dataloader.next_batch()
                wx_h = 0.25 * X @ w_h

                # calculate d
                encryptor.encrypt(wx_h).remote(HOST, "wx_h")
                wx_g = ctx.tensor_utils.phe_get(HOST, "wx_g")
                wx = wx_g + wx_h

                # gradian
                g_h = F.weighted_mean(X, wx)  # g = (d^T * X) / m

                g_h.remote(ARBITER, "g_h")
                dw_h = ctx.tensor_utils.get(ARBITER, "dw_h")

                # update weights
                w_h -= dw_h
