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

from fate.arch import Context
from fate.arch.dataframe import DataLoader

from ..abc.module import HeteroModule

logger = logging.getLogger(__name__)


class LrModuleArbiter(HeteroModule):
    def __init__(
        self,
        batch_size,
        max_iter=100,
    ):
        self.max_iter = max_iter
        self.batch_size = batch_size

    def fit(self, ctx: Context) -> None:
        encryptor, decryptor = ctx.cipher.phe.keygen(options=dict(key_length=2048))
        # ctx.guest("encryptor").put(encryptor)  # ctx.guest.put("encryptor", encryptor)
        ctx.hosts("encryptor").put(encryptor)
        # num_batch = ctx.guest.get("num_batch")
        batch_loader = DataLoader(
            dataset=None, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="arbiter", sync_arbiter=True
        )
        logger.info(f"batch_num={batch_loader.batch_num}")
        step = 0
        for _, iter_ctx in ctx.range(self.max_iter):
            for batch_ctx, _ in iter_ctx.iter(batch_loader):
                g_guest_enc = batch_ctx.guest.get("g_enc")
                g = decryptor.decrypt(g_guest_enc)
                batch_ctx.guest.put("g", g)
                for i, g_host_enc in enumerate(batch_ctx.hosts.get("g_enc")):
                    g = decryptor.decrypt(g_host_enc)
                    batch_ctx.hosts[i].put("g", g)
                loss = decryptor.decrypt(batch_ctx.guest.get("loss"))
                iter_ctx.metrics.log_loss("lr_loss", loss.tolist(), step=step)
                logger.info(f"loss={loss}")
                step += 1
