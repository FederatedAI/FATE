import logging

from fate.arch import tensor
from fate.interface import Context, ModelsLoader, ModelsSaver
from pandas import pandas

from ..abc.module import HeteroModule

logger = logging.getLogger(__name__)


class LrModuleArbiter(HeteroModule):
    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        self.max_iter = max_iter
        self.batch_size = 5

    def fit(self, ctx: Context, train_data) -> None:
        encryptor, decryptor = ctx.cipher.phe.keygen(options=dict(key_length=1024))
        ctx.guest("encryptor").put(encryptor)  # ctx.guest.put("encryptor", encryptor)
        ctx.hosts("encryptor").put(encryptor)
        num_batch = ctx.guest.get("num_batch")
        logger.info(f"num_batch={num_batch}")

        for _, iter_ctx in ctx.range(self.max_iter):
            for _, batch_ctx in iter_ctx.range(num_batch):
                g_guest_enc = batch_ctx.guest.get("g_enc")
                g = decryptor.decrypt(g_guest_enc)
                batch_ctx.guest.put("g", g)
                for i, g_host_enc in enumerate(batch_ctx.hosts.get("g_enc")):
                    g = decryptor.decrypt(g_host_enc)
                    batch_ctx.hosts[i].put("g", g)
                loss = decryptor.decrypt(batch_ctx.guest.get("loss"))
                logger.info(f"loss={loss}")
