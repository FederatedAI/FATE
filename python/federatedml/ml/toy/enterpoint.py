import torch
from fate_arch.tensor import GUEST, HOST, CipherKind, Context
from federatedml.model_base import ComponentOutput, ModelBase
from federatedml.util import LOGGER

from .params import TensorExampleParam


class TensorExampleGuest(ModelBase):
    def __init__(self):
        super(TensorExampleGuest, self).__init__()
        self.model_param = TensorExampleParam()
        self.data_output = None
        self.model_output = None

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)
        self._init_model()

    def _init_model(self):
        self.data_num = self.model_param.data_num
        self.feature_num = self.model_param.feature_num

    def run(self, cpn_input):
        ctx = Context.from_cpn_input(cpn_input)
        LOGGER.info(ctx.describe())
        return self._run(ctx, cpn_input)

    def _run(self, ctx: Context, cpn_input):
        """we may add ctx parameter to run"""

        self._init_runtime_parameters(cpn_input)

        LOGGER.info("begin to make guest data")
        self.a = ctx.random_tensor((self.data_num, self.feature_num))

        LOGGER.info("keygen")
        self.pk, self.sk = ctx.keygen(CipherKind.PHE, 1024)

        LOGGER.info("encrypt data")
        self.ea = self.pk.encrypt(self.a)

        LOGGER.info("share encrypted data to host")
        ctx.push(HOST, "guest_cipher", self.ea)

        LOGGER.info("get encrypted data from host")
        self.eb = ctx.pull(HOST, "host_cipher").unwrap_phe_tensor()

        LOGGER.info("begin to get matmul of guest and host")
        self.es_guest = self.a + self.eb
        # LOGGER.info("begin to get matmul of guest and host")
        # self.es_guest = self.a.T @ self.eb

        LOGGER.info("send encrypted matmul to host")
        ctx.push(HOST, "guest_matmul_encrypted", self.es_guest)

        LOGGER.info("receive encrypted matmul from guest")
        self.es_host = ctx.pull(HOST, "host_matmul_encrypted").unwrap_phe_tensor()

        LOGGER.info("decrypt matmul")
        self.s_host = self.sk.decrypt(self.es_host)

        LOGGER.info("get decrypted matmul")
        self.s_guest = ctx.pull(HOST, "host_matmul").unwrap_tensor()

        LOGGER.info("assert matmul close")
        sb = self.s_host._tensor._blocks_table.count()
        sa = self.s_guest._tensor._blocks_table.count()
        assert sa == sb
        a = list(self.s_guest._tensor._blocks_table.collect())[0]
        b = list(self.s_host._tensor._blocks_table.collect())[0]
        assert torch.allclose(a[1], b[1])
        # assert torch.allclose(self.s_host._tensor.T, self.s_guest._tensor)

        return ComponentOutput(self.save_data(), self.export_model(), self.save_cache())


class TensorExampleHost(ModelBase):
    def __init__(self):
        super(TensorExampleHost, self).__init__()
        self.model_param = TensorExampleParam()
        self.data_output = None
        self.model_output = None

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)
        self._init_model()

    def _init_model(self):
        self.data_num = self.model_param.data_num
        self.feature_num = self.model_param.feature_num

    def run(self, cpn_input):
        ctx = Context.from_cpn_input(cpn_input)
        LOGGER.info(ctx.describe())
        return self._run(ctx, cpn_input)

    def _run(self, ctx: Context, cpn_input):
        LOGGER.info("begin to init parameters of secure add example host")
        self._init_runtime_parameters(cpn_input)

        LOGGER.info("begin to make host data")
        self.b = ctx.random_tensor((self.data_num, self.feature_num))

        with ctx.iter_namespaces(10, prefix_name="tree_") as iteration:
            for i, _ in enumerate(iteration):
                print(ctx.current_namespace())

        LOGGER.info("keygen")
        self.pk, self.sk = ctx.keygen(CipherKind.PHE, 1024)

        LOGGER.info("begin to encrypt")
        self.eb = self.pk.encrypt(self.b)

        LOGGER.info("share encrypted data to guest")
        ctx.push(GUEST, "host_cipher", self.eb)

        LOGGER.info("get encrypted data from guest")
        self.ea = ctx.pull(GUEST, "guest_cipher").unwrap_phe_tensor()

        LOGGER.info("begin to get matmul of host and guest")
        self.es_host = self.b + self.ea
        # LOGGER.info("begin to get matmul of host and guest")
        # self.es_host = self.b.T @ self.ea

        LOGGER.info("send encrypted matmul to guest")
        ctx.push(GUEST, "host_matmul_encrypted", self.es_host)

        LOGGER.info("get encrypted matmul from guest")
        self.es_guest = ctx.pull(GUEST, "guest_matmul_encrypted").unwrap_phe_tensor()

        LOGGER.info("decrypt encrypted matmul from guest")
        self.s_guest = self.sk.decrypt(self.es_guest)

        LOGGER.info("send decrypted matmul to guest")
        ctx.push(GUEST, "host_matmul", self.s_guest)

        return ComponentOutput(self.save_data(), self.export_model(), self.save_cache())
