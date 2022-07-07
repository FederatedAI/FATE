import torch
from fate_arch.tensor._context import CipherKind, Context
from fate_arch.tensor import PHETensor, FPTensor, GUEST, HOST
from federatedml.model_base import ComponentOutput, ModelBase
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables
from federatedml.util import LOGGER

from .params import TensorExampleParam


# noinspection PyAttributeOutsideInit
class TensorExampleTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.guest_cipher = self._create_variable(
            name="guest_cipher", src=["guest"], dst=["host"]
        )
        self.host_cipher = self._create_variable(
            name="host_cipher", src=["host"], dst=["guest"]
        )
        self.host_matmul_encrypted = self._create_variable(
            name="host_matmul_encrypted", src=["host"], dst=["guest"]
        )
        self.host_matmul = self._create_variable(
            name="host_matmul", src=["host"], dst=["guest"]
        )
        self.guest_matmul_encrypted = self._create_variable(
            name="guest_matmul_encrypted", src=["guest"], dst=["host"]
        )


class TensorExampleGuest(ModelBase):
    def __init__(self):
        super(TensorExampleGuest, self).__init__()
        self.transfer_inst = TensorExampleTransferVariable()
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
        ctx = Context()
        ctx.device_init()
        return self._run(ctx, cpn_input)

    def _run(self, ctx: Context, cpn_input):
        """we may add ctx parameter to run"""

        self._init_runtime_parameters(cpn_input)

        LOGGER.info("begin to make guest data")
        self.a = FPTensor(ctx, torch.rand((self.data_num, self.feature_num)))

        LOGGER.info("keygen")
        self.pk, self.sk = ctx.cypher_utils.keygen(CipherKind.PHE, 1024)

        LOGGER.info("encrypt data")
        self.ea = self.pk.encrypt(self.a)

        LOGGER.info("share encrypted data to host")
        self.ea.remote(HOST, "guest_cipher")

        LOGGER.info("get encrypted data from host")
        self.eb = PHETensor.get(ctx, HOST, "host_cipher")

        LOGGER.info("begin to get matmul of guest and host")
        self.es_guest = self.a.T @ self.eb

        LOGGER.info("send encrypted matmul to host")
        self.es_guest.remote(HOST, "guest_matmul_encrypted")

        LOGGER.info("receive encrypted matmul from guest")
        self.es_host = PHETensor.get(ctx, HOST, "host_matmul_encrypted")

        LOGGER.info("decrypt matmul")
        self.s_host = self.sk.decrypt(self.es_host)

        LOGGER.info("get decrypted matmul")
        self.s_guest = FPTensor.get(ctx, HOST, "host_matmul")

        LOGGER.info("assert matmul close")
        assert torch.allclose(self.s_host._tensor.T, self.s_guest._tensor)

        return ComponentOutput(self.save_data(), self.export_model(), self.save_cache())


class TensorExampleHost(ModelBase):
    def __init__(self):
        super(TensorExampleHost, self).__init__()
        self.transfer_inst = TensorExampleTransferVariable()
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
        ctx = Context()
        ctx.device_init()
        return self._run(ctx, cpn_input)

    def _run(self, ctx: Context, cpn_input):
        LOGGER.info("begin to init parameters of secure add example host")
        self._init_runtime_parameters(cpn_input)

        LOGGER.info("begin to make host data")
        self.b = FPTensor(ctx, torch.rand((self.data_num, self.feature_num)))

        LOGGER.info("keygen")
        self.pk, self.sk = ctx.cypher_utils.keygen(CipherKind.PHE, 1024)

        LOGGER.info("begin to encrypt")
        self.eb = self.pk.encrypt(self.b)

        LOGGER.info("share encrypted data to guest")
        self.eb.remote(GUEST, "host_cipher")

        LOGGER.info("get encrypted data from guest")
        self.ea = PHETensor.get(ctx, GUEST, "guest_cipher")

        LOGGER.info("begin to get matmul of host and guest")
        self.es_host = self.b.T @ self.ea

        LOGGER.info("send encrypted matmul to guest")
        self.es_host.remote(GUEST, "host_matmul_encrypted")

        LOGGER.info("get encrypted matmul from guest")
        self.es_guest = PHETensor.get(ctx, GUEST, "guest_matmul_encrypted")

        LOGGER.info("decrypt encrypted matmul from guest")
        self.s_guest = self.sk.decrypt(self.es_guest)

        LOGGER.info("send decrypted matmul to guest")
        self.s_guest.remote(GUEST, "host_matmul")

        return ComponentOutput(self.save_data(), self.export_model(), self.save_cache())
