import pandas as pd
import numpy as np
import uuid
from arch.api import  session
from arch.api.standalone import federation
from arch.api.standalone.federation import FederationRuntime
from arch.api.utils import file_utils
from federatedml.secureprotol.encrypt import PaillierEncrypt, FakeEncrypt
from contrib.fate_script.blas.blas import TensorInEgg, TensorInPy

OBJECT_STORAGE_NAME = "__federation__"
STATUS_TABLE_NAME = "__status__"

CONF_KEY_FEDERATION = "federation"
CONF_KEY_LOCAL = "local"

def init(job_id, runtime_conf):
    federation_runtime = federation.init(job_id, runtime_conf)
    return FateScript(federation_runtime)


class FateScript(FederationRuntime):
    def __init__(self, fed_obj):
        super().__init__(fed_obj.job_id, fed_obj.party_id, fed_obj.role, fed_obj.runtime_conf)
        self.trans_conf = file_utils.load_json_conf('contrib/fate_script/conf/FateScriptTransferVar.json')
        self.encrypt_operator = None

    def remote(self, obj, name: str, tag: str, role=None, idx=-1):
        super().remote(obj, name,tag, role, idx)

    def get(self, name, tag, idx=-1):
        return super().get(name, tag, idx)

    def init_encrypt_operator(self):
        self.encrypt_operator = PaillierEncrypt()

    def init_public_key(self, key_length=1024):
        self.encrypt_operator.generate_key(key_length)
        return self.encrypt_operator.get_public_key()

    def get_public_key(self, public_key):
        self.encrypt_operator.set_public_key(public_key)

