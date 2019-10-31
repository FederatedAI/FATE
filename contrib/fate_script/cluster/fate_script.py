import asyncio
import concurrent
import copy
import grpc
import pandas as pd
import numpy as np
import uuid
from arch.api.cluster.eggroll import _DTable, _EggRoll
from arch.api import  session
from arch.api.cluster import federation
from arch.api.proto import basic_meta_pb2, federation_pb2, federation_pb2_grpc, storage_basic_pb2
from arch.api.utils import file_utils, eggroll_serdes
from arch.api.utils.log_utils import getLogger
from arch.api.cluster.federation import FederationRuntime
from arch.api.cluster.eggroll import _DTable
from federatedml.secureprotol.encrypt import PaillierEncrypt
from contrib.fate_script.blas.blas import TensorInEgg, TensorInPy, Tensor

_serdes = eggroll_serdes.PickleSerdes

OBJECT_STORAGE_NAME = "__federation__"

CONF_KEY_FEDERATION = "federation"
CONF_KEY_LOCAL = "local"
CONF_KEY_SERVER = "servers"

ERROR_STATES = [federation_pb2.CANCELLED, federation_pb2.ERROR]


async def _async_receive(stub, transfer_meta):
    #LOGGER.debug("start receiving {}".format(transfer_meta))
    resp_meta = stub.recv(transfer_meta)
    while resp_meta.transferStatus != federation_pb2.COMPLETE:
        if resp_meta.transferStatus in ERROR_STATES:
            raise IOError(
                "receive terminated, state: {}".format(federation_pb2.TransferStatus.Name(resp_meta.transferStatus)))
        resp_meta = stub.checkStatusNow(resp_meta)
        await asyncio.sleep(1)
    return resp_meta


def _thread_receive(receive_func, check_func, transfer_meta):
    #LOGGER.debug("start receiving {}".format(transfer_meta))
    resp_meta = receive_func(transfer_meta)
    while resp_meta.transferStatus != federation_pb2.COMPLETE:
        if resp_meta.transferStatus in ERROR_STATES:
            raise IOError(
                "receive terminated, state: {}".format(federation_pb2.TransferStatus.Name(resp_meta.transferStatus)))
        resp_meta = check_func(resp_meta)
    #LOGGER.info("finish receiving {}".format(resp_meta))
    return resp_meta


def init(job_id, runtime_conf, server_conf_path):
    server_conf = file_utils.load_json_conf(server_conf_path)
    if CONF_KEY_SERVER not in server_conf:
        raise EnvironmentError("server_conf should contain key {}".format(CONF_KEY_SERVER))
    if CONF_KEY_FEDERATION not in server_conf.get(CONF_KEY_SERVER):
        raise EnvironmentError(
            "The {} should be a json file containing key: {}".format(server_conf_path, CONF_KEY_FEDERATION))
    _host = server_conf.get(CONF_KEY_SERVER).get(CONF_KEY_FEDERATION).get("host")
    _port = server_conf.get(CONF_KEY_SERVER).get(CONF_KEY_FEDERATION).get("port")
    
    federation_runtime = federation.init(job_id, runtime_conf, server_conf_path)
    return FateScript(federation_runtime, _host, _port)


class FateScript(FederationRuntime):
    def __init__(self, fed_obj, _host, _port):
        super().__init__(fed_obj.job_id, fed_obj.party_id, fed_obj.role, fed_obj.runtime_conf, _host, _port)
        self.trans_conf = file_utils.load_json_conf('contrib/fate_script/conf/FateScriptTransferVar.json')
        self.encrypt_operator = None

    def remote(self, obj, name: str, tag: str, role=None, idx=-1):
        if isinstance(obj, Tensor):
            super().remote(obj.store, name, tag, role, idx)
            #print("inside remote obj:{}".format(obj.store.count()))
        else:
            super().remote(obj, name,tag, role, idx)

    def get(self, name, tag, idx=-1):
        obj =copy.deepcopy( super().get(name, tag, idx))
        if isinstance(obj, _DTable):
            return copy.deepcopy(TensorInEgg(self.encrypt_operator, None, obj))        
        elif isinstance(obj, np.ndarray):
            return copy.deepcopy(TensorInPy(self.encrypt_operator, None, obj))
        else:
            return copy.deepcopy(obj)

    def init_encrypt_operator(self):
        self.encrypt_operator = PaillierEncrypt()

    def init_public_key(self, key_length=1024):
        self.encrypt_operator.generate_key(key_length)
        return self.encrypt_operator.get_public_key()

    def get_public_key(self, public_key):
        self.encrypt_operator.set_public_key(public_key)
