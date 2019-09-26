import json
import uuid                                                                                                                                                                                                                                                               
import pandas as pd
import numpy as np

from arch.api import session
from contrib.fate_script import WorkMode
from contrib.fate_script import RuntimeInstance
from contrib.fate_script.standalone import fate_script as standalone_fate_script
from contrib.fate_script.cluster import fate_script as cluster_fate_script
from contrib.fate_script.utils.conf_variable import ConfVar
from contrib.fate_script.blas.blas import TensorInEgg, TensorInPy
from federatedml.util.param_checker import AllChecker

def init(job_id, runtime_conf, mode, server_conf_path="arch/conf/server_conf.json"):
    session.init(job_id, mode)
    print("runtime_conf:{}".format(runtime_conf))
    all_checker = AllChecker(runtime_conf)
    all_checker.check_all()
    with open(runtime_conf) as conf_p:
        runtime_json = json.load(conf_p)

    if mode is None:
        raise EnvironmentError("eggroll should be initialized before fate_script")
    if mode == WorkMode.STANDALONE:
        RuntimeInstance.FEDERATION = standalone_fate_script.init(job_id=job_id, runtime_conf=runtime_json)
    else:
        RuntimeInstance.FEDERATION = cluster_fate_script.init(job_id=job_id, runtime_conf=runtime_json, server_conf_path=server_conf_path)

def get(name, tag:str, idx = -1):
    return RuntimeInstance.FEDERATION.get(name=name, tag=tag, idx=idx)

def remote(obj, name:str, tag:str, role=None, idx=-1):
    return RuntimeInstance.FEDERATION.remote(obj=obj, name=name, tag=tag, role=role, idx=idx)

def get_runtime_conf():
    return RuntimeInstance.FEDERATION.runtime_conf

def init_encrypt_operator():
    return RuntimeInstance.FEDERATION.init_encrypt_operator()

def init_public_key(key_length=1024):
    return RuntimeInstance.FEDERATION.init_public_key(key_length)

def get_public_key(pub_key):
    return RuntimeInstance.FEDERATION.get_public_key(pub_key)

def get_data(file_path):
    return RuntimeInstance.FEDERATION.get_data(file_path)

def init_ml_conf():
    conf_var = ConfVar()
    conf_var.init_conf(RuntimeInstance.FEDERATION.role)
    return conf_var

def tensor_encrypt(tensor):
    return tensor.encrypt()

def tensor_decrypt(tensor):
    return tensor.decrypt(RuntimeInstance.FEDERATION.encrypt_operator)

def get_lr_x_table(file_path):
    ns = str(uuid.uuid1())
    csv_table = pd.read_csv(file_path)
    data = pd.read_csv(file_path).values
    x = session.table('fata_script_test_data_x_' + str(RuntimeInstance.FEDERATION.role + str(RuntimeInstance.FEDERATION.job_id)), ns, partition=2, persistent=True)
    if 'y' in list(csv_table.columns.values):
        data_index = 2
    else:
        data_index = 1
    for i in range(np.shape(data)[0]):
        x.put(data[i][0], data[i][data_index:])
    return TensorInEgg(RuntimeInstance.FEDERATION.encrypt_operator, None, x)


def get_lr_y_table(file_path):
    ns = str(uuid.uuid1())
    csv_table = pd.read_csv(file_path)
    data = pd.read_csv(file_path).values
    y = session.table('fata_script_test_data_y_' + str(RuntimeInstance.FEDERATION.role) + str(RuntimeInstance.FEDERATION.job_id), ns, partition=2, persistent=True)
    if 'y' not in list(csv_table.columns.values):
        raise RuntimeError("input data must contain y column")
    for i in range(np.shape(data)[0]):
        y.put(data[i][0], 1 if data[i][1] == 1 else -1)
    return TensorInEgg(RuntimeInstance.FEDERATION.encrypt_operator, None, y)

def get_lr_shape_w(file_path):
    csv_table = pd.read_csv(file_path)
    data = pd.read_csv(file_path).values
    if 'y' in list(csv_table.columns.values):
        data_index = 2
    else:
        data_index = 1

    shape_w = [data.shape[1] - data_index]
    return shape_w

def get_lr_w(file_path):
    w = np.zeros(get_lr_shape_w(file_path))
    return TensorInPy(RuntimeInstance.FEDERATION.encrypt_operator, None, w)

def get_lr_b(file_path):
    b = np.array([0])
    return TensorInPy(RuntimeInstance.FEDERATION.encrypt_operator, None, b)



