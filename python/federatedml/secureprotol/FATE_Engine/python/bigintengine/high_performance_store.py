from fate_arch.federation import segment_transfer_enabled

from ctypes import cdll, c_buffer, cast
from ctypes import c_char_p, c_void_p, c_uint32, c_size_t
import numpy as np
import os, importlib
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.secureprotol.FATE_Engine.python.bigintengine.fpga.fpga_engine import FLOAT_TYPE, PEN_BASE, MEM_HOST, CIPHER_BYTE, U_INT32_BYTE

HP_LIB = cdll.LoadLibrary(os.path.dirname(__file__) + "/../../C/FPGA/FPGA_LIB.so")

# DEFINE THE RETURN TYPE OF C_malloc #
HP_LIB.c_malloc.restype = c_void_p

'''Functions for interoperatability between heterogeneous FATE and open sourced FATE'''
def PEN_deserialize(PEN_storage): # convert a PEN_store to a ndarray of PaillierEncryptedNumber
    pub_key = PEN_storage.pub_key
    shape = PEN_storage.shape.to_tuple()
    vec_size = PEN_storage.shape.size()
    get_res = c_buffer(CIPHER_BYTE)
    pen_storage = PEN_storage.store.pen_storage
    exp_storage = PEN_storage.store.exp_storage
    cipher_list = []
    for i in range(vec_size):
        HP_LIB.c_memcpy(cast(get_res, c_void_p),c_void_p(pen_storage + i * CIPHER_BYTE), CIPHER_BYTE)
        cipher_list.append(int.from_bytes(get_res.raw, 'little'))
        
    exp_list = (c_uint32 * vec_size)(*[0 for _ in range(vec_size)])
    HP_LIB.c_memcpy(exp_list, c_void_p(exp_storage), c_size_t(vec_size * U_INT32_BYTE))
    PEN_list = []
    for i in range(vec_size):
        PEN_list.append(PaillierEncryptedNumber(pub_key, int(cipher_list[i]), int(round(exp_list[i]))))
        
    PEN_array = np.asarray(PEN_list).reshape(shape)
    return PEN_array

def PEN_serialize(PEN_array, key_list = None, device = 'gpu'): # convert a ndarray of PaillierEncryptedNumber to PEN_store
    lib_path = "federatedml.secureprotol.FATE_Engine.python.bigintengine." + device + '.' + device + "_store"
    module_name = 'PEN_store_'+ device
    lib_module = importlib.import_module(lib_path)
    PEN_store_device = getattr(lib_module, module_name)
    
    temp_shape = lib_module.TensorShapeStorage()
    temp_shape.from_tuple(PEN_array.shape)
    vec_size = temp_shape.size()
    PEN_array_flat = PEN_array.flat
    public_key = PEN_array_flat[0].public_key
    n = public_key.n
    max_int = public_key.max_int
    base_temp = []
    exp_temp = []
    base_storage = HP_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    exp_storage = HP_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    pen_storage = HP_LIB.c_malloc(c_size_t(vec_size * CIPHER_BYTE))
    for i in range(vec_size):
        src_number = PEN_array_flat[i].ciphertext(False).to_bytes(CIPHER_BYTE, 'little')
        HP_LIB.c_memcpy(
            c_void_p(pen_storage + i * CIPHER_BYTE), c_char_p(src_number), 
            c_size_t(CIPHER_BYTE))
        base_temp.append(PEN_BASE)
        exp_temp.append(PEN_array_flat[i].exponent)
    base_array_pointer = np.asarray(base_temp, np.uint32).ctypes.data_as(c_void_p)
    exp_array_pointer = np.asarray(exp_temp, np.uint32).ctypes.data_as(c_void_p)
    HP_LIB.c_memcpy(c_void_p(base_storage), base_array_pointer, c_size_t(vec_size * U_INT32_BYTE))
    HP_LIB.c_memcpy(c_void_p(exp_storage), exp_array_pointer, c_size_t(vec_size * U_INT32_BYTE))
    storage = lib_module.PaillierEncryptedStorage(
        pen_storage, base_storage, exp_storage, vec_size,
        MEM_HOST, FLOAT_TYPE, n, max_int)
    PEN_storage = PEN_store_device(storage, temp_shape, public_key)
    PEN_storage.key = key_list
    return PEN_storage


'''Functions for large data transmission between heterogeneous FATEs'''
class StorageTransfer(metaclass=segment_transfer_enabled()):
    def __init__(self, data):
        self._obj = data

    def get_data(self):
        return self._obj

class PEN_store(object):
    pass

class FPN_store(object):
    pass

class TE_store(object):
    pass

class RSA_store(object):
    pass
