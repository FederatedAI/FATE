# from python.bigintengine.gpu.gpu_engine import *
# from python.fate_script.contract.secureprotol.fate_paillier import PaillierEncryptedNumber
# from python.fate_script.contract.secureprotol.fate_paillier import PaillierPublicKey
from federatedml.secureprotol.FATE_Engine.python.bigintengine.gpu.gpu_engine import *
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.secureprotol.fate_paillier import PaillierPublicKey
from federatedml.secureprotol.FATE_Engine.python.bigintengine.high_performance_store import PEN_store, FPN_store, TE_store
import numpy as np
from federatedml.util import LOGGER

def check_gpu_pub_key(func):
    def wrapper(*args, **kargs):
        for arg in args:
            if isinstance(arg, PEN_store_gpu):
                if arg.gpu_pub_key is None:
                    arg.create_gpu_pub_key()
        res = func(*args, **kargs)
        return res
    return wrapper

class PEN_store_gpu(PEN_store):
    def __init__(self, store: PaillierEncryptedStorage, 
                       shape: TensorShapeStorage, 
                       pub_key: PaillierPublicKey):
        self.pub_key = pub_key
        self.store = store
        self.shape = shape
        self.gpu_pub_key = self.create_gpu_pub_key()
        self.key = None
    
    def __len__(self):
        return self.store.vec_size
    
    def __del__(self):
        del self.store
        del self.shape
        del self.gpu_pub_key
        self.gpu_pub_key = None
        self.store = None
        self.shape = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['store'] = pi_c2bytes(state['store'], None)
        del state['gpu_pub_key']
        state['gpu_pub_key'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.store = pi_bytes2c(self.store, None)
        self.gpu_pub_key = self.create_gpu_pub_key()
        return state

    def create_gpu_pub_key(self):
        c_pub_key = pi_p2c_pub_key(None, self.pub_key)
        gpu_pub_key = pi_h2d_pub_key(None, c_pub_key)
        del c_pub_key
        return gpu_pub_key

    def delete_gpu_pub_key(self):
        del self.gpu_pub_key
        self.gpu_pub_key = None

    def store_c2bytes(self):
        self.store = pi_c2bytes(self.store, None)
    
    def store_bytes2c(self):
        self.store = pi_bytes2c(self.store, None)

    def get_PEN_ndarray(self):
        cipher_array, _, exponent_array = pi_c2p(self.store)
        PEN_list = []
        for i, value in enumerate(cipher_array):
            PEN_list.append(
                PaillierEncryptedNumber(
                    self.pub_key, int(cipher_array[i]), int(round(exponent_array[i]))))
        return np.asarray(PEN_list).reshape(self.get_shape())

    @staticmethod
    def set_from_PaillierEncryptedNumber(cpu_encrypted):
        temp_shape = TensorShapeStorage()
        if isinstance(cpu_encrypted, PaillierEncryptedNumber):
            temp_shape = temp_shape.from_tuple((1,))
            cpu_encrypted = np.asarray([cpu_encrypted])
        elif isinstance(cpu_encrypted, list) or isinstance(cpu_encrypted, np.ndarray):
            cpu_encrypted = np.asarray(cpu_encrypted)
            if len(cpu_encrypted) < 1:
                raise RuntimeError("lenght of cpu_encrypted cannot less than 1")
            temp_shape.from_tuple(cpu_encrypted.shape)
        else:
            raise RuntimeError("cpu_encrypted type must be PaillierEncryptedNumber or list or ndarray")
        if np.ndim(cpu_encrypted) == 1:
            pub_key = cpu_encrypted[0].public_key
        elif np.ndim(cpu_encrypted) == 2:
            pub_key = cpu_encrypted[0][0].public_key
        else:
            raise RuntimeError("cpu_encrypted ndim must less than 2")
        temp_store = pi_p2c(None, cpu_encrypted)
        return PEN_store_gpu(temp_store, temp_shape, pub_key)

    @staticmethod
    def init_from_arr(arr, pub_key):
        te_store = TE_store_gpu(arr)
        fpn_store = te_store.encode(pub_key.n, pub_key.max_int)
        pen_store = fpn_store.encrypt(pub_key)
        return pen_store
    
    def obfuscation(self):
        obf_seeds = pi_gen_obf_seed(None, self.gpu_pub_key, len(self), 1024 // 8, None, None)
        res_store = pi_obfuscate(self.gpu_pub_key, self.store, obf_seeds, None, None)
        return PEN_store_gpu(res_store, self.shape, self.pub_key)
    
    # @check_gpu_pub_key
    def decrypt(self, priv_key):
        
        if self.store.vec_size == 0:
            return np.asarray([]).reshape(self.get_shape())
        gpu_priv_key = pi_h2d_priv_key(None, pi_p2c_priv_key(None, priv_key))
        te_res = pi_decrypt(self.gpu_pub_key, gpu_priv_key, self.store, None, None, None)
        res_shape = self.shape.to_tuple()
        res_array = te_c2p(te_res).reshape(res_shape)
        del gpu_priv_key
        return res_array
    
    # @check_gpu_pub_key
    def host2device(self):
        res_store = pi_h2d(self.gpu_pub_key, None, self.store, None)
        return PEN_store_gpu(res_store, self.shape, self.pub_key)
    
    def device2host(self):
        res_store = pi_d2h(None, self.store, None)
        return PEN_store_gpu(res_store, self.shape, self.pub_key)
    
    # @check_gpu_pub_key
    def __add__(self, other):
        if isinstance(other, PEN_store_gpu):
            res_store, res_shape = pi_add(self.gpu_pub_key, self.store, other.store, self.shape, other.shape, None, None, None)
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, FPN_store_gpu):
            pen_store = pi_encrypt(self.gpu_pub_key, other.store, None, None)
            res_store, res_shape = pi_add(self.gpu_pub_key, self.store, pen_store, self.shape, other.shape, None, None, None)
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, TE_store_gpu):
            fpn_store = other.encode(self.pub_key.n, self.pub_key.max_int)
            pen_store = fpn_store.encrypt(self.pub_key)
            res_store, res_shape = pi_add(self.gpu_pub_key, self.store, pen_store.store, self.shape, pen_store.shape, None, None, None)
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, int) or isinstance(other, float):
            other_array = np.asarray([other])
            te_shape = TensorShapeStorage()
            te_shape = te_shape.from_tuple(other_array.shape)
            te_store = te_p2c(other_array, None)
            fpn_store = fp_encode(te_store, self.pub_key.n, self.pub_key.max_int)
            pen_store = pi_encrypt(self.gpu_pub_key, fpn_store, None, None)
            res_store, res_shape = pi_add(self.gpu_pub_key, self.store, pen_store, self.shape, te_shape, None, None, None)
            del te_store
            del fpn_store
            del pen_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, list):
            other_array = np.asarray(other)
            shape_tuple = other_array.shape
            te_shape = TensorShapeStorage()
            te_shape = te_shape.from_tuple(shape_tuple)
            te_store = te_p2c(other_array, None)
            fpn_store = fp_encode(te_store, self.pub_key.n, self.pub_key.max_int)
            pen_store = pi_encrypt(self.gpu_pub_key, fpn_store, None, None)
            res_store, res_shape = pi_add(self.gpu_pub_key, self.store, pen_store, self.shape, te_shape, None, None, None)
            del te_store
            del fpn_store
            del pen_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, np.ndarray):
            te_store = te_p2c(other, None)
            shape_tuple = other.shape
            te_shape = TensorShapeStorage()
            te_shape = te_shape.from_tuple(shape_tuple)
            fpn_store = fp_encode(te_store, self.pub_key.n, self.pub_key.max_int)
            pen_store = pi_encrypt(self.gpu_pub_key, fpn_store, None, None)
            res_store, res_shape = pi_add(self.gpu_pub_key, self.store, pen_store, self.shape, te_shape, None, None, None)
            del te_store
            del fpn_store
            del pen_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
    
    # @check_gpu_pub_key
    def __mul__(self, other):
        if isinstance(other, FPN_store_gpu):
            res_store, res_shape = pi_mul(self.gpu_pub_key, self.store, other.store, self.shape, other.shape, None, None, None)
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, TE_store_gpu):
            fp_store = other.encode(self.pub_key.n, self.pub_key.max_int)
            res_store, res_shape = pi_mul(self.gpu_pub_key, self.store, fp_store.store, self.shape, fp_store.shape, None, None, None)
            del fp_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, int) or isinstance(other, float):
            other_array = np.asarray([other])
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(other_array.shape)
            te_store = te_p2c(other_array, None)
            fpn_store = fp_encode(te_store, self.pub_key.n, self.pub_key.max_int)
            res_store, res_shape = pi_mul(self.gpu_pub_key, self.store, fpn_store, self.shape, fpn_shape, None, None, None)
            del te_store
            del fpn_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, list):
            other_array = np.asarray(other)
            shape_tuple = other_array.shape
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(shape_tuple)
            te_store = te_p2c(other_array, None)
            fpn_store = fp_encode(te_store, self.pub_key.n, self.pub_key.max_int)
            res_store, res_shape = pi_mul(self.gpu_pub_key, self.store, fpn_store, self.shape, fpn_shape, None, None, None)
            del te_store
            del fpn_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, np.ndarray):
            te_store = te_p2c(other, None)
            shape_tuple = other.shape
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(shape_tuple)
            fpn_store = fp_encode(te_store, self.pub_key.n, self.pub_key.max_int)
            res_store, res_shape = pi_mul(self.gpu_pub_key, self.store, fpn_store, self.shape, fpn_shape, None, None, None)
            del te_store
            del fpn_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
    
    # @check_gpu_pub_key
    def __matmul__(self, other):
        if isinstance(other, PEN_store_gpu):
            raise NotImplementedError("Cannot do matmul between two encrypted number")
        if isinstance(other, FPN_store_gpu):
            res_store, res_shape = pi_matmul(self.gpu_pub_key, self.store, other.store, self.shape, other.shape, None, None, None)
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, TE_store_gpu):
            fp_store = other.encode(self.pub_key.n, self.pub_key.max_int)
            res_store, res_shape = pi_matmul(self.gpu_pub_key, self.store, fp_store.store, self.shape, fp_store.shape, None, None, None)
            del fp_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, int) or isinstance(other, float):
            other_array = np.asarray([other])
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(other_array.shape)
            te_store = te_p2c(other_array, None)
            fpn_store = fp_encode(te_store, self.pub_key.n, self.pub_key.max_int)
            res_store, res_shape = pi_matmul(self.gpu_pub_key, self.store, fpn_store, self.shape, fpn_shape, None, None, None)
            del te_store
            del fpn_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, list):
            other_array = np.asarray(other)
            shape_tuple = other_array.shape
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(shape_tuple)
            te_store = te_p2c(other_array, None)
            fpn_store = fp_encode(te_store, self.pub_key.n, self.pub_key.max_int)
            res_store, res_shape = pi_matmul(self.gpu_pub_key, self.store, fpn_store, self.shape, fpn_shape, None, None, None)
            del te_store
            del fpn_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
        if isinstance(other, np.ndarray):
            te_store = te_p2c(other, None)
            shape_tuple = other.shape
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(shape_tuple)
            fpn_store = fp_encode(te_store, self.pub_key.n, self.pub_key.max_int)
            res_store, res_shape = pi_matmul(self.gpu_pub_key, self.store, fpn_store, self.shape, fpn_shape, None, None, None)
            del te_store
            del fpn_store
            return PEN_store_gpu(res_store, res_shape, self.pub_key)
    
    # @check_gpu_pub_key
    def __rmul__(self, other):
        return self.__mul__(other)
    
    # @check_gpu_pub_key
    def __radd__(self, other):
        return self.__add__(other)

    # @check_gpu_pub_key
    def __sub__(self, other):
        return self + (other * -1)

    # @check_gpu_pub_key
    def __rsub__(self, other):
        return other + (self * -1)
    
    # @check_gpu_pub_key
    def __truediv__(self, scalar):
        return self.__mul__(1/scalar)
    
    # @check_gpu_pub_key
    def transpose(self):
        res_store, res_shape = pi_transpose(self.store, self.shape, None, None, None)
        return PEN_store_gpu(res_store, res_shape, self.pub_key)
    
    # @check_gpu_pub_key
    def slice(self, start, end, axis):
        res_store, res_shape = pi_slice(self.store, self.shape, start, end, axis, None, None, None)
        return PEN_store_gpu(res_store, res_shape, self.pub_key)

    # get one element from PEN_store_gpu
    # @check_gpu_pub_key
    def get_element(self, index, axis=1):
        end = index + 1
        if index == -1:
            if len(self.get_shape()) == 1:
                end = self.get_shape()[1 - axis]
            elif len(self.get_shape()) == 2:
                end = self.get_shape()[axis]
        return self.slice(index, end, axis)

    # @check_gpu_pub_key
    def cat(self, other, axis=0):
        if self.pub_key != other.pub_key:
            raise RuntimeError("Cannot do cat between two encrypted number of different pub_key")
        temp_store = [self.store, other.store]
        temp_shape = [self.shape, other.shape]
        temp_store, temp_shape = pi_cat(temp_store, temp_shape, axis, None, None)
        return PEN_store_gpu(temp_store, temp_shape, self.pub_key)

    # @check_gpu_pub_key
    def sum(self, axis=None):
        temp_store, temp_shape = pi_sum(self.gpu_pub_key, self.store, self.shape, axis, None, None, None)
        return PEN_store_gpu(temp_store, temp_shape, self.pub_key)
    
    # @check_gpu_pub_key
    def sum_with_index(self, indices, axis = None):
        res_store, res_shape = pi_sum_with_index(self.gpu_pub_key, self.store, self.shape, indices)
        return PEN_store_gpu(res_store, res_shape, self.pub_key)
    
    # @check_gpu_pub_key
    def sum_multi_index(self, indices, min_value = 0, max_value = None):
        # TODO: add batch in case indices is too long or self.store is too large
        max_num = max(indices) if max_value is None else max_value
        min_num = 0 if max_value is None else min_value
        
        res_store, res_shape = pi_sum_multi_index(
            self.gpu_pub_key, self.store, self.shape, indices, 
            min_num, max_num)
        return PEN_store_gpu(res_store, res_shape, self.pub_key)
    
    # @check_gpu_pub_key
    def batch_sum_multi_index(self, indices, min_value = 0, max_value = None):
        max_num = max(indices) if max_value is None else max_value
        min_num = 0 if max_value is None else min_value

        res_store, res_shape = pi_sum_batch_multi_index(
            self.gpu_pub_key, self.store, self.shape, indices, 
            min_value = min_num, max_value = max_num)
        return PEN_store_gpu(res_store, res_shape, self.pub_key)
    
    # @check_gpu_pub_key
    def accumulate_sum(self):
        res_store, res_shape = pi_accumulate(self.gpu_pub_key, self.pub_key.nsquare, self.store, self.shape)
        return PEN_store_gpu(res_store, res_shape, self.pub_key)
    
    # @check_gpu_pub_key
    def add_with_index(self, valid_index, valid_num):
        if isinstance(valid_num, int) or isinstance(valid_num, float):
            valid_store = PEN_store_gpu.init_from_arr([valid_num], self.pub_key)
        elif isinstance(valid_num, list) or isinstance(valid_num, np.ndarray):
            valid_store = PEN_store_gpu.init_from_arr((valid_num), self.pub_key)
        elif isinstance(valid_num, PaillierEncryptedNumber):
            valid_store = PEN_store_gpu.set_from_PaillierEncryptedNumber(valid_num)
        elif isinstance(valid_num, PEN_store_gpu):
            valid_store = valid_num
        res_store, res_shape = pi_add_with_index(
            self.gpu_pub_key, self.pub_key.nsquare, 
            self.store, self.shape, valid_store.store, valid_store.shape, 
            valid_index)
        return PEN_store_gpu(res_store, res_shape, self.pub_key)

    def r_dot(self, arr: np.ndarray):
        '''
        ndarray dot-mul PEN_store_gpu
        -----------------------------------
        '''
        pen_shape = self.shape.to_tuple()
        arr_shape = arr.shape
        te_arr = TE_store_gpu(arr)
        print(arr_shape, pen_shape)
        if len(arr_shape) == 0 or len(pen_shape) == 0:
            # res= te_arr * self
            res = self.__mul__(te_arr)
            return res
        if len(arr_shape) == 1 and len(pen_shape) == 1:
            if pen_shape[0] == arr_shape[0]:
                te_arr = TE_store_gpu(arr)
                return self.__mul__(te_arr).sum()
            else:
                raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
        if len(arr_shape) == 2 and len(pen_shape) == 2:
            te_arr = TE_store_gpu(arr)
            return te_arr @ self
        
        if len(arr_shape) == 2 and len(pen_shape) == 1:
            if arr_shape[1] == pen_shape[0]:
                # vertically broadcast
                # become (1, pen_shape[0]), and vertically broadcast
                # result should be (pen_shape[0], ) * (arr_shape[0], pen_shape[0])
                # result is (arr_shape[0], pen_shape[0])
                # then we should horizontally sum and got the final as (arr_shape[0],)
                return self.__mul__(te_arr).sum(1)
            else:
                raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
        
        if len(arr_shape) == 1 and len(pen_shape) == 2:
            if arr_shape[0] == pen_shape[0]:
                # return te_arr @ pen
                temp_arr = np.ascontiguousarray(arr.reshape(-1,1))
                te_temp = TE_store_gpu(temp_arr)
                return self.__mul__(te_temp).sum(0)
                # temp_pen = pen.transpose()
                # res = te_arr * temp_pen
                # return res.sum(1)
            else:
                raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
                

    def dot(self, arr: np.ndarray):
        '''PEN_store_gpu dot-mul ndarray'''
        pen_shape = self.shape.to_tuple()
        arr_shape = arr.shape
        te_arr = TE_store_gpu(arr)
        print(pen_shape, arr_shape)
        
        if len(pen_shape) == 0 or len(arr_shape) == 0:
            res = self.__mul__(te_arr)
            return res
        
        if len(pen_shape) == 1 and len(arr_shape) == 1:
            if pen_shape[0] == arr_shape[0]:
                te_arr = TE_store_gpu(arr)
                return self.__mul__(te_arr).sum()
                return (pen * te_arr).sum()
            else:
                raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
        
        if len(pen_shape) ==2 and len(arr_shape) == 2:
            return self.__matmul__(te_arr)
        
        if len(pen_shape) == 2 and len(arr_shape) == 1:
            if pen_shape[1] == arr_shape[0]:
                # horizontal broadcast
                # become (pen_shape[0], pen_shape[1]) and (arr_shape[])
                return self.__mul__(te_arr).sum(1)
            else:
                raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
        
        if len(pen_shape) == 1 and len(arr_shape) == 2:
            if pen_shape[0] == arr_shape[0]:
                # return te_arr @ pen
                # vertically broadcast
                # pen_shape should be (n,1) and broad
                '''old format, regarding transpose a large PEN_store_gpu in sum(0)'''
                # new_shape = TensorShapeStorage(pen_shape[0],1)
                # pen.reshape(new_shape)
                # return (pen * te_arr).sum(0)
                '''new format, regarding tranpose for an ndarray, not faster'''
                new_shape = TensorShapeStorage(1, pen_shape[0])
                self.reshape(new_shape)
                
                temp_list = arr.transpose().tolist()
                temp_te = TE_store_gpu(temp_list)
                
                res = self.__mul__(temp_te).sum(1)
                res_shape = TensorShapeStorage(arr_shape[1],)
                res.reshape(res_shape)
                return res
            else:
                raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
        
    def l_dot(self, arr: np.ndarray):
        '''
        PEN_store_gpu dot-mul ndarray
        '''
        pen_shape = self.shape.to_tuple()
        arr_shape = arr.shape
        te_arr = TE_store_gpu(arr)
        print(pen_shape, arr_shape)

        if len(pen_shape) == 0 or len(arr_shape) == 0:
            res = self.__mul__(te_arr)
            return res
        
        if len(pen_shape) == 1 and len(arr_shape) == 1:
            if pen_shape[0] == arr_shape[0]:
                te_arr = TE_store_gpu(arr)
                return (self.__mul__(te_arr)).sum()
            else:
                raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
        
        if len(pen_shape) ==2 and len(arr_shape) == 2:
            return self.__matmul__(te_arr)
        
        if len(pen_shape) == 2 and len(arr_shape) == 1:
            if pen_shape[1] == arr_shape[0]:
                # horizontal broadcast
                # become (pen_shape[0], pen_shape[1]) and (arr_shape[])
                return (self.__mul__(te_arr)).sum(1)
            else:
                raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
        
        if len(pen_shape) == 1 and len(arr_shape) == 2:
            if pen_shape[0] == arr_shape[0]:
                # vertically broadcast
                # pen_shape should be (n,1) and broad
                # '''old format, regarding transpose a large PEN_store_gpu in sum(0)'''
                # new_shape = TensorShapeStorage(pen_shape[0],1)
                # pen.reshape(new_shape)
                # return (pen * te_arr).sum(0)
                # '''new format, regarding tranpose for an ndarray, not faster'''
                # new_shape = TensorShapeStorage(1, pen_shape[0])
                # pen.reshape(new_shape)
                
                temp_list = arr.transpose().tolist()
                temp_te = TE_store_gpu(temp_list)
                
                res = (self.__mul__(temp_te)).sum(1)
                res_shape = TensorShapeStorage(arr_shape[1],)
                res.reshape(res_shape)
                return res
            else:
                raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")

    # @check_gpu_pub_key
    def reshape(self, dim0, dim1):
        new_shape = TensorShapeStorage(dim0, dim1)
        res_store, res_shape = pi_reshape(self.store, self.shape, new_shape, None, None, None)
        return PEN_store_gpu(res_store, res_shape, self.pub_key)

    # @check_gpu_pub_key
    def get_shape(self):
        return self.shape.to_tuple()
    
    # @check_gpu_pub_key
    def get_size(self):
        return self.shape.size()
    
    def partition_by_index(self, valid_index, valid_cnt):
        res_stores = pi_partition_by_index(self.store, valid_index, valid_cnt)
        return [PEN_store_gpu(store, shape, self.pub_key) for store, shape in res_stores]


class FPN_store_gpu(FPN_store):
    def __init__(self, store: FixedPointStorage, shape: TensorShapeStorage):
        self.store = store
        self.shape = shape
    
    @staticmethod
    def init_from_arr(arr, n, max_int, alignment = False):
        if isinstance(arr, list):
            arr = np.asarray(list)
        if not isinstance(arr, np.ndarray):
            raise NotImplementedError("FPN_store_gpu init now only support list or ndarray")
        shape_tuple = arr.shape
        fpn_shape = TensorShapeStorage()
        fpn_shape = fpn_shape.from_tuple(shape_tuple)

        arr_store = te_p2c(arr, None)
        fpn_store = fp_encode(arr_store, n, max_int)
        return FPN_store_gpu(fpn_store, fpn_shape)
    
    def encrypt(self, pub_key):
        c_pub_key = pi_p2c_pub_key(None, pub_key)
        gpu_pub_key = pi_h2d_pub_key(None, c_pub_key)
        pi_store = pi_encrypt(gpu_pub_key, self.store, None, None)
        del c_pub_key
        del gpu_pub_key
        return PEN_store_gpu(pi_store, self.shape, pub_key)
    
    def decode(self):
        te_store = fp_decode(self.store, None, None)
        res_shape = self.shape.to_tuple()
        res_array = te_c2p(te_store).reshape(res_shape)
        return res_array
    
    # @check_gpu_pub_key
    def __mul__(self, other):
        if isinstance(other, PEN_store_gpu):
            res_store, res_shape = pi_mul(other.gpu_pub_key, other.store, self.store, other.shape, self.shape, None, None, None)
            return PEN_store_gpu(res_store, res_shape, other.pub_key)
        if isinstance(other, FPN_store_gpu):
            res_store, res_shape = fp_mul(self.store, other.store, self.shape, other.shape, None, None, None)
            return FPN_store_gpu(res_store, res_shape)
        if isinstance(other, TE_store_gpu):
            fp_store = other.encode(self.store.encode_n, self.store.max_int)
            res_store, res_shape = fp_mul(self.store, fp_store.store, self.shape, fp_store.shape, None, None, None)
            return FPN_store_gpu(res_store, res_shape)
        if isinstance(other, int) or isinstance(other, float):
            other_array = np.asarray([other])
            fpn_shape = TensorShapeStorage(1,1)
            te_store = te_p2c(other_array, None)
            fpn_store = fp_encode(te_store, self.store.encode_n, self.store.max_int)
            res_store, res_shape = fp_mul(self.store, fpn_store, self.shape, fpn_shape, None, None, None)
            del te_store
            del fpn_store
            return FPN_store_gpu(res_store, res_shape)
        if isinstance(other, list):
            other_array = np.asarray(other)
            shape_tuple = other_array.shape
            if len(shape_tuple)>1:
                fpn_shape = TensorShapeStorage(shape_tuple[0], shape_tuple[1])
            else:
                fpn_shape = TensorShapeStorage(shape_tuple[0])
            te_store = te_p2c(other_array, None)
            fpn_store = fp_encode(te_store, self.store.encode_n, self.store.max_int)
            res_store, res_shape = fp_mul(self.store, fpn_store, self.shape, fpn_shape, None, None, None)
            del te_store
            del fpn_store
            return FPN_store_gpu(res_store, res_shape)
        if isinstance(other, np.ndarray):
            te_store = te_p2c(other, None)
            shape_tuple = other.shape
            if len(shape_tuple)>1:
                fpn_shape = TensorShapeStorage(shape_tuple[0], shape_tuple[1])
            else:
                fpn_shape = TensorShapeStorage(shape_tuple[0])
            fpn_store = fp_encode(te_store, self.store.encode_n, self.store.max_int)
            res_store, res_shape = fp_mul(self.store, fpn_store, self.shape, fpn_shape, None, None, None)
            del te_store
            del fpn_store
            return FPN_store_gpu(res_store, res_shape)
    
    # @check_gpu_pub_key
    def __matmul__(self, other):
        '''
        We assume that other will always be a PEN_store_gpu
        because FPN_store_gpu @ FPN_store_gpu/TE_store_gpu is meaningless,
        do it in numpy may be faster
        '''
        if isinstance(other, PEN_store_gpu):
            res_store, res_shape = \
                pi_rmatmul(other.gpu_pub_key, self.store, other.store, self.shape, other.shape,
                           None, None, None)
            return PEN_store_gpu(res_store, res_shape, other.pub_key)
        else:
            raise NotImplementedError("Currently only support FPN_store_gpu matmul PEN_store_gpu")
    
    def __len__(self):
        return self.store.vec_size
    
    def transpose(self):
        res_store, res_shape = fp_transpose(self.store, self.shape, None, None, None)
        return FPN_store_gpu(res_store, res_shape)
    
    def __del__(self):
        del self.store
        del self.shape
        self.store = None
        self.shape = None


class TE_store_gpu(TE_store):
    '''TensorStorage class for lists and ndarray'''
    def __init__(self, arr):
        '''
        Initialize a TE_store_gpu from a list or ndarray
        ------------------------
        arr: list or ndarray, other datatype not supported
        output:
            self.store: TensorStorage class, defined in gpu_engine.py
            self.shape: TensorShapeStorage class
        '''
        self.store = None
        self.shape = None
        if isinstance(arr, list):
            arr = np.asarray(arr)
        if isinstance(arr, np.ndarray):
            self.store = te_p2c(arr, None)
            shape_tuple = arr.shape
            self.shape = TensorShapeStorage()
            self.shape.from_tuple(shape_tuple)
        else:
            raise PermissionError("Currently only support tranpose list/ndarray into TE_store_gpu!")
    
    def __del__(self):
        del self.store
        self.store = None
        del self.shape
        self.shape = None
    
    def encode(self, n, max_int):
        fpn_store = fp_encode(self.store, n, max_int)
        return FPN_store_gpu(fpn_store, self.shape)
    
    def get_arr(self):
        '''
        return an ndarray representation of self.store
        '''
        arr = te_c2p(self.store)
        shape = self.shape.to_tuple()
        return arr.reshape(shape)
    
    # @check_gpu_pub_key
    def __add__(self, other):
        '''
        Only support that other is a PEN_store_gpu:
        '''
        if isinstance(other, PEN_store_gpu):
            temp_n = other.pub_key.n
            temp_max_int = other.pub_key.max_int
            fpn_store = self.encode(temp_n, temp_max_int)
            pen_store = fpn_store.encrypt(other.pub_key)
            res_store, res_shape = pi_add(other.gpu_pub_key, pen_store.store, other.store,
                pen_store.shape, other.shape, None, None, None)
            return PEN_store_gpu(res_store, res_shape, other.pub_key)
        else:
            raise NotImplementedError("Currently only support TE_store_gpu add PEN_store_gpu")
    
    # @check_gpu_pub_key
    def __mul__(self, other):
        '''
        Only support that other is a PEN_store_gpu:
        '''
        if isinstance(other, PEN_store_gpu):
            temp_n = other.pub_key.n
            temp_max_int = other.pub_key.max_int
            fpn_store = self.encode(temp_n, temp_max_int)
            res_store, res_shape = \
                pi_mul(other.gpu_pub_key, other.store, fpn_store.store, other.shape, fpn_store.shape, None, None, None)
            return PEN_store_gpu(res_store, res_shape, other.pub_key)
        elif isinstance(other, FPN_store_gpu):
            temp_n = other.pub_key.n
            temp_max_int = other.pub_key.max_int
            fpn_store = self.encode(temp_n, temp_max_int)
            res_store, res_shape = \
                fp_mul(fpn_store.store, other.store, fpn_store.shape, other.shape, None, None, None)
            return FPN_store_gpu(res_store, res_shape)
        else:
            raise NotImplementedError("Currently only support TE_store_gpu matmul PEN_store_gpu/ FPN_store_gpu")

    # @check_gpu_pub_key
    def __matmul__(self, other):
        '''
        We assume that other is a PEN_store_gpu:
        Because TE_store_gpu @ FPN/TE_store_gpu will be meaningless,
        and using numpy maybe faster
        '''
        if isinstance(other, PEN_store_gpu):
            temp_n = other.pub_key.n
            temp_max_int = other.pub_key.max_int  
            fpn_store = self.encode(temp_n, temp_max_int)
            res_store, res_shape = \
                pi_rmatmul(other.gpu_pub_key, fpn_store.store, other.store, fpn_store.shape, other.shape,
                           None, None, None)
            return PEN_store_gpu(res_store, res_shape, other.pub_key)
        else:
            raise NotImplementedError("Currently only support TE_store_gpu matmul PEN_store_gpu")
    
