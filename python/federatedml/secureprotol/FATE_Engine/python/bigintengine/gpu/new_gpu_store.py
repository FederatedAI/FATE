from python.bigintengine.gpu.gpu_engine import *
from python.bigintengine.gpu.gpu_store_uuid import Store_uuid
from python.fate_script.contract.secureprotol.fate_paillier import PaillierEncryptedNumber
from python.fate_script.contract.secureprotol.fate_paillier import PaillierPublicKey
import numpy as np
import uuid

def check_gpu_pub_key(func):
    def wrapper(*args, **kargs):
        for arg in args:
            if isinstance(arg, PEN_store):
                if arg.gpu_pub_key is None:
                    arg.create_gpu_pub_key()
        res = func(*args, **kargs)
        return res
    return wrapper

class PEN_store:
    def __init__(self, store_id: Store_uuid, 
                       shape: TensorShapeStorage, 
                       pub_key: PaillierPublicKey):
        if isinstance(store_id, Store_uuid) is False:
            raise RuntimeError(f"Illegal store_id type : {type(store_id)}, params need type : {Store_uuid}")
        self.pub_key = pub_key
        self.store_id = store_id
        self.shape = shape
        self.gpu_pub_key = None
    
    def __len__(self):
        return self.store_id.get_store().vec_size
    
    def __del__(self):
        del self.store_id
        del self.shape
        del self.gpu_pub_key
        self.gpu_pub_key = None
        self.shape = None
    

    def __getstate__(self):
        state = self.__dict__.copy()
        state['store_id'] = pi_c2bytes(self.store_id.get_store(), None)
        del state['gpu_pub_key']
        state['gpu_pub_key'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

        t_id = Store_uuid.pi_alloc(self.shape.size())
        
        _ = pi_bytes2c(self.store_id, t_id.get_store())
        self.store_id = t_id
        self.gpu_pub_key = None
        return state

    def create_gpu_pub_key(self):
        c_pub_key = pi_p2c_pub_key(None, self.pub_key)
        self.gpu_pub_key = pi_h2d_pub_key(None, c_pub_key)
        del c_pub_key

    def delete_gpu_pub_key(self):
        del self.gpu_pub_key
        self.gpu_pub_key = None

    def get_PaillierEncryptedNumber_ndarray(self):
        cipher_array, _, exponent_array = pi_c2p(self.store_id.get_store())
        paillierEncryptedNumber_list = []
        for i, value in enumerate(cipher_array):
            paillierEncryptedNumber_list.append(PaillierEncryptedNumber(self.pub_key, \
                    int(cipher_array[i]), int(round(exponent_array[i]))))
        return np.asarray(paillierEncryptedNumber_list).reshape(self.get_shape())

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
        
        temp_store_id = Store_uuid.pi_alloc(temp_shape.size())
        _ = pi_p2c(temp_store_id.get_store(), cpu_encrypted)

        return PEN_store(temp_store_id, temp_shape, pub_key)

    @check_gpu_pub_key
    def decrypt(self, priv_key):
        gpu_priv_key = pi_h2d_priv_key(None, pi_p2c_priv_key(None, priv_key))

        te_res_id = Store_uuid.te_alloc(self.shape.size())

        _ = pi_decrypt(self.gpu_pub_key, gpu_priv_key, self.store_id.get_store(),
                            te_res_id.get_store(), None, None)
        res_shape = self.shape.to_tuple()
        res_array = te_c2p(te_res_id.get_store()).reshape(res_shape)
        del gpu_priv_key
        return res_array
    
    @check_gpu_pub_key
    def __add__(self, other):
        if isinstance(other, PEN_store):
            res_size = get_add_mul_size(self.shape, other.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)

            _, res_shape = pi_add(self.gpu_pub_key, self.store_id.get_store(), other.store_id.get_store(),
                                     self.shape, other.shape, res_store_id.get_store(),None, None)
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, FPN_store):
            pen_store_id = Store_uuid.pi_alloc(other.shape.size())

            _ = pi_encrypt(self.gpu_pub_key, other.store_id.get_store(), pen_store_id.get_store(), None)

            res_size = get_add_mul_size(self.shape, other.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)

            _, res_shape = pi_add(self.gpu_pub_key, self.store_id.get_store(), pen_store_id.get_store(),
                                     self.shape, other.shape, res_store_id.get_store(), None, None)
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, TE_store):
            fpn_store = other.encode(self.pub_key.n, self.pub_key.max_int)
            pen_store = fpn_store.encrypt(self.pub_key)

            res_size = get_add_mul_size(self.shape, pen_store.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_add(self.gpu_pub_key, self.store_id.get_store(), pen_store.store_id.get_store(),
                                        self.shape, pen_store.shape, res_store_id.get_store(), None, None)
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, int) or isinstance(other, float):
            other_array = np.asarray([other])
            te_shape = TensorShapeStorage()
            te_shape = te_shape.from_tuple(other_array.shape)

            te_store_id = Store_uuid.te_alloc(1)
            _ = te_p2c(other_array, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(1)
            _ = fp_encode(te_store_id.get_store(), self.pub_key.n, self.pub_key.max_int,
                                        None, None, fpn_store_id.get_store(), None)

            pen_store_id = Store_uuid.pi_alloc(1)
            _ = pi_encrypt(self.gpu_pub_key, fpn_store_id.get_store(), pen_store_id.get_store(), None)
            
            res_size = get_add_mul_size(self.shape, te_shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_add(self.gpu_pub_key, self.store_id.get_store(), pen_store_id.get_store(),
                                         self.shape, te_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            del pen_store_id
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, list):
            other_array = np.asarray(other)
            shape_tuple = other_array.shape
            te_shape = TensorShapeStorage()
            te_shape = te_shape.from_tuple(shape_tuple)

            te_store_id = Store_uuid.te_alloc(te_shape.size())
            _ = te_p2c(other_array, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(te_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.pub_key.n, self.pub_key.max_int,
                            None, None, fpn_store_id.get_store(), None)
            
            pen_store_id = Store_uuid.pi_alloc(te_shape.size())
            _ = pi_encrypt(self.gpu_pub_key, fpn_store_id.get_store(), pen_store_id.get_store(), None)

            res_size = get_add_mul_size(self.shape, te_shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_add(self.gpu_pub_key, self.store_id.get_store(), pen_store_id.get_store(),
                                     self.shape, te_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            del pen_store_id
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, np.ndarray):
            shape_tuple = other.shape
            te_shape = TensorShapeStorage()
            te_shape = te_shape.from_tuple(shape_tuple)

            te_store_id = Store_uuid.te_alloc(te_shape.size())
            _ = te_p2c(other, te_store_id.get_store())

            
            fpn_store_id = Store_uuid.fp_alloc(te_shape.size())
            _  = fp_encode(te_store_id.get_store(), self.pub_key.n, self.pub_key.max_int,
                            None, None, fpn_store_id.get_store(), None)
            
            
            pen_store_id = Store_uuid.pi_alloc(te_shape.size())
            _ = pi_encrypt(self.gpu_pub_key, fpn_store_id.get_store(), pen_store_id.get_store(), None)
            
            res_size = get_add_mul_size(self.shape, te_shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_add(self.gpu_pub_key, self.store_id.get_store(), pen_store_id.get_store(),
                             self.shape, te_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            del pen_store_id
            return PEN_store(res_store_id, res_shape, self.pub_key)
    
    @check_gpu_pub_key
    def __mul__(self, other):
        if isinstance(other, FPN_store):
            res_size = get_add_mul_size(self.shape, other.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)

            _, res_shape = pi_mul(self.gpu_pub_key, self.store_id.get_store(), other.store_id.get_store(),
                     self.shape, other.shape, res_store_id.get_store(), None, None)
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, TE_store):
            fp_store = other.encode(self.pub_key.n, self.pub_key.max_int)

            res_size = get_add_mul_size(self.shape, other.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)

            _, res_shape = pi_mul(self.gpu_pub_key, self.store_id.get_store(), fp_store.store_id.get_store(),
                             self.shape, fp_store.shape, res_store_id.get_store(), None, None)
            del fp_store
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, int) or isinstance(other, float):
            other_array = np.asarray([other])
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(other_array.shape)

            te_store_id = Store_uuid.te_alloc(fpn_shape.size())
            _ = te_p2c(other_array, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(fpn_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.pub_key.n, self.pub_key.max_int,
                                None, None, fpn_store_id.get_store(), None)

            res_size = get_add_mul_size(self.shape, fpn_shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_mul(self.gpu_pub_key, self.store_id.get_store(), fpn_store_id.get_store(),
                                self.shape, fpn_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, list):
            other_array = np.asarray(other)
            shape_tuple = other_array.shape
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(shape_tuple)

            te_store_id = Store_uuid.te_alloc(fpn_shape.size())
            _ = te_p2c(other_array, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(fpn_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.pub_key.n, self.pub_key.max_int,
                                None, None, fpn_store_id.get_store(), None)

            res_size = get_add_mul_size(self.shape, fpn_shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_mul(self.gpu_pub_key, self.store_id.get_store(), fpn_store_id.get_store(),
                                self.shape, fpn_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, np.ndarray):
            shape_tuple = other.shape
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(shape_tuple)

            te_store_id = Store_uuid.te_alloc(fpn_shape.size())
            _ = te_p2c(other, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(fpn_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.pub_key.n, self.pub_key.max_int,
                                None, None, fpn_store_id.get_store(), None)            
            
            res_size = get_add_mul_size(self.shape, fpn_shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_mul(self.gpu_pub_key, self.store_id.get_store(), fpn_store_id.get_store(),
                                self.shape, fpn_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            return PEN_store(res_store_id, res_shape, self.pub_key)

    @check_gpu_pub_key
    def __matmul__(self, other):
        if isinstance(other, PEN_store):
            raise NotImplementedError("Cannot do matmul between two encrypted number")
        if isinstance(other, FPN_store):
            res_size = get_matmul_rmatmul_size(self.shape, other.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)

            _, res_shape = pi_matmul(self.gpu_pub_key, self.store_id.get_store(), other.store_id.get_store(),
                                         self.shape, other.shape, res_store_id.get_store(), None, None)
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, TE_store):
            fp_store = other.encode(self.pub_key.n, self.pub_key.max_int)

            res_size = get_matmul_rmatmul_size(self.shape, fp_store.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)

            _, res_shape = pi_matmul(self.gpu_pub_key, self.store_id.get_store(), fp_store.store_id.get_store(),
                                         self.shape, fp_store.shape, res_store_id.get_store(), None, None)
            del fp_store
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, int) or isinstance(other, float):
            # print("single element")
            other_array = np.asarray([other])
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(other_array.shape)
            te_store_id = Store_uuid.te_alloc(fpn_shape.size())
            _ = te_p2c(other_array, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(fpn_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.pub_key.n, self.pub_key.max_int,
                                None, None, fpn_store_id.get_store(), None)

            res_size = get_matmul_rmatmul_size(self.shape, fpn_shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_matmul(self.gpu_pub_key, self.store_id.get_store(), fpn_store_id.get_store(),
                                         self.shape, fpn_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, list):
            other_array = np.asarray(other)
            shape_tuple = other_array.shape
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(shape_tuple)

            te_store_id = Store_uuid.te_alloc(fpn_shape.size())
            _ = te_p2c(other_array, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(fpn_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.pub_key.n, self.pub_key.max_int,
                                None, None, fpn_store_id.get_store(), None)

            res_size = get_matmul_rmatmul_size(self.shape, fpn_shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_matmul(self.gpu_pub_key, self.store_id.get_store(), fpn_store_id.get_store(),
                                 self.shape, fpn_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            return PEN_store(res_store_id, res_shape, self.pub_key)
        if isinstance(other, np.ndarray):
            shape_tuple = other.shape
            fpn_shape = TensorShapeStorage()
            fpn_shape = fpn_shape.from_tuple(shape_tuple)

            te_store_id = Store_uuid.te_alloc(fpn_shape.size())
            _ = te_p2c(other, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(fpn_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.pub_key.n, self.pub_key.max_int,
                                None, None, fpn_store_id.get_store(), None)            
            
            res_size = get_matmul_rmatmul_size(self.shape, fpn_shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_matmul(self.gpu_pub_key, self.store_id.get_store(), fpn_store_id.get_store(),
                                     self.shape, fpn_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            return PEN_store(res_store_id, res_shape, self.pub_key)
    
    @check_gpu_pub_key
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @check_gpu_pub_key
    def __radd__(self, other):
        return self.__add__(other)

    @check_gpu_pub_key
    def __sub__(self, other):
        return self + (other * -1)

    @check_gpu_pub_key
    def __rsub__(self, other):
        return other + (self * -1)
    
    @check_gpu_pub_key
    def __truediv__(self, scalar):
        return self.__mul__(1/scalar)
    
    @check_gpu_pub_key
    def transpose(self):
        res_size = self.shape.size()
        res_store_id = Store_uuid.pi_alloc(res_size)
        _, res_shape = pi_transpose(self.store_id.get_store(), self.shape, res_store_id.get_store(), None, None)
        return PEN_store(res_store_id, res_shape, self.pub_key)

    @check_gpu_pub_key
    def cat(self, other, axis=0):
        if self.pub_key != other.pub_key:
            raise RuntimeError("Cannot do cat between two encrypted number of different pub_key")
        temp_store = [self.store_id.get_store(), other.store_id.get_store()]
        temp_shape = [self.shape, other.shape]
        res_size = get_cat_size(temp_shape)
        res_store_id = Store_uuid.pi_alloc(res_size)
        _, temp_shape = pi_cat(temp_store, temp_shape, axis, res_store_id.get_store(), None)
        return PEN_store(res_store_id, temp_shape, self.pub_key)

    @check_gpu_pub_key
    def slice(self, start, end, axis=1):
        if len(self.get_shape()) == 1:
            if start < 0:
                start = self.get_shape()[1 - axis] + start
            if end < 0:
                end = self.get_shape()[1 - axis] + end
        elif len(self.get_shape()) == 2:
            if start < 0:
                start = self.get_shape()[axis] + start
            if end < 0:
                end = self.get_shape()[axis] + end
        if start <= end:
            RuntimeError("PEN_store slice start must be greater than end!")
        res_size = get_slice_size(self.shape, start, end, axis)
        res_store_id = Store_uuid.pi_alloc(res_size)
        _, slice_shape = pi_slice(self.store_id.get_store(), self.shape, start, end, axis,
                                 res_store_id.get_store(), None, None)
        return PEN_store(res_store_id, slice_shape, self.pub_key)

    # get one element from PEN_store
    @check_gpu_pub_key
    def get_element(self, index, axis=1):
        end = index + 1
        if index == -1:
            if len(self.get_shape()) == 1:
                end = self.get_shape()[1 - axis]
            elif len(self.get_shape()) == 2:
                end = self.get_shape()[axis]
        return self.slice(index, end, axis)

    @check_gpu_pub_key
    def sum(self, axis=None):
        res_size = get_sum_size(self.shape, axis)
        res_store_id = Store_uuid.pi_alloc(res_size)
        _, temp_shape = pi_sum(self.gpu_pub_key, self.store_id.get_store(), self.shape, axis,
                                 res_store_id.get_store(), None, None)
        return PEN_store(res_store_id, temp_shape, self.pub_key)
    
    @check_gpu_pub_key
    def reshape(self, new_shape):
        res_size = self.shape.size()
        res_store_id = Store_uuid.pi_alloc(res_size)

        _, self.shape = pi_reshape(self.store_id.get_store(), self.shape, new_shape, 
                                res_store_id.get_store(), None, None)
        self.store_id = res_store_id

    @check_gpu_pub_key
    def get_shape(self):
        return self.shape.to_tuple()
    
    @check_gpu_pub_key
    def get_size(self):
        return self.shape.size()

    @check_gpu_pub_key
    def set_shape(self, res_shape):
        self.shape.from_tuple(res_shape)
        return self
    

class FPN_store:
    def __init__(self, store_id: Store_uuid, shape: TensorShapeStorage):
        if isinstance(store_id, Store_uuid) is False:
            raise RuntimeError(f"Illegal store_id type : {type(store_id)}, params need type : {Store_uuid}")
        self.store_id = store_id
        self.shape = shape
    
    @staticmethod
    def init_from_arr(arr, n, max_int):
        if isinstance(arr, list):
            arr = np.asarray(list)
        if not isinstance(arr, np.ndarray):
            raise NotImplementedError("FPN_store init now only support list or ndarray")
        shape_tuple = arr.shape
        fpn_shape = TensorShapeStorage()
        fpn_shape = fpn_shape.from_tuple(shape_tuple)

        te_store_id = Store_uuid.te_alloc(fpn_shape.size())
        fp_store_id = Store_uuid.fp_alloc(fpn_shape.size())

        _ = te_p2c(arr, te_store_id.get_store())
        _ = fp_encode(te_store_id.get_store(), n, max_int, None, None,
                         fp_store_id.get_store(), None)
        return FPN_store(fp_store_id, fpn_shape)
    
    def encrypt(self, pub_key):
        c_pub_key = pi_p2c_pub_key(None, pub_key)
        gpu_pub_key = pi_h2d_pub_key(None, c_pub_key)

        pi_store_id = Store_uuid.pi_alloc(self.shape.size())

        _ = pi_encrypt(gpu_pub_key, self.store_id.get_store(), pi_store_id.get_store(), None)

        del c_pub_key
        del gpu_pub_key
        return PEN_store(pi_store_id, self.shape, pub_key)
    
    def decode(self):
        te_store_id = Store_uuid.te_alloc(self.shape.size())

        _ = fp_decode(self.store_id.get_store(), te_store_id.get_store(), None)
        res_shape = self.shape.to_tuple()
        res_array = te_c2p(te_store_id.get_store()).reshape(res_shape)
        return res_array
    
    @check_gpu_pub_key
    def __mul__(self, other):
        if isinstance(other, PEN_store):
            res_size = get_add_mul_size(other.shape, self.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_mul(other.gpu_pub_key, other.store_id.get_store(), self.store_id.get_store(),
                                     other.shape, self.shape, res_store_id.get_store(), None, None)
            return PEN_store(res_store_id, res_shape, other.pub_key)
        if isinstance(other, FPN_store):
            res_size = get_add_mul_size(self.shape, other.shape)
            res_store_id = Store_uuid.fp_alloc(res_size)
            _, res_shape = fp_mul(self.store_id.get_store(), other.store_id.get_store(),
                                     self.shape, other.shape, res_store_id.get_store(), None, None)
            return FPN_store(res_store_id, res_shape)
        if isinstance(other, TE_store):
            fp_store = other.encode(self.store_id.get_store().encode_n, self.store_id.get_store().max_int)

            res_size = get_add_mul_size(self.shape, fp_store.shape)
            res_store_id = Store_uuid.fp_alloc(res_size)
            _, res_shape = fp_mul(self.store_id.get_store(), fp_store.store_id.get_store(),
                                     self.shape, fp_store.shape, res_store_id.get_store(), None, None)
            return FPN_store(res_store_id, res_shape)
        if isinstance(other, int) or isinstance(other, float):
            other_array = np.asarray([other])
            fpn_shape = TensorShapeStorage()
            te_store_id = Store_uuid.te_alloc(fpn_shape.size())
            _ = te_p2c(other_array, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(fpn_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.store_id.get_store().encode_n, self.store_id.get_store().max_int,
                                None, None, fpn_store_id.get_store(), None)

            res_size = get_add_mul_size(self.shape, fpn_shape)
            res_store_id = Store_uuid.fp_alloc(res_size)
            _, res_shape = fp_mul(self.store_id.get_store(), fpn_store_id.get_store(),
                                 self.shape, fpn_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            return FPN_store(res_store_id, res_shape)
        if isinstance(other, list):
            other_array = np.asarray(other)
            shape_tuple = other_array.shape
            if len(shape_tuple)>1:
                fpn_shape = TensorShapeStorage(shape_tuple[0], shape_tuple[1])
            else:
                fpn_shape = TensorShapeStorage(shape_tuple[0])

            te_store_id = Store_uuid.te_alloc(fpn_shape.size())
            _ = te_p2c(other_array, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(fpn_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.store_id.get_store().encode_n, self.store_id.get_store().max_int,
                                None, None, fpn_store_id.get_store(), None)
            
            res_size = get_add_mul_size(self.shape, fpn_shape)
            res_store_id = Store_uuid.fp_alloc(res_size)
            _, res_shape = fp_mul(self.store_id.get_store(), fpn_store_id.get_store(),
                                     self.shape, fpn_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            return FPN_store(res_store_id, res_shape)
        if isinstance(other, np.ndarray):
            shape_tuple = other.shape
            if len(shape_tuple)>1:
                fpn_shape = TensorShapeStorage(shape_tuple[0], shape_tuple[1])
            else:
                fpn_shape = TensorShapeStorage(shape_tuple[0])

            te_store_id = Store_uuid.te_alloc(fpn_shape.size())
            _ = te_p2c(other, te_store_id.get_store())

            fpn_store_id = Store_uuid.fp_alloc(fpn_shape.size())
            _ = fp_encode(te_store_id.get_store(), self.store_id.get_store().encode_n, self.store_id.get_store().max_int,
                                None, None, fpn_store_id.get_store(), None)

            res_size = get_add_mul_size(self.shape, fpn_shape)
            res_store_id = Store_uuid.fp_alloc(res_size)
            _, res_shape = fp_mul(self.store_id.get_store(), fpn_store_id.get_store(),
                                     self.shape, fpn_shape, res_store_id.get_store(), None, None)
            del te_store_id
            del fpn_store_id
            return FPN_store(res_store_id, res_shape)
    
    @check_gpu_pub_key
    def __matmul__(self, other):
        '''
        We assume that other will always be a PEN_store
        because FPN_store @ FPN_store/TE_store is meaningless,
        do it in numpy may be faster
        '''
        if isinstance(other, PEN_store):
            res_size = get_matmul_rmatmul_size(self.shape, other.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = \
                pi_rmatmul(other.gpu_pub_key, self.store_id.get_store(), other.store_id.get_store(), self.shape, other.shape,
                           res_store_id.get_store(), None, None)
            return PEN_store(res_store_id, res_shape, other.pub_key)
        else:
            raise NotImplementedError("Currently only support FPN_store matmul PEN_store")
    
    def __len__(self):
        return self.store.vec_size
    
    def transpose(self):
        res_store_id = Store_uuid.fp_alloc(self.shape.size())
        _, res_shape = fp_transpose(self.store_id.get_store(), self.shape, res_store_id.get_store(), None, None)
        return FPN_store(res_store_id, res_shape)
    
    def __del__(self):
        del self.store_id
        del self.shape
        self.store = None
        self.shape = None

class TE_store:
    '''TensorStorage class for lists and ndarray'''
    def __init__(self, arr):
        '''
        Initialize a TE_store from a list or ndarray
        ------------------------
        arr: list or ndarray, other datatype not supported
        output:
            self.store: TensorStorage class, defined in gpu_engine.py
            self.shape: TensorShapeStorage class
        '''
        if isinstance(arr, list):
            arr = np.asarray(arr)
        if isinstance(arr, np.ndarray):
            shape_tuple = arr.shape
            self.shape = TensorShapeStorage()
            self.shape.from_tuple(shape_tuple)
            self.store_id = Store_uuid.te_alloc(self.shape.size())
            _ = te_p2c(arr, self.store_id.get_store())

        else:
            PermissionError("Currently only support tranpose list/ndarray into TE_store!")
    
    def __del__(self):
        del self.store_id
        self.store_id = None
        del self.shape
        self.shape = None
    
    def encode(self, n, max_int):
        res_store_id = Store_uuid.fp_alloc(self.shape.size())
        _ = fp_encode(self.store_id.get_store(), n, max_int,
                         None, None, res_store_id.get_store(), None)
        return FPN_store(res_store_id, self.shape)
    
    def get_arr(self):
        '''
        return an ndarray representation of self.store
        '''
        arr = te_c2p(self.store_id.get_store())
        shape = self.shape.to_tuple()
        return arr.reshape(shape)
    
    @check_gpu_pub_key
    def __add__(self, other):
        '''
        Only support that other is a PEN_store:
        '''
        if isinstance(other, PEN_store):
            temp_n = other.pub_key.n
            temp_max_int = other.pub_key.max_int
            fpn_store = self.encode(temp_n, temp_max_int)
            pen_store = fpn_store.encrypt(other.pub_key)

            res_size = get_add_mul_size(pen_store.shape, other.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_add(other.gpu_pub_key, pen_store.store_id.get_store(), other.store_id.get_store(),
                                    pen_store.shape, other.shape, res_store_id.get_store(), None, None)
            return PEN_store(res_store_id, res_shape, other.pub_key)
        else:
            raise NotImplementedError("Currently only support TE_store add PEN_store")
    
    @check_gpu_pub_key
    def __mul__(self, other):
        '''
        Only support that other is a PEN_store:
        '''
        if isinstance(other, PEN_store):
            temp_n = other.pub_key.n
            temp_max_int = other.pub_key.max_int
            fpn_store = self.encode(temp_n, temp_max_int)

            res_size = get_add_mul_size(other.shape, fpn_store.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = pi_mul(other.gpu_pub_key, other.store_id.get_store(), fpn_store.store_id.get_store(),
                                     other.shape, fpn_store.shape, res_store_id.get_store(), None, None)
            return PEN_store(res_store_id, res_shape, other.pub_key)
        elif isinstance(other, FPN_store):
            temp_n = other.store_id.get_store().encode_n
            temp_max_int = other.store_id.get_store().max_int
            fpn_store = self.encode(temp_n, temp_max_int)

            res_size = get_add_mul_size(fpn_store.shape, other.shape)
            res_store_id = Store_uuid.fp_alloc(res_size)
            _, res_shape = fp_mul(fpn_store.store_id.get_store(), other.store_id.get_store(),
                         fpn_store.shape, other.shape, res_store_id.get_store(), None, None)
            return FPN_store(res_store_id, res_shape)
        else:
            raise NotImplementedError("Currently only support TE_store matmul PEN_store/ FPN_store")

    @check_gpu_pub_key
    def __matmul__(self, other):
        '''
        We assume that other is a PEN_store:
        Because TE_store @ FPN/TE_store will be meaningless,
        and using numpy maybe faster
        '''
        if isinstance(other, PEN_store):
            temp_n = other.pub_key.n
            temp_max_int = other.pub_key.max_int  
            fpn_store = self.encode(temp_n, temp_max_int)

            res_size = get_matmul_rmatmul_size(fpn_store.shape, other.shape)
            res_store_id = Store_uuid.pi_alloc(res_size)
            _, res_shape = \
                pi_rmatmul(other.gpu_pub_key, fpn_store.store_id.get_store(), other.store_id.get_store(),
                             fpn_store.shape, other.shape, res_store_id.get_store(), None, None)
            return PEN_store(res_store_id, res_shape, other.pub_key)
        else:
            raise NotImplementedError("Currently only support TE_store matmul PEN_store")
    


def r_dot(arr: np.ndarray, pen: PEN_store):
    '''
    ndarray dot-mul PEN_store
    -----------------------------------
    '''
    pen_shape = pen.shape.to_tuple()
    arr_shape = arr.shape
    te_arr = TE_store(arr)
    print(arr_shape, pen_shape)
    if len(arr_shape) == 0 or len(pen_shape) == 0:
        res= te_arr * pen
        return res
    if len(arr_shape) == 1 and len(pen_shape) == 1:
        if pen_shape[0] == arr_shape[0]:
            te_arr = TE_store(arr)
            return (te_arr * pen).sum()
        else:
            raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
    if len(arr_shape) == 2 and len(pen_shape) == 2:
        te_arr = TE_store(arr)
        return te_arr @ pen
    
    if len(arr_shape) == 2 and len(pen_shape) == 1:
        if arr_shape[1] == pen_shape[0]:
            # vertically broadcast
            # become (1, pen_shape[0]), and vertically broadcast
            # result should be (pen_shape[0], ) * (arr_shape[0], pen_shape[0])
            # result is (arr_shape[0], pen_shape[0])
            # then we should horizontally sum and got the final as (arr_shape[0],)
            return (te_arr * pen).sum(1)
        else:
            raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
    
    if len(arr_shape) == 1 and len(pen_shape) == 2:
        if arr_shape[0] == pen_shape[0]:
            # return te_arr @ pen
            temp_arr = np.ascontiguousarray(arr.reshape(-1,1))
            te_temp = TE_store(temp_arr)
            return (te_temp * pen).sum(0)
            # temp_pen = pen.transpose()
            # res = te_arr * temp_pen
            # return res.sum(1)
        else:
            raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
            

def dot(pen: PEN_store, arr: np.ndarray):
    '''
    PEN_store dot-mul ndarray
    '''
    pen_shape = pen.shape.to_tuple()
    arr_shape = arr.shape
    te_arr = TE_store(arr)
    print(pen_shape, arr_shape)

    if len(pen_shape) == 0 or len(arr_shape) == 0:
        res = pen * te_arr
        return res
    
    if len(pen_shape) == 1 and len(arr_shape) == 1:
        if pen_shape[0] == arr_shape[0]:
            te_arr = TE_store(arr)
            return (pen * te_arr).sum()
        else:
            raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
    
    if len(pen_shape) ==2 and len(arr_shape) == 2:
        return pen @ te_arr
    
    if len(pen_shape) == 2 and len(arr_shape) == 1:
        if pen_shape[1] == arr_shape[0]:
            # horizontal broadcast
            # become (pen_shape[0], pen_shape[1]) and (arr_shape[])
            return (pen * te_arr).sum(1)
        else:
            raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")
    
    if len(pen_shape) == 1 and len(arr_shape) == 2:
        if pen_shape[0] == arr_shape[0]:
            # return te_arr @ pen
            # vertically broadcast
            # pen_shape should be (n,1) and broad
            '''old format, regarding transpose a large PEN_store in sum(0)'''
            # new_shape = TensorShapeStorage(pen_shape[0],1)
            # pen.reshape(new_shape)
            # return (pen * te_arr).sum(0)
            '''new format, regarding tranpose for an ndarray, not faster'''
            new_shape = TensorShapeStorage(1, pen_shape[0])
            pen.reshape(new_shape)
            
            temp_list = arr.transpose().tolist()
            temp_te = TE_store(temp_list)
            
            res = (pen * temp_te).sum(1)
            res_shape = TensorShapeStorage(arr_shape[1],)
            res.reshape(res_shape)
            return res
        else:
            raise PermissionError(f"Cannot perform dot with shape {pen_shape}, {arr_shape}")



