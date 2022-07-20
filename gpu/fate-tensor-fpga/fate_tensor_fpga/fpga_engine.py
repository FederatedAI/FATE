#
#  Copyright 2022 The FATE Authors. All Rights Reserved.
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
#

# from ctypes.util import find_library
import os
import random
import math
import numpy as np

from ctypes import cdll, c_buffer, cast
from ctypes import c_char_p, c_void_p
from ctypes import (
    c_int32,
    c_uint8,
    c_bool,
    c_uint32,
    c_double,
    c_int64,
    c_uint64,
    c_size_t,
)
from .secureprotol.fate_paillier import (
    PaillierPublicKey,
    PaillierPrivateKey,
    PaillierEncryptedNumber
)
from .secureprotol.fixedpoint import FixedPointNumber

from concurrent.futures import ProcessPoolExecutor as Executor

# define memory types
MEM_HOST = 1
MEM_DEVICE = 2
# the extended memory types, correspond with the device type defined below
MEM_FPGA_NUM_0 = 20
MEM_FPGA_NUM_1 = 21

# aliases defined by WeBank
PaillierPublicKeyStorage = PaillierPublicKey
PaillierPrivateKeyStorage = PaillierPrivateKey

'''##############import ctypes to implement py2c and c2py#################'''
'''############## load the .so library written in C     ##################'''

# note the .so library hasn't be named, use FPGA_1024 as an example
# we made 3 libraries, each one indicating a different .so library
# the number indicating the CIPHER_BIT length
FPGA_LIB = cdll.LoadLibrary(os.path.dirname(__file__) + "/FPGA_LIB.so")
# FPGA_4096 = cdll.LoadLibrary("FPGA_4096.so")

# set the CIPHER_BIT according to the library chosen.
CIPHER_BITS = 2048
PLAIN_BITS = 2048
BYTE_LEN = 8
CIPHER_BYTE = CIPHER_BITS // BYTE_LEN
PLAIN_BYTE = PLAIN_BITS // BYTE_LEN

# ### DEFINE THE BTYE_LENGTHS OF DATA TYPES ####
CHAR_BYTE = 1
U_INT32_BYTE = 4
DOUBLE_BYTE = 8
INT64_BYTE = 8

# DEFINE THE RETURN TYPE OF C_malloc####
FPGA_LIB.c_malloc.restype = c_void_p

# DEFINE TWO DIFFERENT TYPE OF DATA####
INT64_TYPE = 1  # datatype flag for int32 and int64
FLOAT_TYPE = 2  # datatype flag for float and double

# define base for Paillier encrypted numbers
PEN_BASE = 16
# as there's no BASE defined in Python PaillierEncryptedNumber,
# and we need this in CUDA, we define PEN_BASE as 16


# ############################################################################
# ######################Useful independent functions##########################
# ###################Reconstruct ndaray from C memory type####################
# ############################################################################
''' Device Initializer '''


def initialize_device():
    FPGA_LIB.init()
    # FPGA_LIB.print_example_banner()


'''reset FPGA functions'''


def reset_device(dev_num):
    FPGA_LIB.reset_device(c_uint8(dev_num))


def init_dev_reg(dev_num):
    FPGA_LIB.init_regs(c_uint8(dev_num))


def check_FPGA_status(dev_num):
    FPGA_LIB.status_check(c_uint8(dev_num))


def __get_C_fpn(fpn_space, size):
    '''
    copy FixedPointNumber (FPN) object out from C memory space,
    reform a ndarray, return it to upper python level
    --------------------
    Para:
    res_fpn_space: int, indicating the start address of a c_memory space
    size: int, the number of FPN in the C memory space
    Return:
    A ndarray, each element is a bigint
    '''
    res_fpn = []
    get_res = c_buffer(PLAIN_BYTE)
    for i in range(size):
        FPGA_LIB.bigint_get(
            cast(get_res, c_void_p),
            c_void_p(fpn_space),
            c_size_t(PLAIN_BITS),
            c_size_t(i),
        )
        res_fpn.append(int.from_bytes(get_res.raw, 'little'))
    return np.asarray(res_fpn)


def __get_C_pen(pen_space, index, size):
    '''
    copy PaillierEncryptedNumber(PEN) object out from C memory space,
    reform a ndarray, return it to upper python level
    ------------------
    Para:
    res_pen_space: int, indicating the start address of a continuous C memory space
    index: int, the offset from start address that we start to get PEN
    size: int, the number of PEN ought to get
    Return:
    A ndarray, each element is a bigint
    '''
    res_pen = []
    get_res = c_buffer(CIPHER_BYTE)
    for i in range(size):
        FPGA_LIB.bigint_get(
            cast(get_res, c_void_p),
            c_void_p(pen_space + index * CIPHER_BYTE),
            c_size_t(CIPHER_BITS),
            c_size_t(i),
        )
        res_pen.append(int.from_bytes(get_res.raw, 'little'))
    return np.asarray(res_pen)


bi_c2p = __get_C_pen


def __get_C_uint32(uint32_space, size):
    '''
    copy uint32 out from C memory space, form a ndarraay
    since numpy has a very good support for basic C numeric objects,
    A single memcpy will be sufficient
    ------------------------
    Para:
    res_uint32_space: int, indicating the start address of a continuous C memory space
    size: int, the number of uint32 ought to get
    '''
    uint32_list = [0 for _ in range(size)]
    int_list = (c_uint32 * size)(*uint32_list)
    FPGA_LIB.unsigned_get(
        int_list, c_void_p(uint32_space), c_size_t(size), c_bool(False)
    )
    uint32_list = [int_list[i] for i in range(size)]
    return np.asarray(uint32_list)


def __get_C_double(double_space, size):
    '''copy double out from C memory space, form a ndarray'''
    res_double_list = [0 for _ in range(size)]
    double_list = (c_double * size)(*res_double_list)
    FPGA_LIB.double_get(
        double_list, c_void_p(double_space), c_size_t(size), c_bool(False)
    )
    # TODO: convert all the data in one step, no loop
    res_double_list = [double_list[i] for i in range(size)]
    return np.asarray(res_double_list)


def __get_C_int64(int64_space, size):
    '''copy int64 out from C memory space, form a ndarray'''
    res_int64_list = [0 for _ in range(size)]
    int64_list = (c_int64 * size)(*res_int64_list)
    FPGA_LIB.int64_get(
        int64_list,
        c_void_p(int64_space),
        c_size_t(size),
        c_bool(False))
    # TODO: convert all the data in one step, no loop
    res_int64_list = [int64_list[i] for i in range(size)]
    return np.asarray(res_int64_list)


def __get_c_fpn_storage(fpn, base, exp, vec_size, n, max_int):
    '''
    Construct array of FixedPointNumber from given C memory spaces
    -------------------
    Para:
    fpn:  int, start address of a C memory space,
               inside which stores FPN's encodings(bigint, PLAIN_BITS long)
    base: int, start address of a C memory space,
               inside which stores FPN's base(uint32)
    exp:  int, start address of a C memory space,
               inside which stores FPN's exp(uint32)
    vec_size:   int, the number of bigint
    n, max_int: int, the key used to encode the original plaintext

    Return:
    A ndarray, each element is a FixedPointNumber
    '''
    res_fpn = __get_C_fpn(fpn, vec_size)
    # res_base = __get_C_uint32(base,size)
    res_exp = __get_C_uint32(exp, vec_size)
    result_FixedPointNumber = []
    for i in range(vec_size):
        result_FixedPointNumber.append(
            FixedPointNumber(res_fpn[i], float(res_exp[i]), n, max_int)
        )
    return result_FixedPointNumber


def __get_c_pen_storage_raw(pen, base, exp, vec_size, n):
    res_cipher = __get_C_pen(pen, 0, vec_size)
    res_base = __get_C_uint32(base, vec_size)
    res_exp = __get_C_uint32(exp, vec_size)
    return res_cipher, res_base, res_exp


def __get_c_pen_storage_mp(pen, base, exp, vec_size, n, thread_num=4):
    '''
    Use multi-process to accelerate __get_C_pen process.

    Since on Linux, python use fork to create sub-process,
    thus the C memory space is shared between father and child processes.
    And the whole process concerns no CUDA and cuda-context,
    even the return result is in python object form.
    So we can use multi-process for acceleration here safely
    ---------------------------------
    Para:
        thread_num: number of processes used in multi-processing
    Return:
        tuple, (ndarray, ndarray, ndarray)
    '''
    job_cnt = round(vec_size / thread_num)
    job_idx = 0
    job_idx_list, job_cnt_list = [0], []
    for i in range(thread_num - 1):
        job_idx += job_cnt
        job_idx_list.append(job_idx)
        job_cnt_list.append(job_cnt)
    job_cnt_list.append(vec_size - job_cnt * (thread_num - 1))
    # for __get_C_pen, use multiprocess to accelerate
    executor = Executor()
    futures = []
    for i in range(thread_num):
        futures.append(
            executor.submit(__get_C_pen, pen, job_idx_list[i], job_cnt_list[i])
        )
    res_list = [r.result() for r in futures]
    res_pen = []
    for res in res_list:
        res_pen.extend(res)
    # for uint32, no special demand for multiprocess
    res_base = __get_C_uint32(base, vec_size)
    res_exp = __get_C_uint32(exp, vec_size)
    return np.asarray(res_pen), res_base, res_exp


def __get_c_pen_storage(pen, base, exp, vec_size, n):
    '''
    Construct array of PaillierEncryptedNumber storage from given memory space
    ------------------
    pen:  int, start address of a C memory space,
               inside which stores PEN's encodings(bigint, CIPHER_BITS long)
    base: int, start address of a C memory space,
               inside which stores PEN's base(uint32)
    exp:  int, start address of a C memory space,
               inside which stores PEN's exp(uint32)
    vec_size:   int, the number of bigint
    n, max_int: int, the key used to encode the original plaintext

    Return:
    A ndarray, each element is a PaillierEncryptedNumber (PEN)
    '''
    res_cipher = __get_C_pen(pen, 0, vec_size)
    res_exp = __get_C_uint32(exp, vec_size)

    res_PaillierEncryptedNumber = []
    public_key = PaillierPublicKey(n)
    for i in range(vec_size):
        res_PaillierEncryptedNumber.append(
            PaillierEncryptedNumber(
                public_key, res_cipher[i], int(
                    round(
                        res_exp[i]))))

    return np.asarray(res_PaillierEncryptedNumber)


#######################################################################
# #########################DEFINITION OF CLASSES#######################
#######################################################################
'''#############  the definition of functions and classes #################'''

'''
    TensorStorage.data Containing the address pointing to a double type
    All the int32/int64 have been transformed to int64_t type
    All the float32/float64 have been transformed to double type
    We assume that TensorStorage has 2 types:
    1. data is ndarray, caculation can be performed directly by ndarray.
    2. data is C memory pointer, used for performing further encoding for
       the lower bound
'''


class TensorStorage:
    '''
    TensorStorage Class is used for store plaintexts.
    Currently support
    1. int32, int64 (all transformed to int64_t type)
    2. float32, float64 (all transformed to double type)

    Attributes:
        data: ndarray or int,
            1. ndarray means data is a python object
            2. int means data is a C memory object, the value of int is the C memory's
               start address
        vec_size: int, the number of data stored in current class
                       saved here since it may lost when data transfered to C memory
        mem_type: int, value is MEM_HOST or MEM_DEVICE, where the data is stored
                       default MEM_HOST
        data_type: int, value is INT_TYPE or FLOAT_TYPE, the data type of plaintext,
                        saved here since it may lost when data transfered to C memory
    '''

    def __init__(self, data, vec_size, mem_type: int, data_type: int):
        # numpy has some strange shallowcopies which causes incontinuous memory space
        # so add np.ascontinuousarray here to prevent potential errors
        self.data = np.ascontiguousarray(
            data) if isinstance(data, np.ndarray) else data
        self.vec_size = vec_size
        self.mem_type = mem_type
        self.data_type = data_type

    def __str__(self):
        return f"{self.__class__}:{self.data}"

    def __del__(self):
        te_free(self)


class BigIntStorage:
    '''
    Used for store bigint objects:

    Attributes:
        bigint_storage: int, the start address of the C memory storing bigint
        elem_size:      int, the size of the bigint,
                            useless since we unified into CIPHER_BITS
        vec_size:       int, the number of bigint stored in this class
        mem_type:       int, MEM_HOST or MEM_DEVICE, where data is stored, default MEM_HOST

    '''

    def __init__(self, data, vec_size, mem_type: int, elem_size: int):
        # 1:cpu/host  2:FPGA/device
        self.mem_type = mem_type
        # self.data = data
        self.bigint_storage = data
        self.elem_size = elem_size
        self.vec_size = vec_size

    def __len__(self):
        return len(self.data)

    def __del__(self):
        bi_free(self)


class FixedPointStorage:
    '''
    Contains the 3 pointers indicating start address of C memory,
    which can be handled directly by passing it to C functions in GPU_LIB
    ------------------
    Attributes:
        bigint_storage: int, start address of C memory,
                                in which stores the mantissa of a fpn array
        base_storage:   int, start address of C memory,
                                in which stores the base array of the fpn array
        exp_storage:    int, start address of C memory,
                                in which stores the exponent array of fpn array
        vec_size:       int, the number of data stored in current class
                                saved here since it may lost when data transfered to C memory
        mem_type:       int, value is MEM_HOST or MEM_DEVICE, where the data is stored
                                default MEM_HOST
        data_type:      int, value is INT_TYPE or FLOAT_TYPE, the data type of plaintext,
                                saved here since it may lost when data transfered to C memory
        encode_n, max_int: bigint, the para used for encode the plaintext
    '''

    def __init__(
            self,
            bigint_storage,
            base_storage,
            exp_storage,
            vec_size,
            n,
            max_int,
            mem_type: int,
            data_type,
    ):
        # 1:cpu/host  2:FPGA/device
        self.mem_type = mem_type
        '''Actual data and length for fpn'''
        self.bigint_storage = bigint_storage
        self.base_storage = base_storage
        self.exp_storage = exp_storage
        self.vec_size = vec_size
        '''TensorStorage needed paras'''
        self.data_type = data_type
        '''En/Decode needed paras '''
        # these 2 are just python int, not BigintStorage nor C_types
        self.encode_n = n
        self.max_int = max_int

    def __len__(self):
        return self.vec_size

    def __del__(self):
        fp_free(self)


class PaillierEncryptedStorage:
    '''
    Contains the 3 pointers indicating start address of C memory,
    which can be handled directly by passing it to C functions in GPU_LIB
    --------------------
    Attributes:
        pen_storage:    int, start address of C memory,
                                in which stores the mantissa of the pen array
        base_storage:   int, start address of C memory,
                                in which stores the bases of the pen array
        exp_storage:    int, start address of C memory,
                                in which stores the exponents of the pen array
        vec_size:       int, the number of data stored in current class
                                saved here since it may lost when data transfered to C memory
        mem_type:       int, value is MEM_HOST or MEM_DEVICE, where the data is stored
                                default MEM_HOST
        data_type:      int, value is INT_TYPE or FLOAT_TYPE, the data type of plaintext,
                                saved here since it may lost when data transfered to C memory
        encode_n, max_int: bigint, the para used for encode the plaintext
    '''

    def __init__(
            self,
            pen_storage,
            base_storage,
            exp_storage,
            vec_size,
            mem_type: int,
            data_type,
            fpn_encode_n,
            fpn_encode_max_int,
    ):
        # 1:cpu/host  2:FPGA/device
        self.mem_type = mem_type
        '''Actual data and length for pen'''
        self.pen_storage = pen_storage
        self.base_storage = base_storage
        self.exp_storage = exp_storage
        self.vec_size = vec_size
        '''TensorStorage needed paras'''
        self.data_type = data_type
        '''En/Decode needed paras '''
        self.encode_n = fpn_encode_n
        self.encode_max_int = fpn_encode_max_int
        '''Pub_key paras'''

    def __len__(self):
        return self.vec_size

    def __del__(self):
        pi_free(self)


class TensorShapeStorage:
    '''
    Used for store the shape, currently support 2 dim
    The behavior is identical to numpy
    -------------------
    Attributes:
        dim1: the 1st dim, aka the row
        dim2: the 2nd dim, aka the col
    '''

    def __init__(self, dim1=None, dim2=None):
        if dim1 is not None and not isinstance(dim1, int):
            raise TypeError("invalid dimension")
        if dim2 is not None and not isinstance(dim2, int):
            raise TypeError("invalid dimension")
        self.dim1 = dim1
        self.dim2 = dim2

    def size(self):
        dim1 = 1 if self.dim1 is None else self.dim1
        dim2 = 1 if self.dim2 is None else self.dim2
        return dim1 * dim2

    def __getitem__(self, item):
        return self.to_tuple().__getitem__(item)

    def __len__(self):
        return len(self.to_tuple())

    def to_tuple(self):
        if self.dim1 is None:
            return ()
        else:
            if self.dim2 is None:
                return (self.dim1,)
            else:
                return (self.dim1, self.dim2)

    def from_tuple(self, v):
        if len(v) == 1:
            self.dim1 = v[0]
            self.dim2 = None
        elif len(v) == 2:
            self.dim1 = v[0]
            self.dim2 = v[1]
        else:
            self.dim1 = None
            self.dim2 = None
        return self

    def transpose(self):
        return TensorShapeStorage(self.dim2, self.dim1)

    def matmul(self, other):
        return TensorShapeStorage(self.dim1, other.dim2)


class PubKeyStorage:
    '''
    Used for store PaillierPublicKey info as C-accpetable data type
    -------------
    Attributes:
       n,g, nsquare, max_int:
            c_char_p, actual value is bytes
            all identical to PaillierPublicKey, which is defined in fate_script
    '''

    def __init__(self, n, g, nsquare, max_int):
        self.n = c_char_p(n.to_bytes(CIPHER_BYTE, 'little'))
        self.g = c_char_p(g.to_bytes(CIPHER_BYTE, 'little'))
        self.nsquare = c_char_p(nsquare.to_bytes(CIPHER_BYTE, 'little'))
        self.max_int = c_char_p(max_int.to_bytes(CIPHER_BYTE, 'little'))


class PrivKeyStorage:
    '''
    Used for store PaillierPrivateKey info as C-acceptable data type
    ------------
    Attributes are all identical to PaillierPrivateKey, defined in fate_script
    '''

    def __init__(self, p, q, psquare, qsquare, q_inverse, hp, hq):
        self.p = c_char_p(p.to_bytes(CIPHER_BYTE, 'little'))
        self.q = c_char_p(q.to_bytes(CIPHER_BYTE, 'little'))
        self.psquare = c_char_p(psquare.to_bytes(CIPHER_BYTE, 'little'))
        self.qsquare = c_char_p(qsquare.to_bytes(CIPHER_BYTE, 'little'))
        self.q_inverse = c_char_p(q_inverse.to_bytes(CIPHER_BYTE, 'little'))
        self.hp = c_char_p(hp.to_bytes(CIPHER_BYTE, 'little'))
        self.hq = c_char_p(hq.to_bytes(CIPHER_BYTE, 'little'))


##########################################################################
# ###############FUNCTION DEFINITION START################################
##########################################################################


def te_p2c_shape(shape, res):
    '''
    Change a 2-elem tuple into a TensorShapeStorage object
    -------------
    Para:
        shape:   tuple, with no more than 2 elements
        res:     return value
    Return:
        res,     TensorShapeStorage
    '''
    if res is None:
        res = TensorShapeStorage()
    res.from_tuple(shape)
    return res


def te_c2p_shape(shape):
    '''
    recover the shape_tuple from TensorShapeStorage
    --------------
    Para:   shape:   TensorShapeStorage
    Return: tuple
    '''
    return shape.to_tuple()


def te_free(tes):
    '''
    free the c memory space in a TensorStorage class
    ------------
    Para:   tes: TensorStorage
    Return: None
    '''
    if isinstance(tes.data, int):
        # means that it is a C memory pointer
        FPGA_LIB.c_free(c_void_p(tes.data))
        tes.data = None
    # otherwise, tes.data is a python datatype(list or ndarray)


def te_p2c(data, res=None):
    '''
    transmit the data storage form from Python to C
    we assume data's structure has already been preserved by the upper layer
    using the TensorShapeStorage class
    ------------------
    Args:
        data, list or ndarray, the original data array
    Return:
        TensorStorage, and data is a C pointer
    '''
    # flatten the current ndarray for get the actual vec_size
    if isinstance(data, list):
        data = np.asarray(data)
    if not isinstance(data, np.ndarray):
        raise TypeError("Unsupported Data Structure")
    vec_size = data.size

    # malloc c memory space
    if res is None:
        storage_pointer = FPGA_LIB.c_malloc(c_size_t(vec_size * DOUBLE_BYTE))
    else:
        storage_pointer = res.data

    # switch the differnt data types
    if data.dtype == 'int32':
        new_data = data.astype(np.int64)
        data_pointer = new_data.ctypes.data_as(c_void_p)
        data_type = INT64_TYPE
        FPGA_LIB.int64_set(
            c_void_p(storage_pointer),
            data_pointer,
            c_size_t(vec_size))
    elif data.dtype == 'int64':
        data_pointer = data.ctypes.data_as(c_void_p)
        data_type = INT64_TYPE
        FPGA_LIB.int64_set(
            c_void_p(storage_pointer),
            data_pointer,
            c_size_t(vec_size))
    elif data.dtype == 'float32':
        new_data = data.astype(np.float64)
        data_pointer = new_data.ctypes.data_as(c_void_p)
        data_type = FLOAT_TYPE
        FPGA_LIB.float64_set(
            c_void_p(storage_pointer), data_pointer, c_size_t(vec_size)
        )
    elif data.dtype == 'float64':
        data_pointer = data.ctypes.data_as(c_void_p)
        data_type = FLOAT_TYPE
        FPGA_LIB.float64_set(
            c_void_p(storage_pointer), data_pointer, c_size_t(vec_size)
        )
    else:
        raise PermissionError("Invalid Data Type")
    return _te_init_store(
        res,
        storage_pointer,
        vec_size,
        MEM_FPGA_NUM_0,
        data_type)


def te_c2p(store):
    '''
    transmit TensorShapeStorage form from C to Python
    due to different data type, the return array may diff
    -----------
    Para:
        store: TensorShapeStorage, the storage waited to be changed
    Return:
        res_array: np.ndarray, the returned ndarray to Python
    '''
    if store.data_type == FLOAT_TYPE:
        temp_array = __get_C_double(store.data, store.vec_size)
        res_array = temp_array.astype(np.float64)
        return res_array
    elif store.data_type == INT64_TYPE:
        temp_array = __get_C_int64(store.data, store.vec_size)
        res_array = temp_array.astype(np.int64)
        return res_array
    else:
        raise PermissionError("Invalid Data Type")


def te_c2bytes(data, res):
    '''
    transmit TensorShapeStorage form from C to bytes stream.
    Used for communication between sites, since C memory is not shared
    --------------------
    Para:
        data: TensorShapeStorage, data is a C memory ptr
        res:  the return bytes string
    Return:
        res:  bytes
    '''
    data_type = data.data_type
    bytes_result = c_buffer(DOUBLE_BYTE * data.vec_size + U_INT32_BYTE)
    # first 4 bytes: contains the data_type info
    # remain bytes:  contains the data
    FPGA_LIB.get_bytes(
        cast(bytes_result, c_void_p),
        c_char_p(data_type.to_bytes(U_INT32_BYTE, 'little')),
        c_void_p(data.data),
        c_size_t(data.vec_size),
    )
    return bytes_result.raw


def fp_c2bytes(store, res):
    '''
    transmit FixedPointStorage form to bytes stream;
    Used for communication between sites, since C memory is not shared
    Other info besides the C memory, including data_type, mem_type,
    are also included
    -----------------
    Para:
        store: FixedPointStorage
        res:   the return bytes string
    Return:
        res:   bytes
    '''
    # uint32
    data_type = store.data_type
    mem_type = store.mem_type
    # bigint
    encode_n = store.encode_n
    max_int = store.max_int
    # actual storage
    bytes_result = c_buffer(
        (PLAIN_BYTE + U_INT32_BYTE * 2) * store.vec_size + U_INT32_BYTE * 2 + PLAIN_BYTE * 2
    )
    FPGA_LIB.fp_get_bytes(
        cast(bytes_result, c_void_p),
        c_char_p(data_type.to_bytes(U_INT32_BYTE, 'little')),
        c_char_p(mem_type.to_bytes(U_INT32_BYTE, 'little')),
        c_char_p(encode_n.to_bytes(PLAIN_BYTE, 'little')),
        c_char_p(max_int.to_bytes(PLAIN_BYTE, 'little')),
        c_void_p(store.bigint_storage),
        c_void_p(store.base_storage),
        c_void_p(store.exp_storage),
        c_size_t(store.vec_size),
    )
    return bytes_result.raw


def pi_c2bytes(store, res):
    '''
    transmit PaillierEncryptedNumber form to bytes stream
    Used for communication between sites, since C memory is not shared
    ----------------
    Para:
        store: PaillierEncryptedStorage
        res:   the return bytes string
    Return:
        res:   bytes
    '''
    # uint32
    data_type = store.data_type
    mem_type = store.mem_type
    # bigint
    encode_n = store.encode_n
    max_int = store.encode_max_int
    # actual storage
    bytes_result = c_buffer(
        (CIPHER_BYTE + U_INT32_BYTE * 2) * store.vec_size + U_INT32_BYTE * 2 + CIPHER_BYTE * 2
    )
    FPGA_LIB.pi_get_bytes(
        cast(bytes_result, c_void_p),
        c_char_p(data_type.to_bytes(U_INT32_BYTE, 'little')),
        c_char_p(mem_type.to_bytes(U_INT32_BYTE, 'little')),
        c_char_p(encode_n.to_bytes(CIPHER_BYTE, 'little')),
        c_char_p(max_int.to_bytes(CIPHER_BYTE, 'little')),
        c_void_p(store.pen_storage),
        c_void_p(store.base_storage),
        c_void_p(store.exp_storage),
        c_size_t(store.vec_size),
    )
    return bytes_result.raw


def _te_init_store(store, data, vec_size, mem_type, data_type):
    '''
    initialize tensor storage,
    -----------
    Para:
        store: the return value, TensorStorage, default None
        Other paras' definition are equals to the one in TensorStorage
    Return:
        TensorShapeStorage
    '''
    if store is None:
        store = TensorStorage(data, vec_size, mem_type, data_type)
    else:
        store.data = data
        store.vec_size = vec_size
        if mem_type is not None:
            store.mem_type = mem_type
        store.data_type = data_type
    return store


def te_bytes2c(data, res):
    '''
    Restore TensorStorage from bytes buffer,
    TensorStorage.data is a ptr pointing to the restored C memory space.
    -------------
    Para:
        data: the bytes string
        res:  the return value, TensorStorage
    Return:
        res:  TensorStorage, the restored struct from para.data
    '''
    data_type_result = c_buffer(U_INT32_BYTE)
    len_data = len(data) - U_INT32_BYTE
    if res is None:
        storage_pointer = FPGA_LIB.c_malloc(c_size_t(len_data))
    else:
        storage_pointer = res.data
    FPGA_LIB.from_bytes_get_c(
        cast(data_type_result, c_void_p),
        c_void_p(storage_pointer),
        c_char_p(data),
        c_size_t(len_data),
    )
    data_type = int.from_bytes(data_type_result, 'little')
    # TODO: change according to different data_types,
    # now just use DOUBLE BYTE because we have only INT64 and DOUBLE,
    # all of them are 8 bytes(Equal to DOUBLE_BYTE)
    vec_size = len_data // DOUBLE_BYTE
    return _te_init_store(
        res,
        storage_pointer,
        vec_size,
        MEM_FPGA_NUM_0,
        data_type)


def fp_bytes2c(data, res):
    '''
    Restore FixedPointStorage from bytes buffer.
    ---------------
    Para:
        data: the bytes string
        res:  the return value, FixedPointStorage
    Return:
        res:  FixedPointStorage, the restored struct from para.data.
    '''
    # caculate vec_size
    vec_size = (len(data) - 2 * (U_INT32_BYTE + PLAIN_BYTE)) // (U_INT32_BYTE * 2 + PLAIN_BYTE)
    # uint32
    data_type = c_buffer(U_INT32_BYTE)
    mem_type = c_buffer(U_INT32_BYTE)
    # bigint
    encode_n = c_buffer(PLAIN_BYTE)
    max_int = c_buffer(PLAIN_BYTE)
    # storage
    fpn = FPGA_LIB.c_malloc(c_size_t(PLAIN_BYTE * vec_size))
    base = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * vec_size))
    exp = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * vec_size))

    FPGA_LIB.fp_from_bytes_get_c(
        cast(data_type, c_void_p),
        cast(mem_type, c_void_p),
        cast(encode_n, c_void_p),
        cast(max_int, c_void_p),
        cast(fpn, c_void_p),
        cast(base, c_void_p),
        cast(exp, c_void_p),
        c_char_p(data),
        c_size_t(vec_size),
    )
    return _fp_init_store(
        res,
        fpn,
        base,
        exp,
        vec_size,
        int.from_bytes(encode_n, 'little'),
        int.from_bytes(max_int, 'little'),
        int.from_bytes(mem_type, 'little'),
        int.from_bytes(data_type, 'little'),
    )


def pi_bytes2c(data, res):
    '''
    Restored PaillierEncryptedStorage from bytes buffer
    --------------
    Para:
        data: the bytes string
        res:  the return value, PaillierEncryptedStorage
    Return:
        res:  PaillierEncryptedStorage, the restored struct from para.data
    '''
    # caculate vec_size
    vec_size = (len(data) - 2 * (U_INT32_BYTE + CIPHER_BYTE)) // (U_INT32_BYTE * 2 + CIPHER_BYTE)
    # uint32
    data_type = c_buffer(U_INT32_BYTE)
    mem_type = c_buffer(U_INT32_BYTE)
    # bigint
    encode_n = c_buffer(CIPHER_BYTE)
    max_int = c_buffer(CIPHER_BYTE)
    # storage
    pen = FPGA_LIB.c_malloc(c_size_t(CIPHER_BYTE * vec_size))
    base = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * vec_size))
    exp = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * vec_size))

    FPGA_LIB.pi_from_bytes_get_c(
        cast(data_type, c_void_p),
        cast(mem_type, c_void_p),
        cast(encode_n, c_void_p),
        cast(max_int, c_void_p),
        cast(pen, c_void_p),
        cast(base, c_void_p),
        cast(exp, c_void_p),
        c_char_p(data),
        c_size_t(vec_size),
    )
    return _pi_init_store(
        res,
        pen,
        base,
        exp,
        vec_size,
        int.from_bytes(mem_type, 'little'),
        int.from_bytes(data_type, 'little'),
        int.from_bytes(encode_n, 'little'),
        int.from_bytes(max_int, 'little'),
    )


def _te_init_shape(shape_store, shape_tuple):
    '''
    Init TensorShapeStorage
    ----------
    Para:
        shape_store: TensorShapeStorage or None, return value, default None
        shape_tuple: tuple, at most 2 dim, source data of TensorShapeStorage
    Return:
        TensorShapeStorage
    '''
    if shape_store is None:
        shape_store = TensorShapeStorage()
    shape_store.from_tuple(shape_tuple)
    return shape_store


def _te_init_ss(
        res_store, res_data, vec_size, res_shape, shape_tuple, mem_type, data_type
):
    '''
    Init TensorStorage and TensorShapeStorage at the same time
    ------------
    Para:
        res_store: The return value, TensorStorage, default None
        res_data:  int or ndarray
        vec_size:  int
        res_shape: The return value, TensorShapeStorage, default None
        shape_tuple, tuple, at most 2 dim
        mem_type:  int
        data_type: int
    Return:
        tuple, (TensorStorage, TensorShapeStorage)
    '''
    return _te_init_store(
        res_store, res_data, vec_size, mem_type, data_type
    ), _te_init_shape(res_shape, shape_tuple)


'''''' '''
The following calculators are done on TensorStorage
Definition and output are the same with numpy
TensorStorage.data should all be ndarray datatype in order to support numpy

NOT USED IN OUR FATE IMPLEMENTATION,
but Webank's implementation seems to have used them
''' ''''''


def te_slice(store, shape, start, stop, axis, res_store, res_shape, stream):
    if axis == 1:
        res_data = store.data[:, start:stop]
    elif axis == 0:
        res_data = store.data[start:stop]
    else:
        raise NotImplementedError()
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        store.mem_type,
        store.data_type,
    )


def te_cat(stores, axis, res_store, res_shape):
    if axis == 0:
        res_data = np.vstack([x.data for x in stores])
    elif axis == 1:
        res_data = np.hstack([x.data for x in stores])
    else:
        raise NotImplementedError()
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        stores[0].mem_type,
        stores[0].data_type,
    )


# TODO: precise data_type


def te_pow(left_store, right, left_shape, res_store, res_shape, stream):
    res_data = left_store.data ** right
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        left_store.mem_type,
        left_store.data_type,
    )


# TODO: precise data_type


def te_add(
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store,
        res_shape,
        stream):
    res_data = left_store.data + right_store.data
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        left_store.mem_type,
        left_store.data_type,
    )


# TODO: precise data_type


def te_mul(
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store,
        res_shape,
        stream):
    res_data = left_store.data * right_store.data
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        left_store.mem_type,
        left_store.data_type,
    )


def te_truediv(
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store,
        res_shape,
        stream):
    res_data = left_store.data / right_store.data
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        left_store.mem_type,
        FLOAT_TYPE,
    )


def te_floordiv(
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store,
        res_shape,
        stream):
    res_data = left_store.data // right_store.data
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        left_store.mem_type,
        INT64_TYPE,
    )


# TODO: precise data_type


def te_sub(
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store,
        res_shape,
        stream):
    res_data = left_store.data - right_store.data
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        left_store.mem_type,
        left_store.data_type,
    )


# TODO: precise data_type


def te_matmul(
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store,
        res_shape,
        stream):
    res_data = left_store.data @ right_store.data
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        left_store.mem_type,
        left_store.data_type,
    )


def te_abs(left_store, left_shape, res_store, res_shape, stream):
    return _te_init_ss(
        res_store,
        abs(left_store.data),
        left_store.vec_size,
        res_shape,
        left_shape.to_tuple(),
        left_store.mem_type,
        left_store.data_type,
    )


def te_neg(left_store, left_shape, res_store, res_shape, stream):
    return _te_init_ss(
        res_store,
        -left_store.data,
        left_store.vec_size,
        res_shape,
        left_shape.to_tuple(),
        left_store.mem_type,
        left_store.data_type,
    )


def te_transpose(left_store, left_shape, res_store, res_shape, stream):
    res_data = left_store.data.transpose()
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        left_store.mem_type,
        left_store.data_type,
    )


def te_sum(left_store, left_shape, axis, res_store, res_shape, stream):
    res_data = left_store.data.sum(axis=axis)
    return _te_init_ss(
        res_store,
        res_data,
        res_data.size,
        res_shape,
        res_data.shape,
        left_store.mem_type,
        left_store.data_type,
    )


def te_reshape(store, shape, new_shape, res_store, res_shape, stream):
    return _te_init_ss(
        res_store,
        store.data.reshape(new_shape),
        store.vec_size,
        res_shape,
        new_shape.to_tuple(),
        store.mem_type,
        store.data_type,
    )


def te_exp(store, shape, res_store, res_shape, stream):
    return _te_init_ss(
        res_store,
        np.exp(store.data),
        store.vec_size,
        res_shape,
        shape.to_tuple(),
        store.mem_type,
        FLOAT_TYPE,
    )


def te_hstack(
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store,
        res_shape,
        stream):
    _store, _shape = te_cat([left_store, right_store], 1, res_store, res_shape)
    # avoid naming collision
    return _te_init_ss(
        res_store,
        _store.data,
        _store.vec_size,
        _shape,
        _shape.to_tuple(),
        left_store.mem_type,
        left_store.data_type,
    )


def te_c2p_first(store):
    '''
    Get the first element in the C data storage of TensorStorage
    ---------------
    Para:
        store: TensorStorage, store.data must be a pointer to C memory
    Return:
        int or double, the first element in the C memory
    '''
    if store.data_type == FLOAT_TYPE:
        temp_array = __get_C_double(store.data, store.vec_size)
        res_array = temp_array.astype(np.float64)
        return res_array[0]
    elif store.data_type == INT64_TYPE:
        temp_array = __get_C_int64(store.data, store.vec_size)
        res_array = temp_array.astype(np.int64)
        return res_array[0]
    else:
        raise PermissionError("Invalid Data Type")


def bi_alloc(res, vec_size, elem_size, mem_type):
    return _bi_init_store(
        res,
        FPGA_LIB.c_malloc(c_size_t(vec_size * elem_size)),
        vec_size,
        elem_size,
        mem_type,
    )


'''################malloc a space with size elements############### '''
'''
    function: allocate space and form a new PaillierEncryptedStorage Class
    res:    spilted to 3 different parts, indicating the 3 parts
            that are needed for the PaillierEncrytedStorage
    size:   is the number of elements that need to be alloced
    return: A PaillierEncryptedStorage class, wrapping res as a class
'''


def pi_alloc(res, size, mem_type):
    res_pen = FPGA_LIB.c_malloc(c_size_t(size * CIPHER_BYTE))
    res_base = FPGA_LIB.c_malloc(c_size_t(size * U_INT32_BYTE))
    res_exp = FPGA_LIB.c_malloc(c_size_t(size * U_INT32_BYTE))
    # data_type, encode_n and encode_max_int all set to 0
    return _pi_init_store(
        res,
        res_pen,
        res_base,
        res_exp,
        size,
        mem_type,
        0,
        0,
        0)


def fp_alloc(res, size, mem_type):
    res_fpn = FPGA_LIB.c_malloc(c_size_t(size * PLAIN_BYTE))
    res_base = FPGA_LIB.c_malloc(c_size_t(size * U_INT32_BYTE))
    res_exp = FPGA_LIB.c_malloc(c_size_t(size * U_INT32_BYTE))
    return _fp_init_store(
        res,
        res_fpn,
        res_base,
        res_exp,
        size,
        0,
        0,
        mem_type,
        0)


def te_alloc(res, size, mem_type):
    data = FPGA_LIB.c_malloc(c_size_t(size * DOUBLE_BYTE))
    return _te_init_store(res, data, size, mem_type, 0)


def pi_free(ptr):
    '''
    The delete function of PaillierEncryptedStorage,
    Due to different mem_type, the delete method may change
    --------------
    Para:
        ptr: PaillierEncryptedStorage
    '''
    FPGA_LIB.c_free(c_void_p(ptr.pen_storage))
    FPGA_LIB.c_free(c_void_p(ptr.base_storage))
    FPGA_LIB.c_free(c_void_p(ptr.exp_storage))
    ptr.pen_storage, ptr.base_storage, ptr.exp_storage = None, None, None


# Host2Device and Device2Host calculators are not implemented on FPGA currently
def pi_d2h(target, src, size, stream):
    return src


def pi_h2d(target, src, size, stream):
    return src


def pi_h2d_pub_key(target, src):
    return src


def pi_h2d_priv_key(target, src):
    return src


def pi_p2c_pub_key(target, src):
    '''
    Transfer Python form PaillierPublicKey to C form PubKeyStorage,
    the latter can be used for C/FPGA computing
    '''
    target = PubKeyStorage(src.n, src.g, src.nsquare, src.max_int)
    return target


def pi_p2c_priv_key(target, src):
    '''
    Transfer Python form PaillierPrivateKey to C form PrivKeyStorage
    the latter one can be used for C/FPGA computing
    '''
    target = PrivKeyStorage(
        src.p, src.q, src.psquare, src.qsquare, src.q_inverse, src.hp, src.hq
    )
    return target


# ###########PaillierEncrypted STORAGE INITIALIZE#################
def _pi_init_store(
        res_store,
        pen_storage,
        base_storage,
        exp_storage,
        vec_size,
        mem_type,
        data_type,
        encode_n,
        encode_max_int,
):
    '''
    init a new PaillierEncryptedStorage
    ---------------
    Para:
        res_store, PaillierEncryptedStorage or None, return value, default None
        Else paras are identical to the ones described in PaillierEncryptedStorage
    '''
    if res_store is None:
        res_store = PaillierEncryptedStorage(
            pen_storage,
            base_storage,
            exp_storage,
            vec_size,
            mem_type,
            data_type,
            encode_n,
            encode_max_int,
        )
    else:
        res_store.pen_storage = pen_storage
        res_store.base_storage = base_storage
        res_store.exp_storage = exp_storage
        res_store.vec_size = vec_size
        res_store.mem_type = mem_type
        '''TensorStorage needed'''
        res_store.data_type = data_type
        '''FixedPointNumber Needed'''
        res_store.encode_n = encode_n
        res_store.encode_max_int = encode_max_int
    return res_store


_pi_init_shape = _te_init_shape


def _pi_init_ss(
        res_store,
        pen_storage,
        base_storage,
        exp_storage,
        vec_size,
        res_shape,
        res_shape_tuple,
        mem_type,
        data_type,
        encode_n,
        encode_max_int,
):
    '''
    init PaillierEncryptedStorage and corresponding TensorShapeStorage at same time
    Paras are identical to _pi_init_store & _te_init_shape
    '''
    return _pi_init_store(
        res_store,
        pen_storage,
        base_storage,
        exp_storage,
        vec_size,
        mem_type,
        data_type,
        encode_n,
        encode_max_int,
    ), _pi_init_shape(res_shape, res_shape_tuple)


''' transfor PEN tensor from Python memory to C memory '''


def pi_p2c(target, src, data_type=FLOAT_TYPE):
    '''
    Transform list of PaillierEncryptedNumber to
    C-memory style PaillierEncryptedStorage
    --------------------
    Para:
        target:     PaillierEncryptedStorage, return value
        src:        List or ndarray, each element is a PaillierEncryptedNumber
        data_type:  int, src's original datatype, default double
    '''
    if isinstance(src, list):
        src = np.array(src)
    if not isinstance(src, np.ndarray):
        raise TypeError("Unsupported Data Structure")
    src = src.flatten()
    vec_size = src.size
    # malloc the space for the type
    if target is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(vec_size * CIPHER_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    else:
        res_pen = target.pen_storage
        res_base = target.base_storage
        res_exp = target.exp_storage
    # get the two encoding parameters
    n = src[0].public_key.n
    max_int = src[0].public_key.max_int
    base_temp = []
    exp_temp = []
    # Due to the special condition that big_ints in ndaray are not continuously stored
    # they are actually oject type rather than int type
    # Actually ndarray stores its reference/pointer continuously rather than real value
    # So we should use a for loop to handle each bigint and memcpy it
    for i in range(vec_size):
        src_number = src[i].ciphertext(False).to_bytes(CIPHER_BYTE, 'little')
        FPGA_LIB.bigint_set(
            c_char_p(res_pen),
            c_char_p(src_number),
            c_size_t(CIPHER_BITS),
            c_size_t(i))
        base_temp.append(PEN_BASE)
        exp_temp.append(src[i].exponent)
    # base and exp are deepcopyed in order to prevent potential double free
    # here
    base_arr_ptr = np.asarray(base_temp).ctypes.data_as(c_void_p)
    exp_arr_ptr = np.asarray(exp_temp).ctypes.data_as(c_void_p)
    FPGA_LIB.unsigned_set(c_void_p(res_base), base_arr_ptr, c_size_t(vec_size))
    FPGA_LIB.unsigned_set(c_void_p(res_exp), exp_arr_ptr, c_size_t(vec_size))
    return _pi_init_store(
        target,
        res_pen,
        res_base,
        res_exp,
        vec_size,
        MEM_FPGA_NUM_0,
        data_type,
        n,
        max_int,
    )


def _bi_init_store(res_store, data, count, elem_size, mem_type):
    '''init a new BigIntStorage object'''
    if res_store is None:
        res_store = BigIntStorage(data, count, mem_type, elem_size)
    else:
        res_store.bigint_storage = data
        res_store.vec_size = count
        res_store.mem_type = mem_type
        res_store.elem_size = elem_size
    return res_store


_bi_init_shape = _te_init_shape


def _bi_init_ss(
        res_store,
        res_data,
        vec_size,
        res_shape,
        res_shape_tuple,
        elem_size,
        mem_type):
    '''Init BigIntStorage and the corresponding TensorShapeStorage'''
    return _bi_init_store(
        res_store, res_data, vec_size, elem_size, mem_type
    ), _bi_init_shape(res_shape, res_shape_tuple)


def _fp_init_store(
        res_store,
        fpn_storage,
        base_storage,
        exp_storage,
        vec_size,
        n,
        max_int,
        mem_type,
        data_type,
):
    '''
    Init FixedPointStorage class,
    paras are identical to the elements in FixedPointStorage
    '''
    if res_store is None:
        res_store = FixedPointStorage(
            fpn_storage,
            base_storage,
            exp_storage,
            vec_size,
            n,
            max_int,
            mem_type,
            data_type,
        )
    else:
        res_store.bigint_storage = fpn_storage
        res_store.base_storage = base_storage
        res_store.exp_storage = exp_storage
        res_store.vec_size = vec_size
        res_store.mem_type = mem_type
        '''TensorStorage needed paras'''
        res_store.data_type = data_type
        '''En/Decode needed paras '''
        res_store.encode_n = n
        res_store.max_int = max_int
    return res_store


def _fp_init_ss(
        res_store,
        fpn_storage,
        base_storage,
        exp_storage,
        vec_size,
        n,
        max_int,
        res_shape,
        res_shape_tuple,
        mem_type,
        data_type,
):
    '''Init FexiedPointStorage and the corresponding TensorShapeStorage'''
    return _fp_init_store(
        res_store,
        fpn_storage,
        base_storage,
        exp_storage,
        vec_size,
        n,
        max_int,
        mem_type,
        data_type,
    ), _te_init_shape(res_shape, res_shape_tuple)


def __get_FPGA_device_num(device_type: int):
    '''
    get the actual physical number of FPGA device from the current mem_type
    ----------------
    Para: device_type, the mem_type stored in Storage type, since it is mixed
    with GPU and CPU, to get physical No. of FPGA, we should do some pre-process
    '''
    # if device_type >= MIN_FPGA and device_type <= MAX_FPGA:
    #     FPGA_dev_num = device_type % 10
    # else:
    #     raise PermissionError("DEVICE TYPE IS NOT FPGA!")
    return 0


def pi_encrypt(pub_key, fps, res=None, stream=None):
    '''
    perform paillier encryption for FixedPointStorage,
    use raw encrypt with no obfuscation
    ----------------
    Para:
        pubkey: Dev_PubKeyPtr, the PaillierPublicKey class stored in GPU memory
        fps:    FixedPointStorage, fpn value waiting to be encrypted
        res:    None or PaillierEncryptedStorage, return value, default None
        stream: None, currently not used
    Return:
        PaillierEncryptedStorage, the encrypted value
    '''
    src_fpn = fps.bigint_storage
    src_base = fps.base_storage
    src_exp = fps.exp_storage
    vec_size = fps.vec_size

    if res is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(vec_size * CIPHER_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    else:
        res_pen = res.pen_storage
        res_base = res.base_storage
        res_exp = res.exp_storage

    # get the actual FPGA device number and pass it to C-level function
    FPGA_dev_num = __get_FPGA_device_num(fps.mem_type)
    FPGA_LIB.encrypt_without_obf(
        c_char_p(src_fpn),
        c_void_p(src_base),
        c_void_p(src_exp),
        c_char_p(res_pen),
        c_void_p(res_base),
        c_void_p(res_exp),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_size_t(PLAIN_BITS),
        c_size_t(CIPHER_BITS),
        c_size_t(vec_size),
        c_size_t(FPGA_dev_num),
    )
    return _pi_init_store(
        res,
        res_pen,
        res_base,
        res_exp,
        vec_size,
        fps.mem_type,
        fps.data_type,
        fps.encode_n,
        fps.max_int,
    )


def pi_decrypt(pub_key, priv_key, pes, res=None, stream=None):
    '''
    perform decryption and decode as a whole
    ---------------------
    Para:
        pub_key:   Dev_PubKeyStorage, PaillierPublicKey stored in GPU mem
        priv_key:  Dev_PrivKeyStorage, PaillierPrivateKey stored in GPU mem
        pes:       PaillierEncryptedStorage, pens waiting to be decrypted
        res:       TensorStorage, the return value;
        stream:    None, currently not used
        fps:       FixedPointStorage, the middle memory space used
                   after decrypt and before encode
    Return:
        TensorStorage, the decrypted then decoded value
    '''
    src_pen = pes.pen_storage
    src_base = pes.base_storage
    src_exp = pes.exp_storage
    vec_size = pes.vec_size
    '''malloc space for the return FixedPointStorage'''
    res_fpn = FPGA_LIB.c_malloc(c_size_t(vec_size * PLAIN_BYTE))
    res_base = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    res_exp = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    '''call the decrypt function'''
    FPGA_dev_num = __get_FPGA_device_num(pes.mem_type)
    FPGA_LIB.decrypt(
        c_char_p(src_pen),
        c_void_p(src_base),
        c_void_p(src_exp),
        c_char_p(res_fpn),
        c_void_p(res_base),
        c_void_p(res_exp),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        priv_key.p,
        priv_key.q,
        priv_key.psquare,
        priv_key.qsquare,
        priv_key.q_inverse,
        priv_key.hp,
        priv_key.hq,
        c_size_t(PLAIN_BITS),
        c_size_t(CIPHER_BITS),
        c_size_t(vec_size),
        c_size_t(FPGA_dev_num),
    )
    '''call the decode function'''
    decrypt_store = FixedPointStorage(
        res_fpn,
        res_base,
        res_exp,
        vec_size,
        pes.encode_n,
        pes.encode_max_int,
        pes.mem_type,
        pes.data_type,
    )
    return fp_decode(decrypt_store, res, stream)


def pi_obfuscate(pub_key, pes, obf_seeds, res, stream):
    '''
    apply obfuscation to a PaillierEncryptedStorage using the
    obfuscation seed given, actually a mulmod
    ----------------------
    Para:
        pubkey:    Dev_PubKeyStorage, PaillierPublicKey stored in GPU mem
        pes:       PaillierEncryptedStorage, raw pen haven't be obfuscated
        obf_seeds: BigIntStorage, random bigint generated by pi_gen_obf_seed
        res:       PaillierEncryptedStorage, the obfuscated return value
    Return:
        PaillierEncryptedStorage, the same as res
    '''
    # get the pen storage data
    src_pen = pes.pen_storage
    src_base = pes.base_storage
    src_exp = pes.exp_storage
    vec_size = pes.vec_size
    # get the bigint random ptr
    obf_rand = obf_seeds.bigint_storage
    '''initialize the result space'''
    if res is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(vec_size * CIPHER_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    else:
        res_pen = res.pen_storage
        res_base = res.base_storage
        res_exp = res.exp_storage
    '''run the modular mul function'''
    # we will do the obfs on the device same as pes's device
    # Although the obfs_seed may be generated on another device
    # But since all datas are stored in CPU memory, this won't be a serious
    # problem
    FPGA_dev_num = __get_FPGA_device_num(pes.mem_type)
    FPGA_LIB.obf_modular_multiplication(
        c_char_p(src_pen),
        c_void_p(src_base),
        c_void_p(src_exp),
        c_char_p(obf_rand),
        c_char_p(res_pen),
        c_void_p(res_base),
        c_void_p(res_exp),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_size_t(CIPHER_BITS),
        c_size_t(CIPHER_BITS),
        c_size_t(vec_size),
        c_size_t(FPGA_dev_num),
    )
    return _pi_init_store(
        res,
        res_pen,
        res_base,
        res_exp,
        vec_size,
        pes.mem_type,
        pes.data_type,
        pes.encode_n,
        pes.encode_max_int,
    )


def pi_gen_obf_seed(res_store, pub_key, count, elem_size, rand_seed, stream):
    '''
    generate random bigint and perform expmod based on the given public key.
    The calculation result is then used as obfuscation seed for further encrypt.
    --------------
    Para:
        res_store:   BigIntStorage, the return value
        pub_key:     Dev_PubKeyStorage, PaillierPublicKey stored in GPU mem
        count:       int, the number of random numbers need to be generated
        elem_size:   int, the length of the random bigint
        rand_seed:   the seed used for generating random number
    Return:
        BigIntStorage, same as res_store
    '''
    rand_storage = bi_gen_rand(elem_size, count, None, rand_seed, stream)
    rand_data = rand_storage.bigint_storage
    if res_store is None:
        res_data = FPGA_LIB.c_malloc(c_size_t(count * CIPHER_BYTE))
    else:
        res_data = res_store.bigint_storage
    FPGA_dev_num = __get_FPGA_device_num(rand_storage.mem_type)
    FPGA_LIB.obf_modular_exponentiation(
        c_char_p(rand_data),
        c_size_t(1024),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_char_p(res_data),
        c_size_t(CIPHER_BITS),
        c_size_t(count),
        c_size_t(FPGA_dev_num),
    )
    return _bi_init_store(res_store, res_data, count, MEM_DEVICE, elem_size)


def pi_gen_obf_seed_gmp(res_store, pub_key, count, elem_size, stream):
    '''
    generate random bigint and perform expmod based on the given public key.
    The calculation result is then used as obfuscation seed for further encrypt.
    --------------
    Para:
        res_store:   BigIntStorage, the return value
        pub_key:     Dev_PubKeyStorage, PaillierPublicKey stored in GPU mem
        count:       int, the number of random numbers need to be generated
        elem_size:   int, the length of the random bigint
        rand_seed:   the seed used for generating random number
    Return:
        BigIntStorage, same as res_store
    '''
    res_rand = FPGA_LIB.c_malloc(c_size_t(count * 1024 // 8))
    FPGA_LIB.gmp_random(
        c_char_p(res_rand),
        c_size_t(1024),
        c_size_t(1024),
        c_size_t(1024),
        c_size_t(count),
        pub_key.n,
    )
    if res_store is None:
        res_data = FPGA_LIB.c_malloc(c_size_t(count * CIPHER_BYTE))
    else:
        res_data = res_store.bigint_storage
    FPGA_dev_num = 0
    FPGA_LIB.obf_modular_exponentiation(
        c_char_p(res_rand),
        c_size_t(1024),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_char_p(res_data),
        c_size_t(CIPHER_BITS),
        c_size_t(count),
        c_size_t(FPGA_dev_num),
    )
    return _bi_init_store(res_store, res_data, count, MEM_DEVICE, 2048 // 8)


def __shape_decompose(shape):
    '''
    Decompose TensorShapeStorage to 2-D tuple
    satisfying fpga computation demand

    WARNING:
    not same output as numpy,
    extra switch needed after computing to suit numpy shape output
    '''
    shape_tuple = shape.to_tuple()
    if len(shape_tuple) == 0:
        return 1, 1
    elif len(shape_tuple) == 1:
        return 1, shape_tuple[0]
    elif len(shape_tuple) == 2:
        return shape_tuple[0], shape_tuple[1]
    else:
        raise PermissionError("Invalid Shape")


def __shape_resolve(shape_1, shape_2):
    '''check aligment capability of shape_1 & shape_2 to support broadcast'''

    def check_func(a, b):
        return a == b or a == 1 or b == 1

    P, Q = __shape_decompose(shape_1)
    R, S = __shape_decompose(shape_2)
    max_shape_size = max(len(shape_1.to_tuple()), len(shape_2.to_tuple()))
    if check_func(P, R) and check_func(Q, S):
        if max_shape_size == 0:
            return P, Q, R, S, ()
        elif max_shape_size == 1:
            return P, Q, R, S, (max(Q, S),)
        elif max_shape_size == 2:
            return P, Q, R, S, (max(P, R), max(Q, S))
        else:
            raise PermissionError(f"Invalid shape, {shape_1}, {shape_2}")
    else:
        raise PermissionError("shape cannot align", shape_1, shape_2)


def pi_add(
        pub_key,
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store=None,
        res_shape=None,
        stream=None,
):
    '''
    Perform element-wise encrypted add, support broadcast over cols or rows
    ---------------
    Paras:
        pub_key:     Dev_PubKeyStorage, PaillierPublicKey stored in GPU mem
        left_store:  PaillierEncryptedStorage, left_operator
        right_store: PaillierEncryptedStorage, right_operator
        left_shape:  TensorShapeStorage, left_operator's  shape
        right_shape: TensorShapeStorage, right_operator's shape
        res_store:   PaillierEncrpytedStorage, return value's data
        res_shape:   TensorShapeStorage, return value's shape
    Return:
        tuple: (PaillierEncrytedStorage, TensorShapeStorage)
    Raise:
        PermissionError, if left/right operators cannot aligned for compute,
                         even if broadcast is supported
    '''
    # first get the shape of the res type
    P, Q, R, S, res_shape_tuple = __shape_resolve(left_shape, right_shape)
    res_size = max(P, R) * max(Q, S)
    # the left_store data
    l_pen = left_store.pen_storage
    l_base = left_store.base_storage
    l_exp = left_store.exp_storage
    # the right_store data
    r_pen = right_store.pen_storage
    r_base = right_store.base_storage
    r_exp = right_store.exp_storage
    # malloc space for the return value
    if res_store is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(res_size * CIPHER_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
    else:
        res_pen = res_store.pen_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage
    # perform calculation
    FPGA_dev_num = __get_FPGA_device_num(left_store.mem_type)
    FPGA_LIB.pen_matrix_add_pen_matrix(
        c_char_p(l_pen),
        c_void_p(l_base),
        c_void_p(l_exp),
        c_char_p(r_pen),
        c_void_p(r_base),
        c_void_p(r_exp),
        c_char_p(res_pen),
        c_void_p(res_base),
        c_void_p(res_exp),
        c_size_t(P),
        c_size_t(Q),
        c_size_t(R),
        c_size_t(S),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_size_t(CIPHER_BITS),
        c_size_t(FPGA_dev_num),
    )
    # handle the result's data type
    data_type = 0
    if left_store.data_type == INT64_TYPE and right_store.data_type == INT64_TYPE:
        data_type = INT64_TYPE
    else:
        data_type = FLOAT_TYPE
    return _pi_init_ss(
        res_store,
        res_pen,
        res_base,
        res_exp,
        res_size,
        res_shape,
        res_shape_tuple,
        left_store.mem_type,
        data_type,
        left_store.encode_n,
        left_store.encode_max_int,
    )


def pi_mul(
        pub_key,
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store=None,
        res_shape=None,
        stream=None,
):
    '''
    Perform element-wise encrypted muliply, support broadcast for cols and rows
    --------------------
    Paras:
        pub_key:     Dev_PubKeyStorage, PaillierPublicKey stored in GPU mem
        left_store:  PaillierEncryptedStorage, left_operator
        right_store: FixedPointStorage, right_operator
        left_shape:  TensorShapeStorage, left_operator's  shape
        right_shape: TensorShapeStorage, right_operator's shape
        res_store:   PaillierEncrpytedStorage, return value's data
        res_shape:   TensorShapeStorage, return value's shape
    Return:
        tuple: (PaillierEncrytedStorage, TensorShapeStorage)
    Raise:
        PermissionError, if left/right operators cannot aligned for compute,
                         even if broadcast is supported
    '''
    # check for alignment capablity of shapes
    P, Q, R, S, res_shape_tuple = __shape_resolve(left_shape, right_shape)
    # P,Q is the dim of the left_store(pen)
    # R,S is the dim of the right_store(fpn)
    res_size = max(P, R) * max(Q, S)
    # the left_store data
    l_pen = left_store.pen_storage
    l_base = left_store.base_storage
    l_exp = left_store.exp_storage
    # the right_store data
    r_fpn = right_store.bigint_storage
    r_base = right_store.base_storage
    r_exp = right_store.exp_storage
    # malloc space for the return value
    if res_store is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(res_size * CIPHER_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
    else:
        res_pen = res_store.pen_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage
    # '''call the batch_mul function'''
    FPGA_dev_num = __get_FPGA_device_num(left_store.mem_type)
    FPGA_LIB.fpn_matrix_elementwise_multiply_pen_matrix(
        c_char_p(r_fpn),
        c_void_p(r_base),
        c_void_p(r_exp),
        c_char_p(l_pen),
        c_void_p(l_base),
        c_void_p(l_exp),
        c_char_p(res_pen),
        c_void_p(res_base),
        c_void_p(res_exp),
        c_size_t(R),
        c_size_t(S),
        c_size_t(P),
        c_size_t(Q),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_size_t(PLAIN_BITS),
        c_size_t(CIPHER_BITS),
        c_size_t(FPGA_dev_num),
    )
    # handle the result's data type
    data_type = 0
    if left_store.data_type == INT64_TYPE and right_store.data_type == INT64_TYPE:
        data_type = INT64_TYPE
    else:
        data_type = FLOAT_TYPE
    return _pi_init_ss(
        res_store,
        res_pen,
        res_base,
        res_exp,
        res_size,
        res_shape,
        res_shape_tuple,
        left_store.mem_type,
        data_type,
        left_store.encode_n,
        left_store.encode_max_int,
    )


def fp_transpose(left_store, left_shape, res_store, res_shape, stream):
    '''
    transpose the C-memory stored matrix of FixedPointStorage,
    support at most 2-D matrix
    -----------------
    Para:
        left_store:  FixedPointStorage, left_operator
        left_shape:  TensorShapeStorage, left_operator's  shape
        res_store:   PaillierEncrpytedStorage, return value's data
        res_shape:   TensorShapeStorage, return value's shape
    Return:
        tuple: (FixedPointStorage, TensorShapeStorage)
    Raise:
        PermissionError, if dimension is higher than 2-D, not supported
    '''
    # didn't use FPGA driver, no need for check for mem_type
    left_shape_tuple = left_shape.to_tuple()
    # get the left_store parameters
    src_fpn = left_store.bigint_storage
    src_base = left_store.base_storage
    src_exp = left_store.exp_storage
    vec_size = left_store.vec_size
    # malloc space for the res value
    if res_store is None:
        res_fpn = FPGA_LIB.c_malloc(c_size_t(vec_size * PLAIN_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    else:
        res_fpn = res_store.bigint_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage
    # Handling different shapes
    if len(left_shape_tuple) < 2:
        # the tuple is 0-D or 1-D
        # transpose returns the same value as input in numpy
        # make the output same as input, memcpy is to prevent potential double
        # free
        FPGA_LIB.c_memcpy(
            c_void_p(res_fpn),
            c_void_p(src_fpn),
            c_size_t(vec_size * PLAIN_BYTE))
        FPGA_LIB.c_memcpy(
            c_void_p(res_base),
            c_void_p(src_base),
            c_size_t(vec_size * U_INT32_BYTE))
        FPGA_LIB.c_memcpy(
            c_void_p(res_exp),
            c_void_p(src_exp),
            c_size_t(vec_size * U_INT32_BYTE))
        return _fp_init_ss(
            res_store,
            res_fpn,
            res_base,
            res_exp,
            left_store.vec_size,
            left_store.encode_n,
            left_store.max_int,
            left_shape,
            left_shape_tuple,
            left_store.mem_type,
            left_store.data_type,
        )
    elif len(left_shape_tuple) == 2:
        # the tuple is 2-D
        # do a normal transpose
        res_shape_tuple = (left_shape_tuple[1], left_shape_tuple[0])
        FPGA_LIB.transpose(
            c_char_p(src_fpn),
            c_void_p(src_base),
            c_void_p(src_exp),
            c_char_p(res_fpn),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_size_t(res_shape_tuple[1]),
            c_size_t(res_shape_tuple[0]),
            c_size_t(PLAIN_BITS),
        )
        return _fp_init_ss(
            res_store,
            res_fpn,
            res_base,
            res_exp,
            vec_size,
            left_store.encode_n,
            left_store.max_int,
            res_shape,
            res_shape_tuple,
            left_store.mem_type,
            left_store.data_type,
        )
    else:
        raise PermissionError("Unsupported shape")


def pi_matmul(
        pub_key,
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store=None,
        res_shape=None,
        stream=None,
):
    '''
    Perform matrix multiply under encryption
    ------------------
    Paras:
        pub_key:     Dev_PubKeyStorage, PaillierPublicKey stored in GPU mem
        left_store:  PaillierEncryptedStorage, left_operator
        right_store: FixedPointStorage, right_operator
        left_shape:  TensorShapeStorage, left_operator's  shape
        right_shape: TensorShapeStorage, right_operator's shape
        res_store:   PaillierEncrpytedStorage, return value's data
        res_shape:   TensorShapeStorage, return value's shape
    Return:
        tuple: (PaillierEncrytedStorage, TensorShapeStorage)
    Raise:
        PermissionError, if shape is invalid for 1-D or 2-D matrix mul
        ValueError, if left/right operators' shape can't align for matmul
    '''

    # '''pre-process shape'''
    left_tuple = left_shape.to_tuple()
    right_tuple = right_shape.to_tuple()

    if len(left_tuple) == 0 or len(right_tuple) == 0 or len(left_tuple) > 2 or len(right_tuple) > 2:
        raise PermissionError("Invalid shape")
    P, Q = __shape_decompose(left_shape)
    R, S = __shape_decompose(right_shape)
    if len(right_tuple) == 1:
        R, S = S, R
    if Q != R:
        raise ValueError("shape not aligned")
    if len(left_tuple) == 1 and len(right_tuple) == 1:
        res_shape_tuple = ()
    elif len(left_tuple) == 1 and len(right_tuple) == 2:
        res_shape_tuple = (S,)
    elif len(left_tuple) == 2 and len(right_tuple) == 1:
        res_shape_tuple = (P,)
    elif len(left_tuple) == 2 and len(right_tuple) == 2:
        res_shape_tuple = (P, S)
    else:
        raise RuntimeError(
            "Default error, won't occur unless something VERY STRANGE happens"
        )
    res_size = P * S
    # the left_store data
    l_pen = left_store.pen_storage
    l_base = left_store.base_storage
    l_exp = left_store.exp_storage
    # the right_store data
    r_fpn = right_store.bigint_storage
    r_base = right_store.base_storage
    r_exp = right_store.exp_storage
    # malloc space for the return value
    if res_store is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(res_size * CIPHER_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
    else:
        res_pen = res_store.pen_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage
    '''call the matrix_mul function'''
    FPGA_dev_num = __get_FPGA_device_num(left_store.mem_type)
    FPGA_LIB.pen_matrix_multiply_fpn_matrix(
        c_char_p(l_pen),
        c_void_p(l_base),
        c_void_p(l_exp),
        c_char_p(r_fpn),
        c_void_p(r_base),
        c_void_p(r_exp),
        c_char_p(res_pen),
        c_void_p(res_base),
        c_void_p(res_exp),
        c_size_t(P),
        c_size_t(Q),
        c_size_t(S),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_size_t(PLAIN_BITS),
        c_size_t(CIPHER_BITS),
        c_size_t(FPGA_dev_num),
    )
    # check for data type
    data_type = 0
    if left_store.data_type == INT64_TYPE and right_store.data_type == INT64_TYPE:
        data_type = INT64_TYPE
    else:
        data_type = FLOAT_TYPE
    return _pi_init_ss(
        res_store,
        res_pen,
        res_base,
        res_exp,
        res_size,
        res_shape,
        res_shape_tuple,
        left_store.mem_type,
        data_type,
        left_store.encode_n,
        left_store.encode_max_int,
    )


def pi_rmatmul(
        pub_key,
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store=None,
        res_shape=None,
        stream=None,
):
    '''
    Perform matrix multiply under encryption.
    rmatmul means right_op is PaillierEncryptedStorage, differ from pi_matmul
    Due to implementation of cuda code, right_store needs to be transposed
    -------------------------
    Paras:
        pub_key:     Dev_PubKeyStorage, PaillierPublicKey stored in GPU mem
        left_store:  FixedPointStorage, left_operator
        right_store: PaillierEncryptedStorage, right_operator
        left_shape:  TensorShapeStorage, left_operator's  shape
        right_shape: TensorShapeStorage, right_operator's shape
        res_store:   PaillierEncrpytedStorage, return value's data
        res_shape:   TensorShapeStorage, return value's shape
    Return:
        tuple: (PaillierEncrytedStorage, TensorShapeStorage)
    Raise:
        PermissionError, if shape is invalid for 1-D or 2-D matrix mul
        ValueError, if left/right operators' shape can't align for matmul
        RuntimeError,  default error for shape evaluation
    '''
    left_tuple = left_shape.to_tuple()
    right_tuple = right_shape.to_tuple()
    if len(left_tuple) == 0 or len(right_tuple) == 0 or len(left_tuple) > 2 or len(right_tuple) > 2:
        raise PermissionError("Invalid shape")
    P, Q = __shape_decompose(left_shape)
    R, S = __shape_decompose(right_shape)
    if len(right_tuple) == 1:
        R, S = S, R
    if Q != R:
        raise ValueError("shape not aligned")
    if len(left_tuple) == 1 and len(right_tuple) == 1:
        res_shape_tuple = ()
    elif len(left_tuple) == 1 and len(right_tuple) == 2:
        res_shape_tuple = (S,)
    elif len(left_tuple) == 2 and len(right_tuple) == 1:
        res_shape_tuple = (P,)
    elif len(left_tuple) == 2 and len(right_tuple) == 2:
        res_shape_tuple = (P, S)
    else:
        raise RuntimeError(
            "You should never ever see this error unless something VERY STRANGE occurs"
        )
    res_size = P * S
    # the left_store data
    l_fpn = left_store.bigint_storage
    l_base = left_store.base_storage
    l_exp = left_store.exp_storage
    # the right_store data
    r_pen = right_store.pen_storage
    r_base = right_store.base_storage
    r_exp = right_store.exp_storage
    # malloc space for the return value
    if res_store is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(res_size * CIPHER_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
    else:
        res_pen = res_store.pen_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage
    '''call the matrix_mul function'''
    FPGA_dev_num = __get_FPGA_device_num(left_store.mem_type)
    FPGA_LIB.fpn_matrix_multiply_pen_matrix(
        c_char_p(l_fpn),
        c_void_p(l_base),
        c_void_p(l_exp),
        c_char_p(r_pen),
        c_void_p(r_base),
        c_void_p(r_exp),
        c_char_p(res_pen),
        c_void_p(res_base),
        c_void_p(res_exp),
        c_size_t(P),
        c_size_t(Q),
        c_size_t(S),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_size_t(PLAIN_BITS),
        c_size_t(CIPHER_BITS),
        c_uint32(FPGA_dev_num),
    )
    # check for data type
    data_type = 0
    if left_store.data_type == INT64_TYPE and right_store.data_type == INT64_TYPE:
        data_type = INT64_TYPE
    else:
        data_type = FLOAT_TYPE

    return _pi_init_ss(
        res_store,
        res_pen,
        res_base,
        res_exp,
        res_size,
        res_shape,
        res_shape_tuple,
        right_store.mem_type,
        data_type,
        right_store.encode_n,
        right_store.encode_max_int,
    )


def pi_transpose(left_store, left_shape, res_store, res_shape, stream):
    '''
    transpose the C-memory stored matrix of PaillierEncryptedStorage,
    support at most 2-D matrix
    -----------------
    Para:
        left_store:  PaillierEncryptedStorage, left_operator
        left_shape:  TensorShapeStorage, left_operator's  shape
        res_store:   PaillierEncrpytedStorage, return value's data
        res_shape:   TensorShapeStorage, return value's shape
    Return:
        tuple: (PaillierEncryptedStorage, TensorShapeStorage)
    Raise:
        PermissionError, if dimension is higher than 2-D, not supported
    '''
    left_shape_tuple = left_shape.to_tuple()
    # get the left_store parameters
    src_pen = left_store.pen_storage
    src_base = left_store.base_storage
    src_exp = left_store.exp_storage
    vec_size = left_store.vec_size
    # malloc space for the res value
    if res_store is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(vec_size * CIPHER_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    else:
        res_pen = res_store.pen_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage
    '''Start handling different type of data '''
    if len(left_shape_tuple) < 2:
        # just a raw memcpy, no transpose needed for this scene
        FPGA_LIB.c_memcpy(
            c_void_p(res_pen),
            c_void_p(src_pen),
            c_size_t(vec_size * CIPHER_BYTE))
        FPGA_LIB.c_memcpy(
            c_void_p(res_base),
            c_void_p(src_base),
            c_size_t(vec_size * U_INT32_BYTE))
        FPGA_LIB.c_memcpy(
            c_void_p(res_exp),
            c_void_p(src_exp),
            c_size_t(vec_size * U_INT32_BYTE))
        return _pi_init_ss(
            res_store,
            res_pen,
            res_base,
            res_exp,
            left_store.vec_size,
            left_shape,
            left_shape_tuple,
            left_store.mem_type,
            left_store.data_type,
            left_store.encode_n,
            left_store.encode_max_int,
        )
    elif len(left_shape_tuple) == 2:
        res_shape_tuple = (left_shape_tuple[1], left_shape_tuple[0])
        # call the C transpose functions
        FPGA_LIB.transpose(
            c_char_p(src_pen),
            c_void_p(src_base),
            c_void_p(src_exp),
            c_char_p(res_pen),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_size_t(res_shape_tuple[1]),
            c_size_t(res_shape_tuple[0]),
            c_size_t(CIPHER_BITS),
        )
        return _pi_init_ss(
            res_store,
            res_pen,
            res_base,
            res_exp,
            vec_size,
            res_shape,
            res_shape_tuple,
            left_store.mem_type,
            left_store.data_type,
            left_store.encode_n,
            left_store.encode_max_int,
        )
    else:
        raise PermissionError("Invalid Shape")


def pi_sum(
        pub_key,
        left_store,
        left_shape,
        axis,
        res_store=None,
        res_shape=None,
        stream=None):
    '''
    Perform sum according to the axis
    ----------------------
    Para:
        pub_key:     Dev_PubKeyStorage, PaillierPublicKey stored in GPU mem
        left_store:  PaillierEncryptedStorage, left_operator
        left_shape:  TensorShapeStorage, left_operator's  shape
        axis:        int or None, the dimension which sum is performed
                        None: sum over all elements
                        0:    sum vertically, over the 1st demension
                        1:    sum horizontally, over the 2nd demension
        res_store:   PaillierEncrpytedStorage, return value's data
        res_shape:   TensorShapeStorage, return value's shape
    Return:
        tuple, (PaillierEncryptedStorage, TensorShapeStorage)
    Raise:
        Permission error: when the input axis is not aligned to input shape
    '''
    # get the original data
    src_pen = left_store.pen_storage
    src_base = left_store.base_storage
    src_exp = left_store.exp_storage
    vec_size = left_store.vec_size
    # initialize the result
    res_pen, res_base, res_exp = 0, 0, 0
    res_shape_tuple = ()
    # get the original data's tuple
    left_shape_tuple = left_shape.to_tuple()

    if len(left_shape_tuple) == 0:
        # handling shape (), meaning only one element in left_store
        if axis is not None and axis != 0:
            raise PermissionError(
                "Cannot set axis other than 0 or None for dimension 0"
            )
        if res_store is None:
            res_pen = FPGA_LIB.c_malloc(c_size_t(vec_size * CIPHER_BYTE))
            res_base = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
            res_exp = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
        else:
            res_pen = res_store.pen_storage
            res_base = res_store.base_storage
            res_exp = res_store.exp_storage
        FPGA_LIB.c_memcpy(
            c_void_p(res_pen),
            c_void_p(src_pen),
            c_size_t(vec_size * CIPHER_BYTE))
        FPGA_LIB.c_memcpy(
            c_void_p(res_base),
            c_void_p(src_base),
            c_size_t(vec_size * U_INT32_BYTE))
        FPGA_LIB.c_memcpy(
            c_void_p(res_exp),
            c_void_p(src_exp),
            c_size_t(vec_size * U_INT32_BYTE))
        return _pi_init_ss(
            left_store,
            res_pen,
            res_base,
            res_exp,
            vec_size,
            left_shape,
            left_shape_tuple,
            left_store.mem_type,
            left_store.data_type,
            left_store.encode_n,
            left_store.encode_max_int,
        )
    elif axis is None or len(left_shape_tuple) == 1:
        # handling shape (n, ) or axis == None
        # malloc space for results
        if len(left_shape_tuple) == 1 and axis is not None and axis >= 1:
            raise PermissionError(
                "axis is out of bounds for array of dimension 1")
        if res_store is None:
            res_pen = FPGA_LIB.c_malloc(c_size_t(1 * CIPHER_BYTE))
            res_base = FPGA_LIB.c_malloc(c_size_t(1 * U_INT32_BYTE))
            res_exp = FPGA_LIB.c_malloc(c_size_t(1 * U_INT32_BYTE))
        else:
            res_pen = res_store.pen_storage
            res_base = res_store.base_storage
            res_exp = res_store.exp_storage
        # other return paras
        result_size = 1
        res_shape_tuple = ()
        '''call the C pen_sum function'''
        FPGA_dev_num = __get_FPGA_device_num(left_store.mem_type)
        FPGA_LIB.pen_sum(
            c_char_p(src_pen),
            c_void_p(src_base),
            c_void_p(src_exp),
            c_char_p(res_pen),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_size_t(1),
            c_size_t(vec_size),
            pub_key.n,
            pub_key.g,
            pub_key.nsquare,
            pub_key.max_int,
            c_size_t(CIPHER_BITS),
            c_size_t(FPGA_dev_num),
        )
    elif axis == 0:
        # handling 2-D matrix, axis == 0 means sum vertically
        # since current sum only support horizontal sum
        # aka batch sum over continuous memory space
        transpose_store, transpose_shape = pi_transpose(
            left_store, left_shape, None, None, stream
        )
        src_pen = transpose_store.pen_storage
        src_base = transpose_store.base_storage
        src_exp = transpose_store.exp_storage
        transpose_tuple = transpose_shape.to_tuple()
        '''perform sum on the transposed matrix'''
        if res_store is None:
            res_pen = FPGA_LIB.c_malloc(
                c_size_t(transpose_tuple[0] * CIPHER_BYTE))
            res_base = FPGA_LIB.c_malloc(
                c_size_t(transpose_tuple[0] * U_INT32_BYTE))
            res_exp = FPGA_LIB.c_malloc(
                c_size_t(transpose_tuple[0] * U_INT32_BYTE))
        else:
            res_pen = res_store.pen_storage
            res_base = res_store.base_storage
            res_exp = res_store.exp_storage
        result_size = transpose_tuple[0]
        res_shape_tuple = (transpose_tuple[0],)
        '''Call the C function'''
        # print(transpose_tuple[0])
        FPGA_dev_num = __get_FPGA_device_num(left_store.mem_type)
        FPGA_LIB.pen_sum(
            c_char_p(src_pen),
            c_void_p(src_base),
            c_void_p(src_exp),
            c_char_p(res_pen),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_size_t(transpose_tuple[0]),
            c_size_t(transpose_tuple[1]),
            pub_key.n,
            pub_key.g,
            pub_key.nsquare,
            pub_key.max_int,
            c_size_t(CIPHER_BITS),
            c_size_t(FPGA_dev_num),
        )
    elif axis == 1:
        # handling 2-D matrix, axis == 1 means sum horizontally
        if res_store is None:
            res_pen = FPGA_LIB.c_malloc(
                c_size_t(left_shape_tuple[0] * CIPHER_BYTE))
            res_base = FPGA_LIB.c_malloc(
                c_size_t(left_shape_tuple[0] * U_INT32_BYTE))
            res_exp = FPGA_LIB.c_malloc(
                c_size_t(left_shape_tuple[0] * U_INT32_BYTE))
        else:
            res_pen = res_store.pen_storage
            res_base = res_store.base_storage
            res_exp = res_store.exp_storage
        result_size = left_shape_tuple[0]
        # the res_shape tuple is also clear
        result_size = left_shape_tuple[0]
        res_shape_tuple = (left_shape_tuple[0],)
        '''Call the pen_sum: a C function'''
        FPGA_dev_num = __get_FPGA_device_num(left_store.mem_type)
        FPGA_LIB.pen_sum(
            c_char_p(src_pen),
            c_void_p(src_base),
            c_void_p(src_exp),
            c_char_p(res_pen),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_size_t(left_shape_tuple[0]),
            c_size_t(left_shape_tuple[1]),
            pub_key.n,
            pub_key.g,
            pub_key.nsquare,
            pub_key.max_int,
            c_size_t(CIPHER_BITS),
            c_size_t(FPGA_dev_num),
        )
    else:
        raise PermissionError("Invalid Axis or Shape")

    return _pi_init_ss(
        res_store,
        res_pen,
        res_base,
        res_exp,
        result_size,
        res_shape,
        res_shape_tuple,
        left_store.mem_type,
        left_store.data_type,
        left_store.encode_n,
        left_store.encode_max_int,
    )


def fp_encode(
        store, n, max_int, precision=None, max_exponent=None, res=None, stream=None
):
    '''
    Perform encode to a TensorStorage
    -----------------
    Paras:
        store:        TensorStorage, raw data to be encoded
        n:            big int, the same n in pubkey used for encryption
        max_int:      big int, same max_int in pubkey.
        precision:    int, the precision of encoding, default None
        max_exponent: None or int, currently not used
        res:          FixedPointStorage, the return value
    Return:
        FixedPointStorage, same as res
    Raise:
        PermissionError: For unsupported data type or encoding style
    '''
    if max_exponent is not None:
        raise PermissionError("max_exponent not supported")
    if precision is None:
        precision = -1
    data_storage = store.data
    vec_size = store.vec_size
    # malloc the return memory space
    if res is None:
        res_fpn = FPGA_LIB.c_malloc(c_size_t(PLAIN_BYTE * vec_size))
        res_base = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * vec_size))
        res_exp = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * vec_size))
    else:
        res_fpn = res.bigint_storage
        res_base = res.base_storage
        res_exp = res.exp_storage
    # Due to the different nature of encoding float/int
    # Handle the two different data type seperately
    FPGA_dev_num = __get_FPGA_device_num(store.mem_type)
    if store.data_type == FLOAT_TYPE:
        FPGA_LIB.encode_double(
            c_void_p(data_storage),
            c_void_p(res_fpn),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_int32(precision),
            c_char_p(n.to_bytes(PLAIN_BYTE, 'little')),
            c_char_p(max_int.to_bytes(PLAIN_BYTE, 'little')),
            c_size_t(PLAIN_BITS),
            c_size_t(vec_size),
            c_size_t(FPGA_dev_num),
        )
    elif store.data_type == INT64_TYPE:
        FPGA_LIB.encode_int(
            c_void_p(data_storage),
            c_void_p(res_fpn),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_int32(precision),
            c_char_p(n.to_bytes(PLAIN_BYTE, 'little')),
            c_char_p(max_int.to_bytes(PLAIN_BYTE, 'little')),
            c_size_t(PLAIN_BITS),
            c_size_t(vec_size),
            c_size_t(FPGA_dev_num),
        )
    else:
        raise PermissionError("Invalid Data Type")

    '''get the three elements, store it in a FPNStorage'''

    return _fp_init_store(
        res,
        res_fpn,
        res_base,
        res_exp,
        vec_size,
        n,
        max_int,
        store.mem_type,
        store.data_type,
    )


def __fp_decode(store, res, stream):
    '''
    Decode a FixedPointStorage in CPU, using fp_c2p to implement
    Currently not used, as a GPU version has been done
    ------------------
    Paras:
        store:   FixedPointStorage, the raw data to be decoded
        res:     TensorStorage, the decoded result
    Return:
        TensorStorage, same as res
    '''
    vec_size = store.vec_size
    fpn_array = __get_c_fpn_storage(
        store.bigint_storage,
        store.base_storage,
        store.exp_storage,
        vec_size,
        store.encode_n,
        store.max_int,
    )

    CPU_decode = []
    if store.data_type == INT64_TYPE:
        for i in range(vec_size):
            CPU_decode.append(int(fpn_array[i].decode()))
    elif store.data_type == FLOAT_TYPE:
        for i in range(vec_size):
            CPU_decode.append(fpn_array[i].decode())
    else:
        raise PermissionError("Invalid Data Type")

    # reform the value to TensorStorage
    decode_data = te_p2c(CPU_decode, None)
    res_data = decode_data.data
    decode_data.data = None
    return _te_init_store(
        res,
        res_data,
        vec_size,
        store.mem_type,
        store.data_type)


def fp_decode(store, res, stream):
    '''
    Decode a FixedPointStorage in GPU
    ------------------
    Paras:
        store:   FixedPointStorage, the raw data to be decoded
        res:     TensorStorage, the decoded result
    Return:
        TensorStorage, same as res
    '''
    if store.data_type == FLOAT_TYPE:
        res_store = (
            FPGA_LIB.c_malloc(c_size_t(store.vec_size * DOUBLE_BYTE))
            if res is None
            else res.data
        )
        FPGA_LIB.decode_double(
            c_void_p(store.bigint_storage),
            c_void_p(store.base_storage),
            c_void_p(store.exp_storage),
            c_char_p(store.encode_n.to_bytes(PLAIN_BYTE, 'little')),
            c_char_p(store.max_int.to_bytes(PLAIN_BYTE, 'little')),
            c_size_t(PLAIN_BITS),
            c_void_p(res_store),
            c_size_t(store.vec_size),
        )
    elif store.data_type == INT64_TYPE:
        res_store = (
            FPGA_LIB.c_malloc(c_size_t(store.vec_size * INT64_BYTE))
            if res is None
            else res.data
        )
        FPGA_LIB.decode_int(
            c_void_p(store.bigint_storage),
            c_void_p(store.base_storage),
            c_void_p(store.exp_storage),
            c_char_p(store.encode_n.to_bytes(PLAIN_BYTE, 'little')),
            c_char_p(store.max_int.to_bytes(PLAIN_BYTE, 'little')),
            c_size_t(PLAIN_BITS),
            c_void_p(res_store),
            c_size_t(store.vec_size),
        )
    else:
        raise PermissionError("Invalid Data Type")
    return _te_init_store(
        res, res_store, store.vec_size, store.mem_type, store.data_type
    )


def fp_d2h(target, src, stream):
    return src


def bi_free(src):
    FPGA_LIB.c_free(c_void_p(src.bigint_storage))
    src.bigint_storage = None


def fp_free(src):
    FPGA_LIB.c_free(c_void_p(src.bigint_storage))
    FPGA_LIB.c_free(c_void_p(src.base_storage))
    FPGA_LIB.c_free(c_void_p(src.exp_storage))
    src.bigint_storage, src.base_storage, src.exp_storage = None, None, None


'''
    function: change the FixedPointStorage's data back into a C type
    As there is no shape involved in the function,
    we cannot know the return shape of the function
    input:
            src: FixedPointStorage, containing the data that need to be changed
    output:
            return value: containing 3 ndarray:
                            fpn_array,base_array,exp_array
'''


def fp_c2p(src):
    return __get_c_fpn_storage(
        src.bigint_storage,
        src.base_storage,
        src.exp_storage,
        src.vec_size,
        src.encode_n,
        src.max_int,
    )


def pi_c2p_mp(src):
    '''
    convert PaillierEncryptedStorage from C mem type to Python one
    this one use multiprocess to accelerate
    --------------
    Para:    src, PaillierEncryptedStorage
    Return:  tuple, each element is a ndarray,
                    identical to sequence of encoding, base, exponent
    '''
    return __get_c_pen_storage_mp(
        src.pen_storage,
        src.base_storage,
        src.exp_storage,
        src.vec_size,
        src.encode_n)


def pi_c2p(src):
    '''convert PaillierEncryptedStorage from C mem type to Python one'''
    return __get_c_pen_storage_raw(
        src.pen_storage,
        src.base_storage,
        src.exp_storage,
        src.vec_size,
        src.encode_n)


def fp_mul(
        left_store,
        right_store,
        left_shape,
        right_shape,
        res_store,
        res_shape,
        stream):
    '''
    Perform element-wise multiplication between two FixedPointStorage.
    This is a plaintext computation rather than an encrypted one.
    ------------------
    Paras:
        left_store, right_store: FixedPointStorage
        left_shape, right_shape: TensorShapeStorage
    Return:
        tuple, (FixedPointStorage, TensorShapeStorage)
    '''
    P, Q, R, S, res_shape_tuple = __shape_resolve(left_shape, right_shape)
    # P,Q is the dim of the left_store(pen)
    # R,S is the dim of the right_store(fpn)
    res_size = max(P, R) * max(Q, S)
    # the left_store data
    l_fpn = left_store.bigint_storage
    l_base = left_store.base_storage
    l_exp = left_store.exp_storage
    # the right_store data
    r_fpn = right_store.bigint_storage
    r_base = right_store.base_storage
    r_exp = right_store.exp_storage
    # malloc space for the return value
    if res_store is None:
        res_fpn = FPGA_LIB.c_malloc(c_size_t(res_size * PLAIN_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
    else:
        res_fpn = res_store.bigint_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage
    FPGA_dev_num = __get_FPGA_device_num(left_store.mem_type)
    FPGA_LIB.fpn_mul(
        c_char_p(l_fpn),
        c_void_p(l_base),
        c_void_p(l_exp),
        c_char_p(r_fpn),
        c_void_p(r_base),
        c_void_p(r_exp),
        c_char_p(res_fpn),
        c_void_p(res_base),
        c_void_p(res_exp),
        c_size_t(P),
        c_size_t(Q),
        c_size_t(R),
        c_size_t(S),
        c_char_p(left_store.encode_n.to_bytes(PLAIN_BYTE, 'little')),
        c_size_t(PLAIN_BITS),
        c_size_t(FPGA_dev_num),
    )
    # handle data_type
    data_type = 0
    if left_store.data_type == INT64_TYPE and right_store.data_type == INT64_TYPE:
        data_type = INT64_TYPE
    else:
        data_type = FLOAT_TYPE
    return _fp_init_ss(
        res_store,
        res_fpn,
        res_base,
        res_exp,
        res_size,
        left_store.encode_n,
        left_store.max_int,
        res_shape,
        res_shape_tuple,
        left_store.mem_type,
        data_type,
    )


def fp_p2c(target, src, data_type=FLOAT_TYPE):
    '''change a FixedPointNumber ndarray into a FixedPointStorage Class'''
    if isinstance(src, list):
        vec_size = len(src)
    elif isinstance(src, np.ndarray):
        vec_size = src.size
        src = src.flat
    else:
        raise TypeError("Unsupported Data Structure")
    # malloc the space for the type
    if target is None:
        res_fpn = FPGA_LIB.c_malloc(c_size_t(vec_size * PLAIN_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    else:
        res_fpn = target.bigint_storage
        res_base = target.base_storage
        res_exp = target.exp_storage
    # the temp ndarray buffer
    base_temp, exp_temp = [], []
    # get the two encoding parameters
    n = src[0].n
    max_int = src[0].max_int
    for i in range(vec_size):
        src_number = src[i].encoding.to_bytes(PLAIN_BYTE, 'little')
        FPGA_LIB.bigint_set(
            c_char_p(res_fpn),
            c_char_p(src_number),
            c_size_t(PLAIN_BITS),
            c_size_t(i))
        base_temp.append(src[i].BASE)
        exp_temp.append(src[i].exponent)

    base_arr_ptr = np.asarray(base_temp).ctypes.data_as(c_void_p)
    exp_arr_ptr = np.asarray(exp_temp).ctypes.data_as(c_void_p)
    FPGA_LIB.unsigned_set(c_void_p(res_base), base_arr_ptr, c_size_t(vec_size))
    FPGA_LIB.unsigned_set(c_void_p(res_exp), exp_arr_ptr, c_size_t(vec_size))

    return _fp_init_store(
        target,
        res_fpn,
        res_base,
        res_exp,
        vec_size,
        n,
        max_int,
        MEM_FPGA_NUM_0,
        data_type,
    )


def fp_h2d(target, src):
    return src


def _index_reset(index, dim_size):
    if index < 0:
        res_index = index + dim_size
        res_index = max(0, res_index)
    elif index > dim_size:
        res_index = dim_size
    else:
        res_index = index
    return res_index


def fp_slice(store, shape, start, stop, axis, res_store, res_shape, stream):
    '''
    slice a contiguous memory space, now support two directions.
    -----------------------------
    Para:
    store: FixedPointStorage, the data to be sliced
    shape: TensorShapeStorage, the original shape of the storage
    start: int, the start index of the slice (included)
    end:   int, the end index of the slice(not included),
           if larger than the last index, concatencate it into the dim size
    axis:  0 or 1, 0 means cut it horizontally, 1 means cut it vertically
    stream: the current stream of the task, not used now
    -----------------------------
    Return:
    res_store, res_shape, FixedPointStorage, TensorShapeStorage
    Raise:
        PermissionError: if the input start/stop/axis is not valid
    '''
    src_fpn = store.bigint_storage
    src_base = store.base_storage
    src_exp = store.exp_storage
    fpn_shape_tuple = shape.to_tuple()
    dim0, dim1 = 0, 0
    '''handle shape and index'''
    if len(fpn_shape_tuple) == 0:
        raise PermissionError("Cannot slice 0 dim!")
    elif len(fpn_shape_tuple) == 1:
        dim0, dim1 = 1, fpn_shape_tuple[0]
        if axis == 0:
            raise PermissionError("Cannot slice 1 dim horizontally!")
        start = _index_reset(start, dim1)
        stop = _index_reset(stop, dim1)
    elif len(fpn_shape_tuple) == 2:
        dim0, dim1 = fpn_shape_tuple[0], fpn_shape_tuple[1]
        if axis == 0:
            start = _index_reset(start, dim0)
            stop = _index_reset(stop, dim0)
        if axis == 1:
            start = _index_reset(start, dim1)
            stop = _index_reset(stop, dim1)
    else:
        raise PermissionError("Invalid shape")
    # handle condition that a[k: l] k>=l for 2-d array
    # will cause the result shape to be (0, dim1)
    if axis == 0 and start >= stop:
        res_fpn, res_base, res_exp = None, None, None
        return _fp_init_ss(
            None,
            res_fpn,
            res_base,
            res_exp,
            0,
            store.encode_n,
            store.encode_max_int,
            None,
            (0, dim1),
            store.mem_type,
            store.data_type,
        )
    # handle condition that a[:,k:l] k>=l for 2-d array
    # will cause the result shape to be (dim0, 0)
    if axis == 1 and start >= stop:
        res_fpn, res_base, res_exp = None, None, None
        res_shape_tuple = (dim0, 0) if len(fpn_shape_tuple) == 2 else (0,)
        return _fp_init_ss(
            None,
            res_fpn,
            res_base,
            res_exp,
            0,
            store.encode_n,
            store.encode_max_int,
            None,
            res_shape_tuple,
            store.mem_type,
            store.data_type,
        )
        # handle the normal slice
    res_shape_tuple, vec_size = (), 0
    '''useful paras'''
    bigint_row_bytelen = dim1 * PLAIN_BYTE
    uint32_row_bytelen = dim1 * U_INT32_BYTE
    gap_length = stop - start
    # start slice
    if axis == 1:
        'axis == 1 means that we need to cut the matrix vertically'
        res_bigint_row_bytelen = gap_length * PLAIN_BYTE
        res_uint32_row_bytelen = gap_length * U_INT32_BYTE
        if res_store is None:
            res_fpn = FPGA_LIB.c_malloc(
                c_size_t(res_bigint_row_bytelen * dim0))
            res_base = FPGA_LIB.c_malloc(
                c_size_t(res_uint32_row_bytelen * dim0))
            res_exp = FPGA_LIB.c_malloc(
                c_size_t(res_uint32_row_bytelen * dim0))
        else:
            res_fpn = res_store.bigint_storage
            res_base = res_store.base_storage
            res_exp = res_store.exp_storage
        # call the raw function
        FPGA_LIB.slice_vertical(
            c_char_p(src_fpn),
            c_void_p(src_base),
            c_void_p(src_exp),
            c_char_p(res_fpn),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_size_t(dim0),
            c_size_t(dim1),
            c_size_t(start),
            c_size_t(stop),
            c_size_t(PLAIN_BITS),
            c_uint32(0),
        )
        if len(fpn_shape_tuple) == 1:
            res_shape_tuple = (gap_length,)
            vec_size = res_shape_tuple[0]
        else:
            res_shape_tuple = (dim0, gap_length)
            vec_size = res_shape_tuple[0] * res_shape_tuple[1]

    elif axis == 0:
        'axis == 0 means that we nned to cut the matrix horizontally'
        if res_store is None:
            res_fpn = FPGA_LIB.c_malloc(
                c_size_t(bigint_row_bytelen * gap_length))
            res_base = FPGA_LIB.c_malloc(
                c_size_t(uint32_row_bytelen * gap_length))
            res_exp = FPGA_LIB.c_malloc(
                c_size_t(uint32_row_bytelen * gap_length))
        else:
            res_fpn = res_store.bigint_storage
            res_base = res_store.base_storage
            res_exp = res_store.exp_storage
        FPGA_LIB.slice_horizontal(
            c_char_p(src_fpn),
            c_void_p(src_base),
            c_void_p(src_exp),
            c_char_p(res_fpn),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_size_t(dim0),
            c_size_t(dim1),
            c_size_t(start),
            c_size_t(stop),
            c_size_t(PLAIN_BITS),
            c_uint32(0),
        )
        res_shape_tuple = (gap_length, dim1)
        vec_size = res_shape_tuple[0] * res_shape_tuple[1]
    else:
        raise NotImplementedError("Only support 2 dimensional slice")

    return _fp_init_ss(
        res_store,
        res_fpn,
        res_base,
        res_exp,
        vec_size,
        store.encode_n,
        store.max_int,
        res_shape,
        res_shape_tuple,
        store.mem_type,
        store.data_type,
    )


def pi_slice(store, shape, start, stop, axis, res_store, res_shape, stream):
    '''
    slice a contiguous memory space, now support two directions.
    -----------------------------
    Para:
    store: PaillierEncryptedStorage, the data to be sliced
    shape: TensorShapeStorage, the original shape of the storage
    start: int, the start index of the slice (included)
    end:   int, the end index of the slice(not included),
           if it is larger than the last index, then it concatencate into the dim size
    axis:  0 or 1, 0 means cut it horizontally, 1 means cut it vertically
    stream: the current stream of the task, not used now
    -----------------------------
    Return:
    res_store, res_shape, PaillierEncryptedStorage, TensorShapeStorage
    '''
    src_pen = store.pen_storage
    src_base = store.base_storage
    src_exp = store.exp_storage
    # get the two dims and check for illegal status
    pen_shape_tuple = shape.to_tuple()
    dim0, dim1 = 0, 0
    if len(pen_shape_tuple) == 0:
        raise PermissionError("Cannot slice 0 dim!")
    elif len(pen_shape_tuple) == 1:
        dim0, dim1 = 1, pen_shape_tuple[0]
        if axis == 0:
            raise PermissionError("Cannot slice 1 dim horizontally!")
        start = _index_reset(start, dim1)
        stop = _index_reset(stop, dim1)
    elif len(pen_shape_tuple) == 2:
        dim0, dim1 = pen_shape_tuple[0], pen_shape_tuple[1]
        if axis == 0:
            start = _index_reset(start, dim0)
            stop = _index_reset(stop, dim0)
        if axis == 1:
            start = _index_reset(start, dim1)
            stop = _index_reset(stop, dim1)
    else:
        raise PermissionError("Invalid shape")

    # handle condition that a[k, l], k>=l for 2-d array
    # will cause the result shape to be (0, dim1)
    if axis == 0 and start >= stop:
        res_pen, res_base, res_exp = None, None, None
        return _pi_init_ss(
            None,
            res_pen,
            res_base,
            res_exp,
            0,
            None,
            (0, dim1),
            store.mem_type,
            store.data_type,
            store.encode_n,
            store.encode_max_int,
        )
    # handle condition that a[:, k, l] k>=l for 2-d array
    # will cause the result shape to be (dim0, 0)
    if axis == 1 and start >= stop:
        res_pen, res_base, res_exp = None, None, None
        res_shape_tuple = (dim0, 0) if len(pen_shape_tuple) == 2 else (0,)
        return _pi_init_ss(
            None,
            res_pen,
            res_base,
            res_exp,
            0,
            None,
            res_shape_tuple,
            store.mem_type,
            store.data_type,
            store.encode_n,
            store.encode_max_int,
        )
    # handle the normal slice
    res_shape_tuple = ()
    vec_size = 0
    '''useful paras'''
    bigint_row_bytelen = dim1 * PLAIN_BYTE
    uint32_row_bytelen = dim1 * U_INT32_BYTE
    gap_length = stop - start
    # start slice
    if axis == 1:
        'axis == 1 means that we need to cut the matrix vertically'
        res_bigint_row_bytelen = gap_length * PLAIN_BYTE
        res_uint32_row_bytelen = gap_length * U_INT32_BYTE
        # malloc space for result
        if res_store is None:
            res_pen = FPGA_LIB.c_malloc(
                c_size_t(res_bigint_row_bytelen * dim0))
            res_base = FPGA_LIB.c_malloc(
                c_size_t(res_uint32_row_bytelen * dim0))
            res_exp = FPGA_LIB.c_malloc(
                c_size_t(res_uint32_row_bytelen * dim0))
        else:
            res_pen = res_store.bigint_storage
            res_base = res_store.base_storage
            res_exp = res_store.exp_storage
        # call the raw function
        FPGA_LIB.slice_vertical(
            c_char_p(src_pen),
            c_void_p(src_base),
            c_void_p(src_exp),
            c_char_p(res_pen),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_size_t(dim0),
            c_size_t(dim1),
            c_size_t(start),
            c_size_t(stop),
            c_size_t(CIPHER_BITS),
            c_uint32(0),
        )
        if len(pen_shape_tuple) == 1:
            res_shape_tuple = (gap_length,)
            vec_size = res_shape_tuple[0]
        else:
            res_shape_tuple = (dim0, gap_length)
            vec_size = res_shape_tuple[0] * res_shape_tuple[1]

    elif axis == 0:
        'axis == 0 means that we nned to cut the matrix horizontally'
        if res_store is None:
            res_pen = FPGA_LIB.c_malloc(
                c_size_t(bigint_row_bytelen * gap_length))
            res_base = FPGA_LIB.c_malloc(
                c_size_t(uint32_row_bytelen * gap_length))
            res_exp = FPGA_LIB.c_malloc(
                c_size_t(uint32_row_bytelen * gap_length))
        else:
            res_pen = res_store.bigint_storage
            res_base = res_store.base_storage
            res_exp = res_store.exp_storage
        FPGA_LIB.slice_horizontal(
            c_char_p(src_pen),
            c_void_p(src_base),
            c_void_p(src_exp),
            c_char_p(res_pen),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_size_t(dim0),
            c_size_t(dim1),
            c_size_t(start),
            c_size_t(stop),
            c_size_t(CIPHER_BITS),
            c_uint32(0),
        )
        # since 1-dim shape will not occur here, result shape is always 2-D
        res_shape_tuple = (gap_length, dim1)
        vec_size = res_shape_tuple[0] * res_shape_tuple[1]
    else:
        raise NotImplementedError()

    return _pi_init_ss(
        res_store,
        res_pen,
        res_base,
        res_exp,
        vec_size,
        res_shape,
        res_shape_tuple,
        store.mem_type,
        store.data_type,
        store.encode_n,
        store.encode_max_int,
    )


def bi_p2c(data, res, bytelen=CIPHER_BYTE):
    '''
    copy data to the C memory pointed to by res
    -------------------
    Para:
        data: List[object], each object is a bigint CIPHER_BIT long
        res:  int, actually a pointer pointing to C memory
    Return:
        None, but the contents in c_void_p(res) has been changed
    '''
    for i in range(len(data)):
        src_number = data[i].to_bytes(bytelen, 'little')
        FPGA_LIB.bigint_set(
            c_char_p(res), c_char_p(src_number), c_size_t(bytelen), c_size_t(i)
        )


def bi_gen_rand(elem_size, count, res, rand_seed, stream):
    '''
    generate random bigint for pi_obfuscation
    ------------------
    Para:
        elem_size: int, length of random bigint, upper bound is CIPHER_BYTE
        count:     int, number of random bigint to be generated
        res:       BigintStorage, the return value
        rand_seed: seed used for generating random data
    Return:
        BigintStorage, same as res
    '''
    # Didn't use vectorize since that we need to_bytes()
    # But ndarray_float65 has no to_bytes method
    random.seed(rand_seed)
    rands = np.array([random.randrange(1, 8 ** elem_size) for i in range(count)])
    if res is None:
        data_storage = FPGA_LIB.c_malloc(c_size_t(count * CIPHER_BYTE))
    else:
        data_storage = res.bigint_storage
    # CIPHER_BYTE is the upper bound of the length of the rand number
    '''
    We assume that the store of random bigint is on FPGA device_0
    TODO: Add configuration for choosing divice
    '''
    bi_p2c(rands, data_storage)
    return _bi_init_store(
        res,
        data_storage,
        count,
        mem_type=MEM_FPGA_NUM_0,
        elem_size=CIPHER_BYTE)


def __get_shape_size(shape_tuple):
    shape_size = 1
    if len(shape_tuple) == 1:
        shape_size = shape_tuple[0]
    elif len(shape_tuple) == 2:
        shape_size = shape_tuple[0] * shape_tuple[1]
    else:
        raise PermissionError("Invalid Shape Tuple")

    return shape_size


def pi_reshape(store, shape, new_shape, res_store, res_shape, stream):
    '''
    Change a PaillierEcnryptedStorage's shape.
    No need for change the continuous storage, only change the shape.
    -------------------
    Paras:
        store, shape:  PaillierEncryptedStorage, TensorShapeStorage
        new_shape:     TensorShapeStorage, the new shape for the pi_storage
    Returns:
        tuple: (PaillierEncryptedStorage, TensorShapeStorage)
    Raise:
        ValueError:    If shape and new_shape's size is unequal
    '''
    res_shape_tuple = new_shape.to_tuple()
    old_shape_tuple = shape.to_tuple()
    res_shape_size = __get_shape_size(res_shape_tuple)
    old_shape_size = __get_shape_size(old_shape_tuple)
    res_vec_size = store.vec_size
    if res_shape_size != old_shape_size:
        raise ValueError("total size of new array must be unchanged!")
    # Still, we do a malloc and memcpy in order to avoid double free in python
    if res_store is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(CIPHER_BYTE * res_vec_size))
        res_base = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * res_vec_size))
        res_exp = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * res_vec_size))
    else:
        res_pen = res_store.pen_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage

    FPGA_LIB.c_memcpy(
        c_void_p(res_pen),
        c_void_p(store.pen_storage),
        c_size_t(CIPHER_BYTE * res_vec_size),
    )
    FPGA_LIB.c_memcpy(
        c_void_p(res_base),
        c_void_p(store.base_storage),
        c_size_t(U_INT32_BYTE * res_vec_size),
    )
    FPGA_LIB.c_memcpy(
        c_void_p(res_exp),
        c_void_p(store.exp_storage),
        c_size_t(U_INT32_BYTE * res_vec_size),
    )

    return _pi_init_ss(
        res_store,
        res_pen,
        res_base,
        res_exp,
        store.vec_size,
        res_shape,
        res_shape_tuple,
        store.mem_type,
        store.data_type,
        store.encode_n,
        store.encode_max_int,
    )


def fp_cat(stores, shapes, axis, res_store, res_shape):
    '''
    concat several FixedPointStorage according to axis
    --------------------
    Para:
        stores: List or ndarray, elements are FixedPointStorage
        shapes: List or ndarray, elements are TensorShapeStorage
        axis:   int, how stores will be stacked
                    0 means a vertical stack, stack along 1st dim
                    1 means a horizontal stack, stack along 2nd dim
        res_store: FixedPointStorage, the stacked result
        res_shape: TensorShapeStorage, the result's shape
    Return:
        tuple, (FixedPointStorage, TensorShapeStorage)
    Raise:
        PermissionError: Invalid input data or invalid shape
        NotImplementedError: Current only support at most 2-D matrix
    '''
    stores = list(stores)
    shapes = list(shapes)
    num_stores = len(stores)
    res_vec_size = np.sum([v.vec_size for v in stores])
    # Abnormal checks
    if num_stores < 2:
        raise PermissionError("At least 2 Storages required for concatenation")
    if len(shapes) != num_stores:
        raise PermissionError(
            "The number of storages and that of shapes didn't match")
    for v in stores:
        if v.data_type != stores[0].data_type:
            raise PermissionError(
                "All storages should have the same data type")
        if v.encode_n != stores[0].encode_n:
            raise PermissionError("All storages should have the same n")
        if v.max_int != stores[0].max_int:
            raise PermissionError("All storages should have the same max_int")
        if v.mem_type != stores[0].mem_type:
            raise PermissionError(
                "All storages should have the same memory type")
    # num_rows, num_cols is the data demanded by C functions
    # res_rows, res_cols are return values that should be same as numpy's output
    # distinguish them such that upper and lower level won't bother each other
    if axis == 0:
        first_shape_decomposed = __shape_decompose(shapes[0])
        num_rows, num_cols = 0, first_shape_decomposed[1]
        for v in shapes:
            shape_tuple = __shape_decompose(v)
            if shape_tuple[1] != num_cols:
                raise PermissionError("Shapes didn't align")
            num_rows += shape_tuple[0]
        res_rows = num_rows
        res_cols = num_cols
    elif axis == 1:
        first_shape = shapes[0].to_tuple()
        if len(first_shape) <= 1:
            num_rows, num_cols = 1, 0
            for v in shapes:
                if len(v.to_tuple()) == 0:
                    num_cols += 1
                if len(v.to_tuple()) == 1:
                    num_cols += v.to_tuple()[0]
                if len(v.to_tuple()) >= 2:
                    raise PermissionError("Shape cannot align!!!")
            res_rows = num_cols
            res_cols = None
        elif len(first_shape) == 2:
            num_rows, num_cols = first_shape[0], 0
            for v in shapes:
                v_shape = v.to_tuple()
                if len(v_shape) != 2 or num_rows != v_shape[0]:
                    raise PermissionError("Shape cannot align!")
                num_cols += v_shape[1]
            res_rows = num_rows
            res_cols = num_cols
        else:
            raise NotImplementedError("Now only support up to 2-D array")
    else:
        raise PermissionError("Invalid Axis")
    res_shape = TensorShapeStorage(res_rows, res_cols)

    fpn_pointers = [c_void_p(v.bigint_storage) for v in stores]
    base_pointers = [c_void_p(v.base_storage) for v in stores]
    exp_pointers = [c_void_p(v.exp_storage) for v in stores]

    if res_store is None:
        res_fpn = FPGA_LIB.c_malloc(c_size_t(PLAIN_BYTE * res_vec_size))
        res_base = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * res_vec_size))
        res_exp = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * res_vec_size))
    else:
        res_fpn = res_store.bigint_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage

    fpn_arr = (c_void_p * num_stores)(*fpn_pointers)
    base_arr = (c_void_p * num_stores)(*base_pointers)
    exp_arr = (c_void_p * num_stores)(*exp_pointers)
    vec_sizes = (c_uint32 * num_stores)(*[v.vec_size for v in stores])

    if axis == 0:
        '''means that we should cat stores vertically'''
        FPGA_LIB.vstack(
            fpn_arr,
            base_arr,
            exp_arr,
            c_void_p(res_fpn),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_uint64(num_stores),
            vec_sizes,
            c_uint64(res_cols),
            c_size_t(PLAIN_BITS),
        )
    elif axis == 1:
        '''means that we should cat stores horizontally'''
        FPGA_LIB.hstack(
            fpn_arr,
            base_arr,
            exp_arr,
            c_void_p(res_fpn),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_uint64(num_stores),
            vec_sizes,
            c_uint64(res_rows),
            c_size_t(PLAIN_BITS),
        )
        # raise NotImplementedError()
    else:
        raise NotImplementedError()

    return _fp_init_ss(
        res_store,
        res_fpn,
        res_base,
        res_exp,
        int(round(res_vec_size)),
        stores[0].encode_n,
        stores[0].max_int,
        res_shape,
        res_shape.to_tuple(),
        stores[0].mem_type,
        stores[0].data_type,
    )


def pi_cat(stores, shapes, axis, res_store, res_shape):
    '''
    concat several PaillierEncryptedStorage according to axis
    --------------------
    Para:
        stores: List or ndarray, elements are PaillierEncryptedStorage
        shapes: List or ndarray, elements are TensorShapeStorage
        axis:   int, how stores will be stacked
                    0 means a vertical stack, stack along 1st dim
                    1 means a horizontal stack, stack along 2nd dim
        res_store: PaillierEncryptedStorage, the stacked result
        res_shape: TensorShapeStorage, the result's shape
    Return:
        tuple, (PaillierEncryptedStorage, TensorShapeStorage)
    Raise:
        PermissionError: Invalid input data or invalid shape
        NotImplementedError: Current only support at most 2-D matrix
    '''
    stores = list(stores)
    shapes = list(shapes)
    num_stores = len(stores)
    res_vec_size = np.sum([v.vec_size for v in stores])

    # Anomaly checks
    if num_stores < 2:
        raise PermissionError("At least 2 Storages required for concatenation")
    if len(shapes) != num_stores:
        raise PermissionError(
            "The number of storages and that of shapes didn't match")
    for v in stores:
        if v.data_type != stores[0].data_type:
            raise PermissionError(
                "All storages should have the same data type")
        if v.encode_n != stores[0].encode_n:
            raise PermissionError("All storages should have the same n")
        if v.encode_max_int != stores[0].encode_max_int:
            raise PermissionError("All storages should have the same max_int")
        if v.mem_type != stores[0].mem_type:
            raise PermissionError(
                "All storages should have the same memory type")
    # num_rows, num_cols is the data demanded by C functions
    # res_rows, res_cols are return values that should be same as numpy's output
    # distinguish them so upper and lower level won't bother each other
    if axis == 0:
        first_shape_decomposed = __shape_decompose(shapes[0])
        num_rows, num_cols = 0, first_shape_decomposed[1]
        for v in shapes:
            shape_tuple = __shape_decompose(v)
            if shape_tuple[1] != num_cols:
                raise PermissionError("Shapes didn't align")
            num_rows += shape_tuple[0]
        res_rows = num_rows
        res_cols = num_cols
    elif axis == 1:
        '''the horizontal cat'''
        first_shape = shapes[0].to_tuple()
        if len(first_shape) <= 1:
            num_rows = 1
            num_cols = 0
            for v in shapes:
                if len(v.to_tuple()) == 0:
                    num_cols += 1
                if len(v.to_tuple()) == 1:
                    num_cols += v.to_tuple()[0]
                if len(v.to_tuple()) >= 2:
                    raise PermissionError("Shape cannot align!!!")
            res_rows = num_cols
            res_cols = None
            print(num_rows, num_cols, res_rows, res_cols)
        elif len(first_shape) == 2:
            num_rows = first_shape[0]
            num_cols = 0
            for v in shapes:
                v_shape = v.to_tuple()
                if len(v_shape) != 2 or num_rows != v_shape[0]:
                    raise PermissionError("Shape cannot align!")
                # num_rows += v_shape[0]
                num_cols += v_shape[1]
            res_rows = num_rows
            res_cols = num_cols
        else:
            raise NotImplementedError("Now only support up to 2-D array")
    else:
        raise PermissionError("Invalid Axis")
    res_shape = TensorShapeStorage(res_rows, res_cols)

    pen_pointers = [c_void_p(v.pen_storage) for v in stores]
    base_pointers = [c_void_p(v.base_storage) for v in stores]
    exp_pointers = [c_void_p(v.exp_storage) for v in stores]
    # print(res_vec_size)
    if res_store is None:
        res_pen = FPGA_LIB.c_malloc(c_size_t(CIPHER_BYTE * res_vec_size))
        res_base = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * res_vec_size))
        res_exp = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * res_vec_size))
    else:
        res_pen = res_store.pen_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage

    pen_arr = (c_void_p * num_stores)(*pen_pointers)
    base_arr = (c_void_p * num_stores)(*base_pointers)
    exp_arr = (c_void_p * num_stores)(*exp_pointers)
    vec_sizes = (c_uint32 * num_stores)(*[v.vec_size for v in stores])

    if axis == 0:
        FPGA_LIB.vstack(
            pen_arr,
            base_arr,
            exp_arr,
            c_void_p(res_pen),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_uint64(num_stores),
            vec_sizes,
            c_uint64(num_cols),
            c_size_t(CIPHER_BITS),
        )
    elif axis == 1:
        FPGA_LIB.hstack(
            pen_arr,
            base_arr,
            exp_arr,
            c_void_p(res_pen),
            c_void_p(res_base),
            c_void_p(res_exp),
            c_uint64(num_stores),
            vec_sizes,
            c_uint64(num_rows),
            c_size_t(CIPHER_BITS),
        )
    else:
        raise NotImplementedError()

    return _pi_init_ss(
        res_store,
        res_pen,
        res_base,
        res_exp,
        int(res_vec_size),
        res_shape,
        res_shape.to_tuple(),
        stores[0].mem_type,
        stores[0].data_type,
        stores[0].encode_n,
        stores[0].encode_max_int,
    )


def random_p2c(rands, bitlen, size):
    bytelen = bit_change(bitlen, 1) // 8
    data_storage = FPGA_LIB.c_malloc(c_size_t(size * bytelen))
    bi_p2c(rands, data_storage)
    return _bi_init_store(
        None, data_storage, size, mem_type=MEM_FPGA_NUM_0, elem_size=bytelen
    )


def random_c2p(random_store: BigIntStorage, size):
    bytelen = random_store.elem_size
    random_res = c_buffer(bytelen)
    res_list = []
    for i in range(size):
        FPGA_LIB.c_memcpy(
            cast(random_res, c_void_p),
            c_void_p(random_store.bigint_storage + i * bytelen),
            c_size_t(bytelen),
        )
        temp_int = int.from_bytes(random_res.raw, 'little')
        res_list.append(temp_int)
    return res_list


class Hash_key_storage:
    '''
    parameters:
    hash_storage: int, address of C memory storing the big integer
    '''

    def __init__(self, hash_storage):
        self.hash_storage = hash_storage

    def __del__(self):
        hash_free(self.hash_storage)
        self.hash_storage = None


def hash_free(hash_key):
    FPGA_LIB.c_free(c_void_p(hash_key))


def hash_p2c(data, bitlen):
    '''convert the data into Hash_key_storage,
    since all data is identically bitlen, no value/index is needed'''
    if isinstance(data, list):
        data = np.asarray(data)
    if not isinstance(data, np.ndarray):
        raise TypeError("Unsupported Data Structure")
    bytelen = bitlen // 8
    elem_cnt = data.size
    data_ptr = FPGA_LIB.c_malloc(c_size_t(bytelen * elem_cnt))
    for i in range(elem_cnt):
        try:
            FPGA_LIB.c_memcpy(
                c_void_p(data_ptr + i * bytelen),
                c_char_p(data[i].to_bytes(bytelen, "little")),
                c_size_t(bytelen),
            )
        except AttributeError:
            raise AttributeError("Only support int type!!!")
        except BaseException:
            raise RuntimeError("Running c memory copy failed!!")

    return Hash_key_storage(data_ptr)


def hash_c2p(store: Hash_key_storage, size, bitlength):
    '''
    transform the rsa computing result into a python dictionary
    seperate from the sha_c2p due to big endian and small endian difference, especially when there is zero
    '''
    bytelen = bitlength // 8
    sha_res = c_buffer(bytelen)
    res_list = []
    for i in range(size):
        FPGA_LIB.c_memcpy(
            cast(sha_res, c_void_p),
            c_void_p(store.hash_storage + i * bytelen),
            c_size_t(bytelen),
        )
        temp_int = int.from_bytes(sha_res.raw, 'little')
        res_list.append(temp_int)
    return res_list


def rsa_c2bytes(storage: Hash_key_storage, size, bitlength):
    store_size = bitlength // 8 * size
    bytes = c_buffer(store_size)
    FPGA_LIB.c_memcpy(
        cast(
            bytes, c_void_p), c_void_p(
            storage.hash_storage), c_size_t(store_size))
    return bytes.raw


def rsa_bytes2c(bytes, size, bitlength):
    store_size = bitlength // 8 * size
    hash_key = FPGA_LIB.c_malloc(c_size_t(store_size))
    FPGA_LIB.c_memcpy(
        c_void_p(hash_key),
        c_char_p(bytes),
        c_size_t(store_size))
    return Hash_key_storage(hash_key)


def hash_bit_inquiry(hash_method):
    dist_encode_function = {
        "md5": 256,
        "sha1": 256,
        "sha224": 256,
        "sha256": 256,
        "sha384": 512,
        "sha512": 512,
        "sm3": 256,
        "none": 256,
    }
    return dist_encode_function[hash_method]


def gmp_gen_rand(bit_len, vec_size, n):
    RSA_bitlength = bit_change(n, 0)
    output_bitlength = bit_change(bit_len, 1)
    random_bytelength = output_bitlength // 8
    if output_bitlength > RSA_bitlength:
        raise PermissionError(
            f"bitlength should be smaller than the given size {RSA_bitlength}"
        )

    res_rand = FPGA_LIB.c_malloc(c_size_t(vec_size * random_bytelength))
    FPGA_LIB.gmp_random(
        c_char_p(res_rand),
        c_size_t(bit_len),
        c_size_t(output_bitlength),
        c_size_t(RSA_bitlength),
        c_size_t(vec_size),
        c_char_p(n.to_bytes(RSA_bitlength // 8, 'little')),
    )

    return _bi_init_store(
        None,
        res_rand,
        vec_size,
        mem_type=MEM_FPGA_NUM_0,
        elem_size=random_bytelength)


def compute_hash(key_number, hash_method, hash_bitlength, size, salt):
    hash_bytelength = hash_bitlength // 8
    hash_storage = FPGA_LIB.c_malloc(c_size_t(hash_bytelength * size))
    if isinstance(key_number, Hash_key_storage):
        key_length_storage = FPGA_LIB.c_malloc(c_size_t(INT64_BYTE * size))
        FPGA_LIB.hex_to_int(
            c_void_p(key_number.hash_storage),
            c_uint32(hash_bytelength),
            c_size_t(size),
            c_void_p(hash_storage),
            c_void_p(key_length_storage),
        )
    else:
        key_storage, key_length_storage = keyid_p2c(key_number, salt)

    FPGA_LIB.computeSHA256_index(
        c_void_p(key_storage),
        c_void_p(key_length_storage),
        c_size_t(size),
        c_void_p(hash_storage),
    )
    return Hash_key_storage(hash_storage)


def keyid_p2c(data, salt):
    '''
    Change the input list into a SHA_storage

    Parameters:
    ------------------
    data, list or ndarray, contains a butch of id
        we assume that each id should be a string rather than a int or something else
    '''
    # preprocess
    if isinstance(data, list):
        data = np.asarray(data)
    if not isinstance(data, np.ndarray):
        raise TypeError("Unsupported Data Structure")
    # malloc the space
    str_len = 0
    vec_len = []
    for x in data:
        x = str(x) + salt
        str_len += len(x)
        vec_len.append(len(x))
    hash_key = FPGA_LIB.c_malloc(c_size_t(str_len))
    # then we should feed all the strings into this place
    index = 0
    for i in range(len(data)):
        FPGA_LIB.c_memcpy(
            c_void_p(hash_key + index),
            c_char_p(bytes(str(data[i]), encoding="utf-8")),
            c_size_t(vec_len[i]),
        )
        index += vec_len[i]
    # then the vec_len should also be changed to as a pointer
    vec_len = np.asarray(vec_len).astype(np.int64)
    vec_size = vec_len.size
    length_storage_ptr = FPGA_LIB.c_malloc(c_size_t(vec_size * INT64_BYTE))
    len_ptr = vec_len.ctypes.data_as(c_void_p)
    FPGA_LIB.c_memcpy(
        c_void_p(length_storage_ptr), len_ptr, c_size_t(vec_size * INT64_BYTE)
    )

    # switch the differnt data type
    return hash_key, length_storage_ptr


def bit_change(raw_number, type):
    if type == 0:
        bitlength = math.log(raw_number, 2)
    else:
        bitlength = raw_number
    if bitlength > 4096:
        raise PermissionError("Invalid Data range for FPGA")
    if bitlength > 2048:
        return 4096
    if bitlength > 1024:
        return 2048
    if bitlength > 512:
        return 1024
    if bitlength > 256:
        return 512
    return 256


def rsa_pubkey_id_process(
        random: BigIntStorage,
        exponent,
        modulus,
        hash: Hash_key_storage,
        hash_length,
        size):
    if size != random.vec_size:
        raise PermissionError(
            f"The size of random vector {random.vec_size} does not equal to size of hash {size}"
        )
    exp_length = bit_change(exponent, 0)
    modulus_length = bit_change(modulus, 0)
    res_rsa = FPGA_LIB.c_malloc(c_size_t(size * modulus_length // 8))
    exp_ptr = c_char_p(exponent.to_bytes(exp_length // 8, 'little'))
    modulus_ptr = c_char_p(modulus.to_bytes(modulus_length // 8, 'little'))

    FPGA_LIB.rsa_pubkey_id_process(
        c_char_p(random.bigint_storage),
        c_char_p(hash.hash_storage),
        modulus_ptr,
        exp_ptr,
        c_char_p(res_rsa),
        c_size_t(size),
        c_size_t(modulus_length),
        c_size_t(exp_length),
        c_size_t(hash_length),
        c_size_t(8 * random.elem_size),
        c_size_t(0),
    )

    return Hash_key_storage(res_rsa)


def rsa_powmod(hash: Hash_key_storage, exponent, modulus, hash_length, size):
    exp_length = bit_change(exponent, 0)
    modulus_length = bit_change(modulus, 0)
    res_rsa = FPGA_LIB.c_malloc(c_size_t(size * modulus_length // 8))
    exp_ptr = c_char_p(exponent.to_bytes(exp_length // 8, 'little'))
    modulus_ptr = c_char_p(modulus.to_bytes(modulus_length // 8, 'little'))

    FPGA_LIB.rsa_powmod(
        c_char_p(hash.hash_storage),
        exp_ptr,
        modulus_ptr,
        c_char_p(res_rsa),
        c_size_t(size),
        c_size_t(hash_length),
        c_size_t(exp_length),
        c_size_t(modulus_length),
        c_size_t(0),
    )

    return Hash_key_storage(res_rsa)


def rsa_divm(
        hash: Hash_key_storage,
        random: BigIntStorage,
        rsa_n,
        hash_bit,
        size):
    if size != random.vec_size:
        raise PermissionError(
            f"The size of random vector {random.vec_size} does not equal to that of hash {size}"
        )

    modulus_length = bit_change(rsa_n, 0)
    if modulus_length != hash_bit:
        raise PermissionError(
            f"The biglength of hash value from host {hash_bit} does not equal to that of key {modulus_length}"
        )

    modulus_ptr = c_char_p(rsa_n.to_bytes(modulus_length // 8, 'little'))
    res_rsa = FPGA_LIB.c_malloc(c_size_t(size * modulus_length // 8))
    FPGA_LIB.RSA_divmod(
        c_char_p(hash.hash_storage),
        c_char_p(random.bigint_storage),
        modulus_ptr,
        c_char_p(res_rsa),
        c_size_t(size),
        c_size_t(hash_bit),
        c_size_t(8 * random.elem_size),
    )

    return Hash_key_storage(res_rsa)


def fp_align(store, res_store, stream):
    '''
    Perform alignment for elements in a FixedPointStorage.
    ------------------
    Paras:
        store: FixedPointStorage
    Return:
        res_store: FixedPointStorage
    '''
    vec_size = store.vec_size
    # the src_store data
    src_fpn = store.bigint_storage
    src_base = store.base_storage
    src_exp = store.exp_storage
    # malloc space for the return value
    if res_store is None:
        res_fpn = FPGA_LIB.c_malloc(c_size_t(vec_size * PLAIN_BYTE))
        res_base = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
        res_exp = FPGA_LIB.c_malloc(c_size_t(vec_size * U_INT32_BYTE))
    else:
        res_fpn = res_store.bigint_storage
        res_base = res_store.base_storage
        res_exp = res_store.exp_storage
    FPGA_dev_num = __get_FPGA_device_num(store.mem_type)
    FPGA_LIB.fpn_align(
        c_char_p(src_fpn),
        c_void_p(src_base),
        c_void_p(src_exp),
        c_char_p(res_fpn),
        c_void_p(res_base),
        c_void_p(res_exp),
        c_size_t(vec_size),
        c_char_p(store.encode_n.to_bytes(PLAIN_BYTE, 'little')),
        c_size_t(CIPHER_BITS),
        c_size_t(FPGA_dev_num),
    )
    # # handle data_type
    # data_type = 0
    # if store.data_type == INT64_TYPE:
    #     data_type = INT64_TYPE
    # else:
    #     data_type = FLOAT_TYPE
    return _fp_init_store(
        res_store,
        res_fpn,
        res_base,
        res_exp,
        vec_size,
        store.encode_n,
        store.max_int,
        store.mem_type,
        store.data_type,
    )


def pi_sum_multi_index(
        pub_key, store, valid_index, node_id, node_num, min_value=0, max_value=None
):
    '''
    Run sum for data with the same index indicated in the valid_index list
    Return: A PEN_Storage class with max_value-min_value+1 number of PEN values
    ------------
    Parameters:
        pub_key: PubKeyStorage
        store:   PaillierEncryptedStorage, the original PEN_storage class
        valid_index:  ndarray, contains indices like [-1, 1, 2, 1, 3, 3, 2, -1] for each instance,
                        -1 means that this value will not be calculated if min_value >= 0
                        1,2,3 means the different groups that it belongs to
        node_id:      ndarray, contains node_id like [3, 1, 0, 2, 3, 1] for each instance.
                        0,1,2 represent the node that current instance locates in
        node_num:     int, number of nodes
        min_value:    int, The min valid value of the valid index, default 0,
                           in the above example, if min_value == 1, then -1 will be invalid
                           if min_value == -1, -1 is also valid
        max_value:    int, The max valid value of the valid index
    Return:
        tuple   (PaillierEncryptedStorage, TensorShapeStorage)
    '''
    src_pen = store.pen_storage
    src_base = store.base_storage
    src_exp = store.exp_storage
    vec_size = store.vec_size
    valid_store = te_p2c(valid_index, None)
    node_id_store = te_p2c(node_id, None)
    # set max_value to maximum number if it is not designated
    max_value = max(valid_index) if max_value is None else max_value
    index_num = max_value - min_value + 1
    res_size = index_num * node_num

    res_pen = FPGA_LIB.c_malloc(c_size_t(res_size * CIPHER_BYTE))
    res_base = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
    res_exp = FPGA_LIB.c_malloc(c_size_t(res_size * U_INT32_BYTE))
    res_shape_tuple = (node_num, index_num)
    FPGA_dev_num = __get_FPGA_device_num(store.mem_type)
    FPGA_LIB.pen_sum_with_multi_index(
        c_char_p(src_pen),
        c_void_p(src_base),
        c_void_p(src_exp),
        c_char_p(res_pen),
        c_void_p(res_base),
        c_void_p(res_exp),
        c_size_t(index_num),
        c_size_t(node_num),
        c_int64(min_value),
        c_void_p(valid_store.data),
        c_void_p(node_id_store.data),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_size_t(vec_size),
        c_size_t(CIPHER_BITS),
        c_uint32(FPGA_dev_num),
    )
    return _pi_init_ss(
        None,
        res_pen,
        res_base,
        res_exp,
        res_size,
        None,
        res_shape_tuple,
        MEM_HOST,
        store.data_type,
        store.encode_n,
        store.encode_max_int,
    )


def pi_accumulate(pub_key, store, shape):
    '''
    Perform acummulate add for a vector
    ----------------
    Paras:
        pub_key:     PubKeyStorage,
        left_store:  PaillierEncryptedStorage
        left_shape:  TensorShapeStorage
    Return:
        tuple:       (PaillierEncryptedStorage, TensorShapeStorage)
    '''
    src_pen = store.pen_storage
    src_base = store.base_storage
    src_exp = store.exp_storage
    vec_size = store.vec_size

    res_pen = FPGA_LIB.c_malloc(c_size_t(CIPHER_BYTE * vec_size))
    res_base = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * vec_size))
    res_exp = FPGA_LIB.c_malloc(c_size_t(U_INT32_BYTE * vec_size))
    res_shape_tuple = shape.to_tuple()
    if len(res_shape_tuple) == 1:
        res_shape_tuple = (1, res_shape_tuple[0])

    FPGA_LIB.gmp_accumulate(
        c_char_p(src_pen),
        c_void_p(src_base),
        c_void_p(src_exp),
        c_void_p(res_pen),
        c_void_p(res_base),
        c_void_p(res_exp),
        c_size_t(res_shape_tuple[0]),
        c_size_t(res_shape_tuple[1]),
        pub_key.n,
        pub_key.g,
        pub_key.nsquare,
        pub_key.max_int,
        c_size_t(CIPHER_BITS),
        0,
    )

    return _pi_init_ss(
        None,
        res_pen,
        res_base,
        res_exp,
        vec_size,
        None,
        res_shape_tuple,
        store.mem_type,
        store.data_type,
        store.encode_n,
        store.encode_max_int,
    )
