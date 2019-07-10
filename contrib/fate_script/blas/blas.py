import pandas as pd
import numpy as np
import uuid
import time
import json
from numbers import Number
from federatedml.secureprotol.fate_paillier import *
from federatedml.secureprotol.encrypt import *
from federatedml.util.param_checker import AllChecker


# noinspection PyInterpreter
class Tensor:
    def __init__(self, cipher, shape, store = None):
        self.cipher = cipher
        self.shape = shape
        self.store = store
        if self.shape is not None and len(self.shape) > 2:
            raise NotImplementedError("dimensions should less then 3")

    # noinspection PyInterpreter
    def transpose(self):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    def __matmul__(self, other):
        raise NotImplementedError()

    # scalar only
    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)
        else:
            raise NotImplementedError()

    def __rtruediv__(self, other):
        raise NotImplementedError()

    def __neg__(self):
        return self.__mul__(-1)

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __sub__(self, other):
        return self.__add__(-other)

    def decrypt(self, encrypt_operator):
        if self.cipher.get_privacy_key() is None:
            self.cipher.set_privacy_key(encrypt_operator.get_privacy_key())
        vfunc = np.vectorize(self.cipher.decrypt)
        return self.map(vfunc)

    def encrypt(self):
        vfunc = np.vectorize(self.cipher.encrypt)
        return self.map(vfunc)

    def __pow__(self, other):
        vfunc = np.vectorize(lambda x: x**2)
        return self.map(vfunc)

    def __str__(self):
       return str(self.store)

    @staticmethod
    def unwrap(other):
        if isinstance(other, Tensor) and not isinstance(other, TensorInPy):
            raise NotImplementedError()
        if isinstance(other, TensorInPy):
            tmp = other.store
        else:
            tmp = other
        return tmp


class TensorInPy(Tensor):
    def __init__(self, cipher, shape, store = None):
        super().__init__(cipher, shape, store)

    # noinspection PyInterpreter
    def transpose(self):
        store = self.store.transpose()
        return TensorInPy(self.cipher, store.shape, store)

    def __add__(self, other):
        # if other.shape != self.shape:
        #     raise ValueError("diff shape  %s , %s" % (other.shape, self.shape))
        return TensorInPy(self.cipher, self.shape, self.store + self.unwrap(other))

    def __mul__(self, other):
        if isinstance(other, TensorInEgg):
            return other.__mul__(self)
        store = self.store * self.unwrap(other)
        return TensorInPy(self.cipher, store.shape, store)

    def __matmul__(self, other):
        if isinstance(other, TensorInEgg):
            return other.__rmatmul__(self)
        store = self.store @ self.unwrap(other)
        return TensorInPy(self.cipher, store.shape, store)

    def __rtruediv__(self, other):
        store = other / self.store 
        return TensorInPy(self.cipher, store.shape, store)

    def hstack(self, other):
        store = np.hstack((self.store,other.store))
        return TensorInPy(self.cipher, store.shape, store)

    # todo: any ndim    
    def split(self, idx):
        store1 = self.store[:idx] 
        store2 = self.store[idx:] 
        return TensorInPy(self.cipher, store1.shape, store1), TensorInPy(self.cipher, store2.shape, store2)

    def map(self, func):
        # todo: is vectorize?
        store = func(self.store)
        return TensorInPy(self.cipher, store.shape, store)


class TensorInEgg(Tensor):
    def __init__(self, cipher, shape, store = None):
        super().__init__(cipher, shape, store)

    def __add__(self, other):
        # scalar only
        if not isinstance(other,TensorInEgg):
            store = self.store.mapValues(lambda x: x + self.unwrap(other))
        else:
            store = self.store.join(other.store, lambda x,y: x + y )
        return TensorInEgg(self.cipher, self.shape, store)

    def __mul__(self, other):
        if isinstance(other, TensorInEgg):
            return TensorInEgg(self.cipher, self.shape, self.store.join(other.store, lambda x,y: x*y))
        if not isinstance(other, TensorInPy):
            tmp = other
        # vector only
        elif len(other.store.shape) > 1:
            raise NotImplementedError()
            #tmp = other.store[0]
        else:
            tmp = other.store
        store = self.store.mapValues(lambda x: x * tmp)
        return TensorInEgg(self.cipher, self.shape, store)

    def __matmul__(self, other):
        # only matrix @ vector now
        if not isinstance(other, TensorInPy):
            raise NotImplementedError()
        import os
        store = self.store.mapValues(lambda x: x @ other.store)
        return TensorInEgg(self.cipher, [None, 1], store)

    def __rmatmul__(self, other):
        raise NotImplementedError()
        # only matrix @ vector now
        #if not isinstance(other, TensorInPy):
        #    raise NotImplementedError()
        #store = self.store.mapValues(lambda x: x @ other.store)
        #return TensorInEgg(self.cipher, [None, 1], store)

    def hstack(self,other):
        store = self.store.join(other.store, lambda x,y: np.hstack((x,y)))
        return TensorInEgg(self.cipher, self.shape, store)

    def map(self, func):
        store = self.store.mapValues(lambda x: func(x))
        return TensorInEgg(self.cipher, self.shape, store)

    def sum(self):
#        print("store:{}".format(self.store.__dict__))
        return self.store.reduce(lambda x,y: x+y)

    def mean(self):
        ret = self.sum() / self.store.count()
        if not isinstance(ret,np.ndarray):
            ret = np.array(ret)
        return TensorInPy(self.cipher, ret.shape, ret)

    def __rtruediv__(self, other):
        store = self.store.mapValues(lambda x: self.unwrap(other) / x)
        return TensorInEgg(self.cipher, self.shape, store)

    def __truediv__(self, other):
        return self.__mul__(1 / other)
