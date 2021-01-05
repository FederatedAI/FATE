import base64
import hashlib
import libsm3

from federatedml.util import LOGGER


class Hash:
    def __init__(self, method, base64=0):
        self.method = method
        self.base64 = base64

        self.dist_encode_function = {
            "md5": self.__compute_md5,
            "sha1": self.__compute_sha1,
            "sha224": self.__compute_sha224,
            "sha256": self.__compute_sha256,
            "sha384": self.__compute_sha384,
            "sha512": self.__compute_sha512,
            "sm3": self.__compute_sm3,
            "none": self.__compute_no_hash
        }

    @staticmethod
    def is_support(method):
        support_encode_method = ["md5", "sha1", "sha224", "sha256", "sha384", "sha512", "sm3", "none"]
        return method in support_encode_method

    def __compute_md5(self, value):
        if self.base64 == 1:
            return str(base64.b64encode(hashlib.md5(bytes(value, encoding='utf-8')).digest()), "utf-8")
        else:
            return hashlib.md5(bytes(value, encoding='utf-8')).hexdigest()

    def __compute_sha256(self, value):
        if self.base64 == 1:
            return str(base64.b64encode(hashlib.sha256(bytes(value, encoding='utf-8')).digest()), "utf-8")
        else:
            return hashlib.sha256(bytes(value, encoding='utf-8')).hexdigest()

    def __compute_sha1(self, value):
        if self.base64 == 1:
            return str(base64.b64encode(hashlib.sha1(bytes(value, encoding='utf-8')).digest()), "utf-8")
        else:
            return hashlib.sha1(bytes(value, encoding='utf-8')).hexdigest()

    def __compute_sha224(self, value):
        if self.base64 == 1:
            return str(base64.b64encode(hashlib.sha224(bytes(value, encoding='utf-8')).digest()), "utf-8")
        else:
            return hashlib.sha224(bytes(value, encoding='utf-8')).hexdigest()

    def __compute_sha512(self, value):
        if self.base64 == 1:
            return str(base64.b64encode(hashlib.sha512(bytes(value, encoding='utf-8')).digest()), "utf-8")
        else:
            return hashlib.sha512(bytes(value, encoding='utf-8')).hexdigest()

    def __compute_sha384(self, value):
        if self.base64 == 1:
            return str(base64.b64encode(hashlib.sha384(bytes(value, encoding='utf-8')).digest()), "utf-8")
        else:
            return hashlib.sha384(bytes(value, encoding='utf-8')).hexdigest()

    def __compute_sm3(self, value):
        if self.base64 == 1:
            return str(base64.b64encode(libsm3.hash(bytes(value, encoding='utf-8')).encode('utf-8')), "utf-8")
        else:
            return libsm3.hash(bytes(value, encoding='utf-8')).hex()

    def __compute_no_hash(self, value):
        if self.base64 == 1:
            return str(base64.b64encode(bytes(value, encoding='utf-8')), 'utf-8')
        else:
            return str(value)

    def compute(self, value, pre_salt=None, postfit_salt=None):
        if not Hash.is_support(self.method):
            LOGGER.warning("Hash module do not support method:{}".format(self.method))
            return value

        value = str(value)
        if pre_salt is not None and len(pre_salt) > 0:
            value = pre_salt + value

        if postfit_salt is not None and len(postfit_salt) > 0:
            value = value + postfit_salt
        return self.dist_encode_function[self.method](value)
