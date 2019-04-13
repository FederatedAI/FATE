import base64
import hashlib

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class Encode:
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
        }

    @staticmethod
    def is_support(method):
        support_encode_method = ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"]
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

    def compute(self, value, pre_salt=None, postfit_salt=None):
        if not Encode.is_support(self.method):
            LOGGER.warning("Encode module do not support method:{}".format(self.method))
            return value

        if pre_salt is not None:
            value = pre_salt + value

        if postfit_salt is not None:
            value = value + postfit_salt
        return self.dist_encode_function[self.method](value)
