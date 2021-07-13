import base64
import hashlib
import libsm3py

from federatedml.util import consts

SUPPORT_METHOD = [consts.MD5, consts.SHA1, consts.SHA224, consts.SHA256,
                  consts.SHA384, consts.SHA512, consts.SM3, "none"]


def compute_md5(value):
    return hashlib.md5(bytes(value, encoding='utf-8')).hexdigest()


def compute_md5_base64(value):
    return str(base64.b64encode(hashlib.md5(bytes(value, encoding='utf-8')).digest()), "utf-8")


def compute_sha256(value):
    return hashlib.sha256(bytes(value, encoding='utf-8')).hexdigest()


def compute_sha256_base64(value):
    return str(base64.b64encode(hashlib.sha256(bytes(value, encoding='utf-8')).digest()), "utf-8")


def compute_sha1(value):
    return hashlib.sha1(bytes(value, encoding='utf-8')).hexdigest()


def compute_sha1_base64(value):
    return str(base64.b64encode(hashlib.sha1(bytes(value, encoding='utf-8')).digest()), "utf-8")


def compute_sha224(value):
    return hashlib.sha224(bytes(value, encoding='utf-8')).hexdigest()


def compute_sha224_base64(value):
    return str(base64.b64encode(hashlib.sha224(bytes(value, encoding='utf-8')).digest()), "utf-8")


def compute_sha512(value):
    return hashlib.sha512(bytes(value, encoding='utf-8')).hexdigest()


def compute_sha512_base64(value):
    return str(base64.b64encode(hashlib.sha512(bytes(value, encoding='utf-8')).digest()), "utf-8")


def compute_sha384(value):
    return hashlib.sha384(bytes(value, encoding='utf-8')).hexdigest()


def compute_sha384_base64(value):
    return str(base64.b64encode(hashlib.sha384(bytes(value, encoding='utf-8')).digest()), "utf-8")


def compute_sm3(value):
    return libsm3py.hash(bytes(value, encoding='utf-8')).hex()


def compute_sm3_base64(value):
    return str(base64.b64encode(libsm3py.hash(bytes(value, encoding='utf-8')).encode('utf-8')), "utf-8")


def compute_no_hash(value):
    return str(value)


def compute_no_hash_base64(value):
    return str(base64.b64encode(bytes(value, encoding='utf-8')), 'utf-8')


HASH_FUNCTION = {
    consts.MD5: compute_md5,
    consts.SHA1: compute_sha1,
    consts.SHA224: compute_sha224,
    consts.SHA256: compute_sha256,
    consts.SHA384: compute_sha384,
    consts.SHA512: compute_sha512,
    consts.SM3: compute_sm3,
    "none": compute_no_hash
}


HASH_BASE64_FUNCTION = {
    consts.MD5: compute_md5_base64,
    consts.SHA1: compute_sha1_base64,
    consts.SHA224: compute_sha224_base64,
    consts.SHA256: compute_sha256_base64,
    consts.SHA384: compute_sha384_base64,
    consts.SHA512: compute_sha512_base64,
    consts.SM3: compute_sm3_base64,
    "none": compute_no_hash_base64
}


class Hash:
    def __init__(self, method, base64=0):
        self.method = method
        if self.method not in SUPPORT_METHOD:
            raise ValueError("Hash does not support method:{}".format(self.method))

        self.base64 = base64

        if self.base64:
            self.hash_operator = HASH_BASE64_FUNCTION[self.method]
        else:
            self.hash_operator = HASH_FUNCTION[self.method]

    def compute(self, value, prefix_salt=None, suffix_salt=None):
        value = str(value)
        if prefix_salt:
            value = prefix_salt + value

        if suffix_salt:
            value = value + suffix_salt
        return self.hash_operator(value)
