import numpy as np
import tenseal as ts


class CKKSKeypair(object):
    @staticmethod
    def generate_keypair(poly_modulus_degree=None, coeff_mod_bit_sizes=None, global_scale=2 ** 40):
        """Generate a keypair given security parameters"""
        if poly_modulus_degree is None and coeff_mod_bit_sizes is None:
            poly_modulus_degree = 8192
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        context.global_scale = global_scale

        secret_key = context.secret_key()

        context.make_context_public()
        public_context = context

        public_key = CKKSPublicKey(public_context)
        private_key = CKKSPrivateKey(secret_key)
        return public_key, private_key


class CKKSPublicKey(object):
    def __init__(self, public_context):
        if not isinstance(public_context, ts.enc_context.Context):
            raise TypeError("public_context should be a tenseal Context object")
        elif not public_context.is_public():
            raise ValueError("public_context is not a public context")

        self.__public_context = public_context

    def __getstate__(self):
        return self.__public_context.serialize(save_public_key=True,
                                               save_secret_key=False,
                                               save_relin_keys=False,
                                               save_galois_keys=False)

    def __setstate__(self, state):
        self.__public_context = ts.context_from(state)

    def encrypt(self, value):
        """Encrypt a real-valued number"""
        if isinstance(value, list):
            raise TypeError("encrypt only supports a single value encryption, not list of values")
        elif not self.__is_real_number(value):
            raise ValueError("value should be a real-valued number")

        singleton_vector = [value]
        ckks_vector = ts.ckks_vector(self.__public_context, singleton_vector)

        return CKKSEncryptedNumber(ckks_vector, self.__public_context)

    def __is_real_number(self, value):
        return np.isreal(value)


class CKKSPrivateKey(object):
    def __init__(self, secret_key):
        if not isinstance(secret_key, ts.enc_context.SecretKey):
            raise TypeError("secret_key should be a tenseal SecretKey object")

        self.__secret_key = secret_key

    def decrypt(self, encrypted_value):
        """Decrypt a CKKSEncryptedNumber"""
        if not isinstance(encrypted_value, CKKSEncryptedNumber):
            raise ValueError("encrypted_value should be a CKKSEncryptedNumber")

        ts_encrypted_vector = encrypted_value._get_tenseal_encrypted_vector()
        decrypted_vector = ts_encrypted_vector.decrypt(self.__secret_key)
        decrypted_value = decrypted_vector[0]

        return decrypted_value


class CKKSEncryptedNumber(object):
    def __init__(self, encrypted_vector, context):
        if not isinstance(encrypted_vector, ts.CKKSVector):
            raise TypeError(f"encrypted_vector should be a tenseal CKKSVector, got {type(encrypted_vector)}")
        elif not isinstance(context, ts.Context):
            raise TypeError(f"context should be a tenseal context, got {type(context)}")
        elif not self.__is_singleton_vector(encrypted_vector):
            raise ValueError(f"encrypted_vector should only contain one number, got vector of length {encrypted_vector.shape[0]}")

        self.__encrypted_vector = encrypted_vector
        self.__context = context

    def apply_obfuscator(self):
        pass

    def __getstate__(self):
        return self.__encrypted_vector.serialize(), self.__context.serialize(save_public_key=False,
                                                                             save_secret_key=False,
                                                                             save_relin_keys=False,
                                                                             save_galois_keys=False)

    def __setstate__(self, state):
        encrypted_vector_bytes, lightweight_context_bytes = state
        self.__context = ts.context_from(lightweight_context_bytes)
        self.__encrypted_vector = ts.ckks_vector_from(self.__context, encrypted_vector_bytes)

    def __add__(self, other):
        if isinstance(other, CKKSEncryptedNumber):
            return self.__from_ts_enc_vec(self.__encrypted_vector + other.__encrypted_vector)
        else:
            return self.__from_ts_enc_vec(self.__encrypted_vector + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, CKKSEncryptedNumber):
            return self.__from_ts_enc_vec(self.__encrypted_vector * other.__encrypted_vector)
        else:
            return self.__from_ts_enc_vec(self.__encrypted_vector * other)

    def __from_ts_enc_vec(self, ts_enc_vec):
        """Converts tenseal encrypted singleton vector to CKKSEncryptedNumber"""
        return CKKSEncryptedNumber(ts_enc_vec, self.__context)

    def _get_tenseal_encrypted_vector(self):
        """Should only be called by CKKSPrivateKey"""
        return self.__encrypted_vector

    def __is_singleton_vector(self, ckks_vector):
        return ckks_vector.shape[0] == 1
