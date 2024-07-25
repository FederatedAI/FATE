"""OU encryption library for partially homomorphic encryption."""

import numpy as np
import random

from federatedml.secureprotol import gmpy_math
from federatedml.secureprotol.fixedpoint import FixedPointNumber


# according to this paper
# << Accelerating Okamoto-Uchiyama’s Public-Key Cryptosystem >>
# and NIST's recommendation:
# https://www.keylength.com/en/4/
# 160 bits for key size 1024
# 224 bits for key size 2048
# 256 bits for key size 3072
kPrimeFactorSize1024 = 160
kPrimeFactorSize2048 = 224
kPrimeFactorSize3072 = 256

class OUKeypair(object):
    def __init__(self):
        pass

    @staticmethod
    def random_monic_exact_bits(bits):
        global last_generated
        new_value = random.getrandbits(bits)
        
        if 'last_generated' not in globals():
            last_generated = new_value
        else:
            if new_value <= last_generated:
                new_value = last_generated + 1
        
        last_generated = new_value
        return new_value

    def generate_keypair(self, n_length=1024):
        """return a new :class:`OUPublicKey` and :class:`OUPrivateKey`.
        """ 
        secret_size = (n_length + 2) // 3
        
        prime_factor_size = kPrimeFactorSize1024
        if n_length >= 3072:
            prime_factor_size = kPrimeFactorSize3072
        elif n_length >= 2048:
            prime_factor_size = kPrimeFactorSize2048

        assert prime_factor_size * 2 <= secret_size, \
            "Key size must be larger than {} bits".format(prime_factor_size * 2 * 3 - 2)

        # generate p
        while True:
            prime_factor = gmpy_math.getprimeover(prime_factor_size)
            # bits_of(a * b) <= bits_of(a) + bits_of(b),
            # So we add extra two bits to u:
            #    one bit for prime_factor * u; another one bit for p^2;
            # Also, make sure that u > prime_factor
            u = self.random_monic_exact_bits(secret_size - prime_factor_size + 2) # p - 1 has a large prime factor
            p = prime_factor * u + 1

            if gmpy_math.is_prime(p):
                break
        
        # since bits_of(a * b) <= bits_of(a) + bits_of(b)
        # add another 1 bit for q
        q = gmpy_math.getprimeover(secret_size + 1)
        p_square = p ** 2
        t = prime_factor
        n = p_square * q

        # calculate g_p
        while True:
            while True:
                g = random.randint(1, n-1)
                gcd = np.gcd(g, p)
                if gcd == 1:
                    break

            gp = gmpy_math.powmod(g % p_square, p - 1, p_square)
            check = gmpy_math.powmod(gp, p, p_square)
            
            if check == 1:
                break

        # calculate G
        capital_g = gmpy_math.powmod(g, u, n)

        while True:
            g = random.randint(1, n-1)
            if g % p != 0:
                break

        # calculate H
        capital_h = gmpy_math.powmod(g, n * u, n)

        # max_plaintext_ must be a power of 2, for ease of use
        max_plaintext = pow(10, prime_factor_size // 2) // 2
        
        public_key = OUPublicKey(n, capital_g, capital_h, max_plaintext)
        private_key = OUPrivateKey(public_key, p, q, t, gp, max_plaintext)

        return public_key, private_key


class OUPublicKey(object):
    """Contains a public key and associated encryption methods.
    """

    def __init__(self, n, capital_g, capital_h, max_plaintext):
        self.n = n                         # n = p^2 * q
        self.capital_g = capital_g         # G = g^u mod n for some random g \in [0, n)
        self.capital_h = capital_h         # H = g'^{n*u} mod n for some random g' \in [0, n)
        self.max_plaintext = max_plaintext # always power of 2, e.g. max_plaintext_ == 2^681

    def __repr__(self):
        hashcode = hex(hash(self))[2:]

        return "<OUPublicKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.n == other.n and self.capital_g == other.capital_g and self.capital_h == other.capital_h

    def __hash__(self):
        return hash(self.n)

    # multi H^r
    # r is a random number < n
    # H and n is public key
    def apply_obfuscator(self, ciphertext, random_value=None):
        """
        """
        r = random_value or random.SystemRandom().randrange(1, self.n)
        obfuscator = gmpy_math.powmod(self.capital_h, r, self.n)

        return (ciphertext * obfuscator) % self.n

    def raw_encrypt(self, plaintext, random_value=None):
        """
        """
        if not isinstance(plaintext, int):
            raise TypeError("plaintext should be int, but got: %s" %
                            type(plaintext))

        if plaintext >= self.max_plaintext:
            plaintext -= self.max_plaintext * 2

        gm = gmpy_math.powmod(self.capital_g, plaintext, self.n)
        
        ciphertext = self.apply_obfuscator(gm, random_value)
     
        return ciphertext

    def encrypt(self, value, precision=None, random_value=None):
        """Encode and OU encrypt a real number value.
        """
        if isinstance(value, FixedPointNumber):
            value = value.decode()
        encoding = FixedPointNumber.encode(value, self.max_plaintext * 2, self.max_plaintext, precision)
        obfuscator = random_value or 1
        ciphertext = self.raw_encrypt(encoding.encoding, random_value=obfuscator)
        encryptednumber = OUEncryptedNumber(self, ciphertext, encoding.exponent)

        return encryptednumber


class OUPrivateKey(object):
    """Contains a private key and associated decryption method.
    """

    def __init__(self, public_key, p, q, t, gp, max_plaintext):
        self.public_key = public_key
        self.p = p
        self.q = q                                            # primes such that log2(p), log2(q) ~ n_bits / 3
        self.t = t                                            # a big prime factor of p - 1, i.e., p = t * u + 1
        self.gp = gp
        self.gp_inv = gmpy_math.invert((self.gp - 1) // p, p) # L(g^{p-1} mod p^2))^{-1} mod p
        self.p_square = p ** 2
        self.max_plaintext = max_plaintext

    def __repr__(self):
        hashcode = hex(hash(self))[2:]

        return "<OUPrivateKey {}>".format(hashcode[:10])

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q and self.t == other.t and self.gp_inv == other.gp_inv

    def __hash__(self):
        return hash((self.p, self.q))

    def raw_decrypt(self, ciphertext):
        """return raw plaintext.
        """
        if not isinstance(ciphertext, int):
            raise TypeError("ciphertext should be an int, not: %s" %
                            type(ciphertext))
        
        plaintext = 0

        ct = gmpy_math.powmod(ciphertext % self.p_square, self.t, self.p_square)

        plaintext = ((ct // self.p) * self.gp_inv) % self.p

        if plaintext >= self.p / 2:
            plaintext -= self.p
        if plaintext >= self.max_plaintext:
            plaintext = plaintext % (self.max_plaintext * 2)

        return plaintext

    def decrypt(self, encrypted_number):
        """return the decrypted & decoded plaintext of encrypted_number.
        """
        if not isinstance(encrypted_number, OUEncryptedNumber):
            raise TypeError("encrypted_number should be an OUEncryptedNumber, \
                             not: %s" % type(encrypted_number))

        if self.public_key != encrypted_number.public_key:
            raise ValueError("encrypted_number was encrypted against a different key!")

        encoded = self.raw_decrypt(encrypted_number.ciphertext(be_secure=False))
        encoded = FixedPointNumber(encoded,
                                   encrypted_number.exponent,
                                   self.public_key.max_plaintext * 2,
                                   self.public_key.max_plaintext)
        decrypt_value = encoded.decode()

        return decrypt_value


class OUEncryptedNumber(object):
    """Represents the OU encryption of a float or int.
    """

    def __init__(self, public_key, ciphertext, exponent=0):
        self.public_key = public_key
        self.__ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscator = False

        if not isinstance(self.__ciphertext, int):
            raise TypeError("ciphertext should be an int, not: %s" % type(self.__ciphertext))

        if not isinstance(self.public_key, OUPublicKey):
            raise TypeError("public_key should be a OUPublicKey, not: %s" % type(self.public_key))

    def ciphertext(self, be_secure=True):
        """return the ciphertext of the OUEncryptedNumber.
        """
        if be_secure and not self.__is_obfuscator:
            self.apply_obfuscator()

        return self.__ciphertext

    def apply_obfuscator(self):
        """ciphertext by multiplying by H ** r with random r
        """
        self.__ciphertext = self.public_key.apply_obfuscator(self.__ciphertext)
        self.__is_obfuscator = True

    def __add__(self, other):
        if isinstance(other, OUEncryptedNumber):
            return self.__add_encryptednumber(other)
        else:
            return self.__add_scalar(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def __mul__(self, scalar):
        """return Multiply by an scalar(such as int, float)
        """
        if isinstance(scalar, FixedPointNumber):
            scalar = scalar.decode()
        encode = FixedPointNumber.encode(scalar, self.public_key.max_plaintext * 2, self.public_key.max_plaintext)
        plaintext = encode.encoding

        if plaintext < 0 or plaintext >= (self.public_key.max_plaintext * 2):
            raise ValueError("Scalar out of bounds: %i" % plaintext)

        if plaintext > self.public_key.max_plaintext:
            # Very large plaintext, play a sneaky trick using inverses
            plaintext -= self.public_key.max_plaintext * 2

        ciphertext = gmpy_math.powmod(self.ciphertext(False), plaintext, self.public_key.n)

        exponent = self.exponent + encode.exponent

        return OUEncryptedNumber(self.public_key, ciphertext, exponent)
    
    def increase_exponent_to(self, new_exponent):
        """return OUEncryptedNumber:
           new OUEncryptedNumber with same value but having great exponent.
        """
        if new_exponent < self.exponent:
            raise ValueError("New exponent %i should be great than old exponent %i" % (new_exponent, self.exponent))

        factor = pow(FixedPointNumber.BASE, new_exponent - self.exponent)
        new_encryptednumber = self.__mul__(factor)
        new_encryptednumber.exponent = new_exponent

        return new_encryptednumber

    def __align_exponent(self, x, y):
        """return x,y with same exponet
        """
        if x.exponent < y.exponent:
            x = x.increase_exponent_to(y.exponent)
        elif x.exponent > y.exponent:
            y = y.increase_exponent_to(x.exponent)

        return x, y

    def __add_scalar(self, scalar):
        """return OUEncryptedNumber: z = E(x) + y
        """
        if isinstance(scalar, FixedPointNumber):
            scalar = scalar.decode()
        
        encoded = FixedPointNumber.encode(scalar,
                                          self.public_key.max_plaintext * 2,
                                          self.public_key.max_plaintext,
                                          max_exponent=self.exponent)
        
        return self.__add_fixpointnumber(encoded)

    def __add_fixpointnumber(self, encoded):
        """return OUEncryptedNumber: z = E(x) + FixedPointNumber(y)
        # """
        if self.public_key.max_plaintext != encoded.max_int:
            raise ValueError("Attempted to add numbers encoded against different public keys!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, encoded)

        encrypted_scalar = x.public_key.raw_encrypt(y.encoding, 1)
        encryptednumber = self.__raw_add(x.ciphertext(False), encrypted_scalar, x.exponent)

        return encryptednumber
    
    def __add_encryptednumber(self, other):
        """return OUEncryptedNumber: z = E(x) + E(y)
        """
        if self.public_key != other.public_key:
            raise ValueError("add two numbers have different public key!")

        # their exponents must match, and align.
        x, y = self.__align_exponent(self, other)
        
        encryptednumber = self.__raw_add(x.ciphertext(False), y.ciphertext(False), x.exponent)

        return encryptednumber

    def __raw_add(self, e_x, e_y, exponent):
        """return the integer E(x + y) given ints E(x) and E(y).
        """
        ciphertext = gmpy_math.mpz(e_x) * gmpy_math.mpz(e_y) % self.public_key.n

        return OUEncryptedNumber(self.public_key, int(ciphertext), exponent)
