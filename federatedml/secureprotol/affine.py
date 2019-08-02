import random
import math

from federatedml.secureprotol.affine_encoder import AffineEncoder


class AffineCipher(object):
    def __init__(self):
        pass

    @staticmethod
    def generate_keypair(key_size=1024, a_ratio=None, b_ratio=None):
        n = random.getrandbits(key_size)
        if a_ratio is None:
            a_ratio = random.random()
        if b_ratio is None:
            b_ratio = random.random()
        while True:
            a = random.getrandbits(int(key_size * a_ratio))
            if math.gcd(n, a) == 1:
                break
        b = random.getrandbits(int(key_size * b_ratio))
        return AffineCipherKey(a, b, n)


class AffineCipherKey(object):
    def __init__(self, a, b, n):
        self.a = a
        self.b = b
        self.n = n
        self.a_inv = self.mod_inverse()
        self.affine_encoder = AffineEncoder()

    def encrypt(self, plaintext):
        return self.raw_encrypt(self.affine_encoder.encode(plaintext))

    def decrypt(self, ciphertext):
        return self.affine_encoder.decode(self.raw_decrypt(ciphertext), ciphertext.multiplier)

    def raw_encrypt(self, plaintext):
        return AffineCiphertext((self.a * plaintext + self.b) % self.n)

    def raw_decrypt(self, ciphertext):
        plaintext = (self.a_inv * (ciphertext.cipher % self.n - ciphertext.multiplier * self.b)) % self.n
        if plaintext / self.n > 0.9:
            return plaintext - self.n
        return plaintext

    def mod_inverse_brutal(self):
        div = self.a % self.n
        for x in range(1, self.n):
            if (div * x) % self.n == 1:
                return x
        return 1

    def mod_inverse(self):
        """return x such that (x * a) % b == 1"""
        g, x, _ = self.xgcd(self.a, self.n)
        if g == 1:
            return x % self.n
        return None

    def xgcd(self, a, b):
        """return (g, x, y) such that a*x + b*y = g = gcd(a, b)"""
        x0, x1, y0, y1 = 0, 1, 1, 0
        while a != 0:
            q, b, a = b // a, a, b % a
            y0, y1 = y1, y0 - q * y1
            x0, x1 = x1, x0 - q * x1
        return b, x0, y0


class AffineCiphertext(object):
    def __init__(self, cipher, multiplier=1):
        self.cipher = cipher
        self.multiplier = multiplier

    def __add__(self, other):
        if isinstance(other, AffineCiphertext):
            return AffineCiphertext(self.cipher + other.cipher, self.multiplier + other.multiplier)
        elif type(other) is int or type(other) is float:
            return AffineCiphertext(self.cipher + other, self.multiplier)
        else:
            raise TypeError("Addition only supports int and AffineCiphertext")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) is int or type(other) is float:
            return AffineCiphertext(self.cipher * other, self.multiplier * other)
        else:
            raise TypeError("Multiplication only supports int.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, other):
        return self.__mul__(1 / other)
