import random
import gmpy2
from scipy.interpolate import lagrange
from federatedml.secureprotol.secretsharing.primes import Primes
from federatedml.secureprotol import gmpy_math


class Vss(object):
    def __init__(self):
        self.prime = None
        self.prime_gen = Primes()
        self.g = 2
        self.share_amount = -1

    def set_share_amount(self, share_amount):
        self.share_amount = share_amount

    def generate_prime(self, n):
        r = random.SystemRandom().randint(1, n)
        self.prime = int(gmpy2.next_prime(r))

    def set_prime(self, prime):
        self.prime = prime

    def encrypt(self, x):
        coefficients = [int(x)]
        for i in range(self.share_amount - 1):
            random_coefficients = random.SystemRandom().randint(1, self.prime - 1)
            coefficients.append(random_coefficients)

        f_x = []
        for x in range(1, self.share_amount+1):
            y = 0
            for index, c in enumerate(coefficients):
                exponentiation = gmpy_math.powmod(x, index, self.prime)
                term = (c * exponentiation) % self.prime
                y = (y + term) % self.prime
            f_x.append((x, y))

        return f_x

    def decrypt(self, points):
        x_values, y_values = zip(*points)
        scipy_fx = lagrange(x_values, y_values)(0)
        secret = scipy_fx % self.prime

        return secret

    def mod_inverse(self, k):
        res = int(gmpy2.invert(k, self.prime))

        if res == 0:
            raise ZeroDivisionError('invert(a, b) no inverse exists')

        return res

    def calculate_commitments(self, coefficient):
        return gmpy_math.powmod(self.g, coefficient, self.prime)

    def verify(self, xy, commitments):
        x, y = xy[0], xy[1]
        v1 = gmpy_math.powmod(self.g, y, self.prime)
        v2 = 1
        for i in range(len(commitments)):
            v2 *= gmpy_math.powmod(commitments[i], (x**i), self.prime)
            v2 = v2 % self.prime
        return v1 == v2
