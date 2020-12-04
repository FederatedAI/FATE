import random
from federatedml.secureprotol import gmpy_math
import functools


class Vss(object):
    def __init__(self):
        self.prime = None
        self.g = 2
        self.share_amount = -1

    def set_share_amount(self, share_amount):
        self.share_amount = share_amount

    def generate_prime(self):
        self.prime = gmpy_math.getprimeover(512)

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
            for c in reversed(coefficients):
                y *= x
                y += c
                y %= self.prime
            f_x.append((x, y))
        return f_x

    def decrypt(self, x_values, y_values):
        k = len(x_values)
        assert k == len(set(x_values)), 'x_values points must be distinct'
        nums = []
        dens = []
        for i in range(k):
            others = list(x_values)
            cur = others.pop(i)
            nums.append(functools.reduce(lambda a, b: a * b, [0 - o for o in others], 1))
            dens.append(functools.reduce(lambda a, b: a * b, [cur - o for o in others], 1))
        den = functools.reduce(lambda a, b: a * b, dens, 1)

        num = []
        for i in range(k):
            term = nums[i] * den * y_values[i] % self.prime
            inv = self.extended_gcd(dens[i], self.prime)
            num.append(term * inv)

        secret = ((sum(num) * self.extended_gcd(den, self.prime)) + self.prime) % self.prime

        return secret

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

    @staticmethod
    def extended_gcd(a: int, b: int):
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError('a and b must be integers')
        x = 0
        last_x = 1
        y = 1
        last_y = 0
        while b != 0:
            quot = a // b
            a, b = b, a % b
            x, last_x = last_x - quot * x, x
            y, last_y = last_y - quot * y, y
        return last_x
