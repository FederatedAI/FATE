import random
from federatedml.secureprotol import gmpy_math


class Vss(object):
    def __init__(self):
        self.prime = None
        self.share_amount = -1
        self.g = 2
        self.commitments = []

    def set_share_amount(self, share_amount):
        self.share_amount = share_amount

    def generate_prime(self):
        self.prime = gmpy_math.getprimeover(512)

    def set_prime(self, prime):
        self.prime = prime

    def encrypt(self, secret):
        coefficient = [int(secret)]
        for i in range(self.share_amount - 1):
            random_coefficient = random.SystemRandom().randint(0, self.prime - 1)
            coefficient.append(random_coefficient)

        f_x = []
        for x in range(1, self.share_amount+1):
            y = 0
            for c in reversed(coefficient):
                y *= x
                y += c
            f_x.append((x, y))

        commitment = list(map(self.calculate_commitment, coefficient))

        return f_x, commitment

    def decrypt(self, x_values, y_values):
        k = len(x_values)
        assert k == len(set(x_values)), 'x_values points must be distinct'
        secret = 0
        for i in range(k):
            numerator, denominator = 1, 1
            for j in range(k):
                if i == j:
                    continue
                # compute a fraction & update the existing numerator + denominator
                numerator = (numerator * (0 - x_values[j])) % self.prime
                denominator = (denominator * (x_values[i] - x_values[j])) % self.prime
            # get the polynomial from the numerator + denominator mod inverse
            lagrange_polynomial = numerator * gmpy_math.invert(denominator, self.prime)
            # multiply the current y & the evaluated polynomial & add it to f(x)
            secret = (self.prime + secret + (y_values[i] * lagrange_polynomial)) % self.prime

        return secret

    def calculate_commitment(self, coefficient):
        return gmpy_math.powmod(self.g, coefficient, self.prime)

    def verify(self, f_x, commitment):
        x, y = f_x[0], f_x[1]
        v1 = gmpy_math.powmod(self.g, y, self.prime)
        v2 = 1
        for i in range(len(commitment)):
            v2 *= gmpy_math.powmod(commitment[i], (x**i), self.prime)
        v2 = v2 % self.prime
        if v1 != v2:
            raise ValueError("error sharing")
