import random
from federatedml.secureprotol import gmpy_math


class Vss(object):
    def __init__(self):
        self.prime = None
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
