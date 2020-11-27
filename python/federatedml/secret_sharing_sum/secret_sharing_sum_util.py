def calculate_mersenne_primes():
    mersenne_prime_exponents = [
        2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279
    ]
    primes = []
    for exp in mersenne_prime_exponents:
        prime = 1
        for i in range(exp):
            prime *= 2
        prime -= 1
        primes.append(prime)
    return primes


class Primes():
    def __init__(self):
        self.primes = None
        smallest_257_bit_prime = (2 ** 256 + 297)
        smallest_321_bit_prime = (2 ** 320 + 27)
        smallest_385_bit_prime = (2 ** 384 + 231)
        self.standard_primes = calculate_mersenne_primes() + [
            smallest_257_bit_prime, smallest_321_bit_prime, smallest_385_bit_prime
        ]
        self.standard_primes.sort()

    def get_primes(self):
        if self.primes is None:
            self.primes = self.get_large_enough_prime([100000000])
        return self.primes

    def get_large_enough_prime(self, batch):
        primes = self.standard_primes
        for prime in primes:
            numbers_greater_than_prime = [i for i in batch if i > prime]
            if len(numbers_greater_than_prime) == 0:
                self.primes = prime
                return self.primes
        return None
