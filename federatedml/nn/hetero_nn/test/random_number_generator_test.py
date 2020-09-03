from federatedml.nn.hetero_nn.util.random_number_generator import RandomNumberGenerator
from federatedml.nn.hetero_nn.backend.paillier_tensor import PaillierTensor
import numpy as np
from fate_arch.session import Session
import random
import unittest


class TestRandomNumberGenerator(unittest.TestCase):
    def setUp(self):
        Session.create(0, 0).init_computing(str(random.randint(0, 10000000)))
        self.rng_gen = RandomNumberGenerator()

    def test_generate_random_number(self):
        data = np.ones((1, 3))
        random_data = self.rng_gen.generate_random_number(data.shape)

        self.assertTrue(random_data.shape == data.shape)

    def test_fast_generate_random_number(self):
        data = np.ones((1000, 100))

        random_data = self.rng_gen.fast_generate_random_number(data.shape)
        self.assertTrue(isinstance(random_data, PaillierTensor))
        self.assertTrue(tuple(random_data.shape) == tuple(data.shape))


if __name__ == '__main__':
    unittest.main()
