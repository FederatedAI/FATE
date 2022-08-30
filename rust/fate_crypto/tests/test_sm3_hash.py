import unittest
from fate_crypto.hash import sm3_hash


class TestCorrect(unittest.TestCase):
    def test_hash_1(self):
        data = b"abc"
        expected = "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"
        self.assertEqual(sm3_hash(data).hex(), expected)

    def test_hash_2(self):
        data = b"abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd"
        expected = "debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732"
        self.assertEqual(sm3_hash(data).hex(), expected)


if __name__ == "__main__":
    unittest.main()
