class AffineEncoder(object):
    def __init__(self, mult=2 ** 23, trans=0):
        self.mult = mult
        self.trans = trans

    def encode(self, plaintext):
        return int(self.mult * (plaintext + self.trans))

    def decode(self, ciphertext, multiplier=1):
        return ciphertext / self.mult - multiplier * self.trans
