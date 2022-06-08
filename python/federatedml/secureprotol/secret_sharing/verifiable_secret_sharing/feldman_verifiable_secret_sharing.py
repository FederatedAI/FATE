import random
from federatedml.secureprotol import gmpy_math
from gmpy2 import mpz


class FeldmanVerifiableSecretSharing(object):
    def __init__(self):
        self.Q_n = 6
        self.p = None
        self.g = None
        self.q = None
        self.share_amount = -1
        self.commitments = []

    def set_share_amount(self, host_count):
        self.share_amount = host_count + 1

    def encrypt(self, secret):
        coefficient = [self.encode(secret)]
        for i in range(self.share_amount - 1):
            random_coefficient = random.SystemRandom().randint(0, self.p - 1)
            coefficient.append(random_coefficient)

        f_x = []
        for x in range(1, self.share_amount + 1):
            y = 0
            for c in reversed(coefficient):
                y *= x % self.q
                y += c % self.q
                y %= self.q
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
                numerator = (numerator * (0 - x_values[j]))
                denominator = (denominator * (x_values[i] - x_values[j]))
            # get the polynomial from the numerator + denominator mod inverse
            lagrange_polynomial = (numerator * gmpy_math.invert(denominator, self.q)) % self.q
            # multiply the current y & the evaluated polynomial & add it to f(x)
            secret = (self.q + secret + (y_values[i] * lagrange_polynomial)) % self.q
        return self.decode(secret)

    def calculate_commitment(self, coefficient):
        return gmpy_math.powmod(self.g, coefficient, self.p)

    def verify(self, f_x, commitment):
        x, y = f_x[0], f_x[1]
        v1 = gmpy_math.powmod(self.g, y, self.p)
        v2 = 1
        for i in range(len(commitment)):
            v2 *= gmpy_math.powmod(commitment[i], (x**i), self.p)
        v2 = v2 % self.p
        if v1 != v2:
            return False
        return True

    def encode(self, x):
        upscaled = int(x * (10 ** self.Q_n))
        if isinstance(x, int):
            assert (abs(upscaled) < (self.q / (2 * self.share_amount))), (
                f"{x} cannot be correctly embedded: choose bigger q or a lower precision"
            )
        return upscaled

    def decode(self, s):
        gate = s > self.q / 2
        neg_nums = (s - self.q) * gate
        pos_nums = s * (1 - gate)
        integer, fraction = divmod((neg_nums + pos_nums), (10 ** self.Q_n))
        result = integer if fraction == 0 else integer + fraction / (10**self.Q_n)
        return result

    @staticmethod
    def _decode_hex_string(number_str):
        return int(mpz("0x{0}".format("".join(number_str.split()))))

    def key_pair(self):
        """
        from RFC 5114, has 160 bits subgroup size:
        0xF518AA8781A8DF278ABA4E7D64B7CB9D49462353
        refer to https://tools.ietf.org/html/rfc5114
        """
        self.p = FeldmanVerifiableSecretSharing._decode_hex_string("""
        B10B8F96 A080E01D DE92DE5E AE5D54EC 52C99FBC FB06A3C6
        9A6A9DCA 52D23B61 6073E286 75A23D18 9838EF1E 2EE652C0
        13ECB4AE A9061123 24975C3C D49B83BF ACCBDD7D 90C4BD70
        98488E9C 219A7372 4EFFD6FA E5644738 FAA31A4F F55BCCC0
        A151AF5F 0DC8B4BD 45BF37DF 365C1A65 E68CFDA7 6D4DA708
        DF1FB2BC 2E4A4371
       """)
        self.g = FeldmanVerifiableSecretSharing._decode_hex_string("""
        A4D1CBD5 C3FD3412 6765A442 EFB99905 F8104DD2 58AC507F
        D6406CFF 14266D31 266FEA1E 5C41564B 777E690F 5504F213
        160217B4 B01B886A 5E91547F 9E2749F4 D7FBD7D3 B9A92EE1
        909D0D22 63F80A76 A6A24C08 7A091F53 1DBF0A01 69B6A28A
        D662A4D1 8E73AFA3 2D779D59 18D08BC8 858F4DCE F97C2A24
        855E6EEB 22B3B2E5
        """)
        self.q = FeldmanVerifiableSecretSharing._decode_hex_string("""
        F518AA87 81A8DF27 8ABA4E7D 64B7CB9D 49462353
        """)
