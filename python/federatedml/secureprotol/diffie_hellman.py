#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import random

import gmpy2
from gmpy2 import mpz


class DiffieHellman(object):

    @staticmethod
    def _decode_hex_string(number_str):
        return mpz("0x{0}".format("".join(number_str.split())))

    @staticmethod
    def _oakley_group_768_768():
        """
        from RFC 2409, refer to https://tools.ietf.org/html/rfc2409#page-21
        """
        p = DiffieHellman._decode_hex_string("""
        FFFFFFFF FFFFFFFF C90FDAA2 2168C234 C4C6628B 80DC1CD1
        29024E08 8A67CC74 020BBEA6 3B139B22 514A0879 8E3404DD
        EF9519B3 CD3A431B 302B0A6D F25F1437 4FE1356D 6D51C245
        E485B576 625E7EC6 F44C42E9 A63A3620 FFFFFFFF FFFFFFFF
        """)
        g = DiffieHellman._decode_hex_string("2")
        return p, g

    @staticmethod
    def _oakley_group_1024_1024():
        """
        from RFC 2409, refer to https://tools.ietf.org/html/rfc2409#page-22
        """
        p = DiffieHellman._decode_hex_string("""
        FFFFFFFF FFFFFFFF C90FDAA2 2168C234 C4C6628B 80DC1CD1
        29024E08 8A67CC74 020BBEA6 3B139B22 514A0879 8E3404DD
        EF9519B3 CD3A431B 302B0A6D F25F1437 4FE1356D 6D51C245
        E485B576 625E7EC6 F44C42E9 A637ED6B 0BFF5CB6 F406B7ED
        EE386BFB 5A899FA5 AE9F2411 7C4B1FE6 49286651 ECE65381
        FFFFFFFF FFFFFFFF
        """)
        g = DiffieHellman._decode_hex_string("2")
        return p, g

    @staticmethod
    def _additional_group_1024_160():
        """
        from RFC 5114, has 160 bits subgroup size:
        0xF518AA8781A8DF278ABA4E7D64B7CB9D49462353
        refer to https://tools.ietf.org/html/rfc5114
        """
        p = DiffieHellman._decode_hex_string("""
        B10B8F96 A080E01D DE92DE5E AE5D54EC 52C99FBC FB06A3C6
        9A6A9DCA 52D23B61 6073E286 75A23D18 9838EF1E 2EE652C0
        13ECB4AE A9061123 24975C3C D49B83BF ACCBDD7D 90C4BD70
        98488E9C 219A7372 4EFFD6FA E5644738 FAA31A4F F55BCCC0
        A151AF5F 0DC8B4BD 45BF37DF 365C1A65 E68CFDA7 6D4DA708
        DF1FB2BC 2E4A4371
       """)
        g = DiffieHellman._decode_hex_string("""
        A4D1CBD5 C3FD3412 6765A442 EFB99905 F8104DD2 58AC507F
        D6406CFF 14266D31 266FEA1E 5C41564B 777E690F 5504F213
        160217B4 B01B886A 5E91547F 9E2749F4 D7FBD7D3 B9A92EE1
        909D0D22 63F80A76 A6A24C08 7A091F53 1DBF0A01 69B6A28A
        D662A4D1 8E73AFA3 2D779D59 18D08BC8 858F4DCE F97C2A24
        855E6EEB 22B3B2E5
        """)
        return p, g

    @staticmethod
    def _additional_group_2048_224():
        """
        from RFC 5114, has 224 bits subgroup size:
        0x801C0D34C58D93FE997177101F80535A4738CEBCBF389A99B36371EB
        refer to https://tools.ietf.org/html/rfc5114
        """
        p = DiffieHellman._decode_hex_string("""
        AD107E1E 9123A9D0 D660FAA7 9559C51F A20D64E5 683B9FD1
        B54B1597 B61D0A75 E6FA141D F95A56DB AF9A3C40 7BA1DF15
        EB3D688A 309C180E 1DE6B85A 1274A0A6 6D3F8152 AD6AC212
        9037C9ED EFDA4DF8 D91E8FEF 55B7394B 7AD5B7D0 B6C12207
        C9F98D11 ED34DBF6 C6BA0B2C 8BBC27BE 6A00E0A0 B9C49708
        B3BF8A31 70918836 81286130 BC8985DB 1602E714 415D9330
        278273C7 DE31EFDC 7310F712 1FD5A074 15987D9A DC0A486D
        CDF93ACC 44328387 315D75E1 98C641A4 80CD86A1 B9E587E8
        BE60E69C C928B2B9 C52172E4 13042E9B 23F10B0E 16E79763
        C9B53DCF 4BA80A29 E3FB73C1 6B8E75B9 7EF363E2 FFA31F71
        CF9DE538 4E71B81C 0AC4DFFE 0C10E64F
        """)
        g = DiffieHellman._decode_hex_string("""
        AC4032EF 4F2D9AE3 9DF30B5C 8FFDAC50 6CDEBE7B 89998CAF
        74866A08 CFE4FFE3 A6824A4E 10B9A6F0 DD921F01 A70C4AFA
        AB739D77 00C29F52 C57DB17C 620A8652 BE5E9001 A8D66AD7
        C1766910 1999024A F4D02727 5AC1348B B8A762D0 521BC98A
        E2471504 22EA1ED4 09939D54 DA7460CD B5F6C6B2 50717CBE
        F180EB34 118E98D1 19529A45 D6F83456 6E3025E3 16A330EF
        BB77A86F 0C1AB15B 051AE3D4 28C8F8AC B70A8137 150B8EEB
        10E183ED D19963DD D9E263E4 770589EF 6AA21E7F 5F2FF381
        B539CCE3 409D13CD 566AFBB4 8D6C0191 81E1BCFE 94B30269
        EDFE72FE 9B6AA4BD 7B5A0F1C 71CFFF4C 19C418E1 F6EC0179
        81BC087F 2A7065B3 84B890D3 191F2BFA
        """)
        return p, g

    @staticmethod
    def _additional_group_2048_256():
        """
        from RFC 5114, has 256 bits subgroup size:
        0x8CF83642A709A097B447997640129DA299B1A47D1EB3750BA308B0FE64F5FBD3
        refer to https://tools.ietf.org/html/rfc5114
        """
        p = DiffieHellman._decode_hex_string("""
        87A8E61D B4B6663C FFBBD19C 65195999 8CEEF608 660DD0F2
        5D2CEED4 435E3B00 E00DF8F1 D61957D4 FAF7DF45 61B2AA30
        16C3D911 34096FAA 3BF4296D 830E9A7C 209E0C64 97517ABD
        5A8A9D30 6BCF67ED 91F9E672 5B4758C0 22E0B1EF 4275BF7B
        6C5BFC11 D45F9088 B941F54E B1E59BB8 BC39A0BF 12307F5C
        4FDB70C5 81B23F76 B63ACAE1 CAA6B790 2D525267 35488A0E
        F13C6D9A 51BFA4AB 3AD83477 96524D8E F6A167B5 A41825D9
        67E144E5 14056425 1CCACB83 E6B486F6 B3CA3F79 71506026
        C0B857F6 89962856 DED4010A BD0BE621 C3A3960A 54E710C3
        75F26375 D7014103 A4B54330 C198AF12 6116D227 6E11715F
        693877FA D7EF09CA DB094AE9 1E1A1597
        """)
        g = DiffieHellman._decode_hex_string("""
        3FB32C9B 73134D0B 2E775066 60EDBD48 4CA7B18F 21EF2054
        07F4793A 1A0BA125 10DBC150 77BE463F FF4FED4A AC0BB555
        BE3A6C1B 0C6B47B1 BC3773BF 7E8C6F62 901228F8 C28CBB18
        A55AE313 41000A65 0196F931 C77A57F2 DDF463E5 E9EC144B
        777DE62A AAB8A862 8AC376D2 82D6ED38 64E67982 428EBC83
        1D14348F 6F2F9193 B5045AF2 767164E1 DFC967C1 FB3F2E55
        A4BD1BFF E83B9C80 D052B985 D182EA0A DB2A3B73 13D3FE14
        C8484B1E 052588B9 B7D2BBD2 DF016199 ECD06E15 57CD0915
        B3353BBB 64E0EC37 7FD02837 0DF92B52 C7891428 CDC67EB6
        184B523D 1DB246C3 2F630784 90F00EF8 D647D148 D4795451
        5E2327CF EF98C582 664B4C0F 6CC41659
        """)
        return p, g

    @staticmethod
    def _key_pair(num_bits=1024):
        available = {
            768: [DiffieHellman._oakley_group_768_768],
            1024: [DiffieHellman._oakley_group_1024_1024, DiffieHellman._additional_group_1024_160],
            2048: [DiffieHellman._additional_group_2048_224, DiffieHellman._additional_group_2048_256]
        }
        assert num_bits in available,\
            "key pairs with specified size({0} bits) not found, please use one of {1}".format(
                num_bits, available.keys()
            )
        return random.choice(available[num_bits]).__call__()

    @staticmethod
    def key_pair(num_bits=1024, pair_name=None):
        """
        Generate a primitive root for a big prime number is really slow!
        Notice the fact that:
        1. we don't need the generator to be a primitive element of the group
        but the one generates a large prime order.
        2. There is no security issue with Diffie-Hellman if you reuse previously generated 𝑝 and 𝑔.

        We simply use key pairs from RFC 5114 and RFC 2409
        @:param pair_name: one of "additional_group_1024_160", "additional_group_2048_224",
        "additional_group_2048_256", "oakley_group_768_768", "oakley_group_1024_1024"
        use additional_group_1024_160 as default
        @:param num_bits: specify size of p
        @:return p, g, where p is a prime number, g is a generator
        """
        if pair_name is None:
            if num_bits:
                return DiffieHellman._key_pair(num_bits)
            else:
                return DiffieHellman._additional_group_1024_160()

        assert pair_name in {
            "additional_group_1024_160", "additional_group_2048_224", "additional_group_2048_256",
            "oakley_group_768_768", "oakley_group_1024_1024"
        }, "unsupported pair name: {0}".format(pair_name)

        if pair_name == "additional_group_1024_160":
            return DiffieHellman._additional_group_1024_160()
        if pair_name == "additional_group_2048_224":
            return DiffieHellman._additional_group_2048_224()
        if pair_name == "additional_group_2048_256":
            return DiffieHellman._additional_group_2048_256()
        if pair_name == "oakley_group_768_768":
            return DiffieHellman._oakley_group_768_768()
        if pair_name == "oakley_group_1024_1024":
            return DiffieHellman._oakley_group_1024_1024()

    # noinspection PyArgumentList
    @staticmethod
    def generate_secret(p, num_bits=1024):
        return mpz(random.SystemRandom().getrandbits(num_bits)) % p

    @staticmethod
    def encrypt(g, r, p):
        return gmpy2.powmod(g, r, p)

    @staticmethod
    def decrypt(gr, r, p):
        return gmpy2.powmod(gr, r, p)
