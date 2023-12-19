# -*-coding: Utf-8 -*-

'''
from federatedml.util import LOGGER
import tinyec.ec as ec

LOGGER.setLevel('DEBUG')

# 国家密码管理局：SM2椭圆曲线公钥密码算法推荐曲线参数
SM2_p = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
SM2_a = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
SM2_b = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
SM2_n = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
SM2_Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
SM2_Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
PARA_SIZE = 32  # 参数长度（字节）


# 转换为bytes，第二参数为字节数（可不填）
def to_byte(x, size=None):
    if isinstance(x, int):
        if size is None:  # 计算合适的字节数
            size = 0
            tmp = x >> 64
            while tmp:
                size += 8
                tmp >>= 64
            tmp = x >> (size << 3)
            while tmp:
                size += 1
                tmp >>= 8
        elif x >> (size << 3):  # 指定的字节数不够则截取低位
            x &= (1 << (size << 3)) - 1
        return x.to_bytes(size, byteorder='big')


class SM2:
    def __init__(self, p=SM2_p, a=SM2_a, b=SM2_b, n=SM2_n, G=(SM2_Gx, SM2_Gy), h=1,curve_key=None):
        field = ec.SubGroup(p, G, n, h)
        self.curve = ec.Curve(a, b, field)
        self.G = ec.Point(self.curve, G[0], G[1])
        keypair = ec.make_keypair(self.curve)
        self.private_key = curve_key or keypair.priv
        self.p, self.a, self.b, self.n, self.h = p, a, b, n, h

    def mult_point(self, k):
        return k * self.G

    def encrypt(self, hashvalue):
        LOGGER.info('!!!!sm2 encry start')
        int_hashvalue = int.from_bytes(hashvalue, byteorder='big')
        hash_to_curve = self.mult_point(int_hashvalue)
        c = (hash_to_curve.x + hash_to_curve.y) * self.private_key
        LOGGER.info('!!!!sm2 encry finish')
        return to_byte(c)

    def diffie_hellman(self, ciphertext):
        ciphertext = int.from_bytes(ciphertext, byteorder='big')
        c2 = ciphertext * self.private_key
        return to_byte(c2)

    def get_private_key(self):
        return self.private_key
'''


