import numpy as np
from federatedml.nn.hetero_nn.backend.paillier_tensor import PaillierTensor
from arch.api.session import init
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.nn.hetero_nn.util import random_number_generator

init()

arr1 = np.ones((10, 1, 3))
arr1[0] = np.array([[2, 3, 4]])
arr2 = np.ones((10, 3, 3))
arr3 = np.ones([1, 1, 3])

arr4 = np.ones([50, 1])
arr5 = np.ones([32])

pt = PaillierTensor(ori_data=arr1)
pt2 = PaillierTensor(ori_data=arr2)
pt3 = PaillierTensor(ori_data=arr3)

pt4 = PaillierTensor(ori_data=arr4)
pt5 = PaillierTensor(ori_data=arr5)

encrypter = PaillierEncrypt()
encrypter.generate_key(EncryptParam().key_length)
encrypted_calculator = EncryptModeCalculator(encrypter,
                                             EncryptedModeCalculatorParam().mode,
                                             EncryptedModeCalculatorParam().re_encrypted_rate)

rs1 = pt * arr2
rs2 = pt * pt2

rs3 = pt.matmul_3d(pt2)
enpt = pt2.encrypt(encrypted_calculator)
enrs = enpt.matmul_3d(arr1, multiply='right')

rng_generator = random_number_generator.RandomNumberGenerator()

enpt2 = pt4.encrypt(encrypted_calculator)
random_num = rng_generator.generate_random_number(enpt2.shape)
# rs4 = enpt.matmul_3d(pt2)
# dept = rs4.decrypt(encrypter)
#
# pt4 = pt4.encrypt(encrypted_calculator)
#
# # rs5 = pt4.fast_matmul_2d(pt5)
#
# a = np.array([[1, 2], [3, 4]])
# rs = PaillierTensor(a).fast_matmul_2d(a.transpose())