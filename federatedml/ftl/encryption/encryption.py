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

import numpy as np

from arch.api.utils import log_utils
from federatedml.secureprotol.encrypt import PaillierEncrypt
from federatedml.secureprotol.fate_paillier import PaillierPublicKey, PaillierPrivateKey

LOGGER = log_utils.getLogger()


def generate_encryption_key_pair():
    paillierEncrypt = PaillierEncrypt()
    paillierEncrypt.generate_key()
    public_key = paillierEncrypt.get_public_key()
    private_key = paillierEncrypt.get_privacy_key()
    return public_key, private_key


def encrypt_array(public_key: PaillierPublicKey, A):
    encrypt_A = []
    for i in range(len(A)):
        encrypt_A.append(public_key.encrypt(float(A[i])))
    return np.array(encrypt_A)


def encrypt_matrix(public_key: PaillierPublicKey, A):
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    encrypt_A = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[i])):
            if len(A.shape) == 3:
                row.append([public_key.encrypt(float(A[i, j, k])) for k in range(len(A[i][j]))])
            else:
                row.append(public_key.encrypt(float(A[i, j])))
        encrypt_A.append(row)

    result = np.array(encrypt_A)
    if len(A.shape) == 1:
        result = np.squeeze(result, axis=0)
    return result


def encrypt_matmul(public_key: PaillierPublicKey, A, encrypted_B):
    """
     matrix multiplication between a plain matrix and an encrypted matrix

    :param public_key:
    :param A:
    :param encrypted_B:
    :return:
    """
    if A.shape[-1] != encrypted_B.shape[0]:
        LOGGER.debug("A and encrypted_B shape are not consistent")
        exit(1)
    # TODO: need a efficient way to do this?
    res = [[public_key.encrypt(0) for _ in range(encrypted_B.shape[1])] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(encrypted_B.shape[1]):
            for m in range(len(A[i])):
                res[i][j] += A[i][m] * encrypted_B[m][j]
    return np.array(res)


def encrypt_matmul_3(public_key: PaillierPublicKey, A, encrypted_B):
    if A.shape[0] != encrypted_B.shape[0]:
        LOGGER.debug("A and encrypted_B shape are not consistent: " + str(A.shape) + ":" + str(encrypted_B.shape))
        exit(1)
    res = []
    for i in range(len(A)):
        res.append(encrypt_matmul(public_key, A[i], encrypted_B[i]))
    return np.array(res)


def decrypt(private_key: PaillierPrivateKey, x):
    return private_key.decrypt(x)


def decrypt_scalar(private_key: PaillierPrivateKey, x):
    return private_key.decrypt(x)


def decrypt_array(private_key: PaillierPrivateKey, X):
    decrypt_x = []
    for i in range(X.shape[0]):
        elem = private_key.decrypt(X[i])
        decrypt_x.append(elem)
    return np.array(decrypt_x, dtype=np.float64)


def decrypt_matrix(private_key: PaillierPrivateKey, A):
    """
    decrypt matrix with dim 1, 2 or 3
    :param private_key:
    :param A:
    :return:
    """
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    decrypt_A = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[i])):
            if len(A.shape) == 3:
                row.append([private_key.decrypt(A[i, j, k]) for k in range(len(A[i][j]))])
            else:
                row.append(private_key.decrypt(A[i, j]))
        decrypt_A.append(row)

    result = np.array(decrypt_A, dtype=np.float64)
    if len(A.shape) == 1:
        result = np.squeeze(result, axis=0)
    return result

