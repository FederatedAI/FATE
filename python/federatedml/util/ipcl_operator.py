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

from numpy import ndarray
from federatedml.util import LOGGER

try:
    from ipcl_python import PaillierEncryptedNumber as IpclPaillierEncryptedNumber
    from ipcl_python.bindings.ipcl_bindings import ipclCipherText
except ImportError:
    LOGGER.info("ipcl_python failed to import")
    pass


def get_coeffs(weights):
    """
    IPCL encrypts all weights (coefficients and intercept) into one single encrypted number.
    This function allows to get an IPCL encrypted number which contains all coefficents but intercept.

    Args:
        weights (IpclPaillierEncryptedNumber): all model weights in one encrypted number

    Returns:
        (IpclPaillierEncryptedNumber): all coefficients in one encrypted number
    """
    coeff_num = weights.__len__() - 1
    pub_key = weights.public_key

    bn = []
    exp = []
    for i in range(coeff_num):
        bn.append(weights.ciphertextBN(i))
        exp.append(weights.exponent(i))
    ct = ipclCipherText(pub_key.pubkey, bn)
    return IpclPaillierEncryptedNumber(pub_key, ct, exp, coeff_num)


def get_intercept(weights):
    """
    IPCL encrypts all weights (coefficients and intercept) into one single encrypted number.
    This function allows to get the encrypted number of intercept.

    Args:
        weights (IpclPaillierEncryptedNumber): all model weights in one encrypted number

    Returns:
        (IpclPaillierEncryptedNumber): IPCL encrypted number of intercept
    """
    coeff_num = weights.__len__() - 1
    pub_key = weights.public_key
    bn = [weights.ciphertextBN(coeff_num)]
    exp = [weights.exponent(coeff_num)]
    ct = ipclCipherText(pub_key.pubkey, bn)
    return IpclPaillierEncryptedNumber(pub_key, ct, exp, 1)


def merge_encrypted_number_array(values):
    """
    Put all IPCL encrypted numbers of a 1-d array into one encrypted number.

    Args:
        values (numpy.ndarray, list): an array of multiple IPCL encrypted numbers

    Returns:
        (IpclPaillierEncryptedNumber): one encrypted number contains all values
    """
    assert isinstance(values, (list, ndarray))
    pub_key = values[0].public_key
    bn, exp = [], []
    for i in range(len(values)):
        assert values[i].__len__() == 1
        bn.append(values[i].ciphertextBN(0))
        exp.append(values[i].exponent(0))
    ct = ipclCipherText(pub_key.pubkey, bn)
    return IpclPaillierEncryptedNumber(pub_key, ct, exp, len(values))
