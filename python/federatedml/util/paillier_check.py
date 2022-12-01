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

from numpy import ndarray, ndim
from federatedml.util import LOGGER
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber

ipcl_enabled = False
try:
    from ipcl_python import PaillierEncryptedNumber as IpclPaillierEncryptedNumber
    ipcl_enabled = True
except ImportError:
    LOGGER.info("ipcl_python failed to import")
    pass


def is_encrypted_number(values, encrypted_type):
    if isinstance(values, ndarray):
        return isinstance(values.item(0), encrypted_type)
    elif isinstance(values, list):
        return isinstance(values[0], encrypted_type)
    else:
        return isinstance(values, encrypted_type)


def is_fate_paillier_encrypted_number(values):
    return is_encrypted_number(values, PaillierEncryptedNumber)


def is_ipcl_encrypted_number(values):
    if ipcl_enabled:
        return is_encrypted_number(values, IpclPaillierEncryptedNumber)
    return False


def is_paillier_encrypted_number(values):
    return is_fate_paillier_encrypted_number(values) or is_ipcl_encrypted_number(values)


def is_single_ipcl_encrypted_number(values):
    """
    Return True if input numpy array contains only one IPCL encrypted number, not a list

    Args:
        values (numpy.ndarray)
    """
    if ipcl_enabled and isinstance(values, ndarray):
        return ndim(values) == 0 and isinstance(values.item(0), IpclPaillierEncryptedNumber)
    return False
