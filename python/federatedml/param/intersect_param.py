#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import copy

from federatedml.param.base_param import BaseParam
from federatedml.util import consts, LOGGER


class EncodeParam(BaseParam):
    """
    Define the hash method for raw intersect method

    Parameters
    ----------
    salt: the src data string will be str = str + salt, default by empty string

    encode_method: str, the hash method of src data string, support md5, sha1, sha224, sha256, sha384, sha512, sm3, default by None

    base64: bool, if True, the result of hash will be changed to base64, default by False
    """

    def __init__(self, salt='', encode_method='none', base64=False):
        super().__init__()
        self.salt = salt
        self.encode_method = encode_method
        self.base64 = base64

    def check(self):
        if type(self.salt).__name__ != "str":
            raise ValueError(
                "encode param's salt {} not supported, should be str type".format(
                    self.salt))

        descr = "encode param's "

        self.encode_method = self.check_and_change_lower(self.encode_method,
                                                         ["none", consts.MD5, consts.SHA1, consts.SHA224,
                                                          consts.SHA256, consts.SHA384, consts.SHA512,
                                                          consts.SM3],
                                                         descr)

        if type(self.base64).__name__ != "bool":
            raise ValueError(
                "hash param's base64 {} not supported, should be bool type".format(self.base64))

        LOGGER.debug("Finish EncodeParam check!")
        LOGGER.warning(f"'EncodeParam' will be renamed to 'RawParam' in future release."
                       f"Please do not rely on current param naming in application.")
        return True


class RSAParam(BaseParam):
    """
    Define the hash method for RSA intersect method

    Parameters
    ----------
    salt: the src data string will be str = str + salt, default ''

    hash_method: str, the hash method of src data string, support sha256, sha384, sha512, sm3, default sha256

    final_hash_method: str, the hash method of result data string, support md5, sha1, sha224, sha256, sha384, sha512, sm3, default sha256

    split_calculation: bool, if True, Host & Guest split operations for faster performance, recommended on large data set

    random_base_fraction: positive float, if not None, generate (fraction * public key id count) of r for encryption and reuse generated r;
        note that value greater than 0.99 will be taken as 1, and value less than 0.01 will be rounded up to 0.01

    key_length: positive int, bit count of rsa key, default 1024

    """

    def __init__(self, salt='', hash_method='sha256',  final_hash_method='sha256',
                 split_calculation=False, random_base_fraction=None, key_length=1024):
        super().__init__()
        self.salt = salt
        self.hash_method = hash_method
        self.final_hash_method = final_hash_method
        self.split_calculation = split_calculation
        self.random_base_fraction = random_base_fraction
        self.key_length = key_length

    def check(self):
        if type(self.salt).__name__ != "str":
            raise ValueError(
                "rsa param's salt {} not supported, should be str type".format(
                    self.salt))

        descr = "rsa param's hash_method "
        self.hash_method = self.check_and_change_lower(self.hash_method,
                                                       [consts.SHA256, consts.SHA384, consts.SHA512, consts.SM3],
                                                       descr)

        descr = "rsa param's final_hash_method "
        self.final_hash_method = self.check_and_change_lower(self.final_hash_method,
                                                             [consts.MD5, consts.SHA1, consts.SHA224,
                                                              consts.SHA256, consts.SHA384, consts.SHA512,
                                                              consts.SM3],
                                                             descr)

        descr = "rsa param's split_calculation"
        self.check_boolean(self.split_calculation, descr)

        descr = "rsa param's random_base_fraction"
        if self.random_base_fraction:
            self.check_positive_number(self.random_base_fraction, descr)
            self.check_decimal_float(self.random_base_fraction, descr)

        descr = "rsa param's key_length"
        self.check_positive_integer(self.key_length, descr)

        LOGGER.debug("Finish RSAParam parameter check!")
        return True


class PHParam(BaseParam):
    """
    Define the hash method for PH intersect method

    Parameters
    ----------
    salt: the src data string will be str = str + salt, default ''

    hash_method: str, the hash method of src data string, support none, md5, sha1, sha 224, sha256, sha384, sha512, sm3,
        default sha256

    key_length: positive int, value >= 768 is suggested, the key length of the commutative cipher, default 1024

    """

    def __init__(self, salt='', hash_method='sha256', key_length=1024):
        super().__init__()
        self.salt = salt
        self.hash_method = hash_method
        self.key_length = key_length

    def check(self):
        if type(self.salt).__name__ != "str":
            raise ValueError(
                "ph param's salt {} not supported, should be str type".format(
                    self.salt))

        descr = "ph param's hash_method "
        self.hash_method = self.check_and_change_lower(self.hash_method,
                                                       ["none", consts.MD5, consts.SHA1, consts.SHA224,
                                                        consts.SHA256, consts.SHA384, consts.SHA512,
                                                        consts.SM3],
                                                       descr)

        descr = "ph param's key_length"
        self.check_positive_integer(self.key_length, descr)

        LOGGER.debug("Finish PHParam parameter check!")
        return True


class IntersectCache(BaseParam):
    def __init__(self, use_cache=False, id_type=consts.PHONE, encrypt_type=consts.SHA256):
        super().__init__()
        self.use_cache = use_cache
        self.id_type = id_type
        self.encrypt_type = encrypt_type

    def check(self):
        if type(self.use_cache).__name__ != "bool":
            raise ValueError(
                "IntersectCache param's use_cache {} not supported, should be bool type".format(
                    self.use_cache))

        descr = "intersect cache param's "
        self.check_and_change_lower(self.id_type,
                                    [consts.PHONE, consts.IMEI],
                                    descr)
        self.check_and_change_lower(self.encrypt_type,
                                    [consts.MD5, consts.SHA256],
                                    descr)


class IntersectParam(BaseParam):
    """
    Define the intersect method

    Parameters
    ----------
    intersect_method: str, it supports 'rsa', 'raw', and 'ph', default by 'raw'

    random_bit: positive int, it will define the encrypt length of rsa algorithm. It effective only for intersect_method is rsa

    sync_intersect_ids: bool. In rsa, 'synchronize_intersect_ids' is True means guest or host will send intersect results to the others, and False will not.
                            while in raw, 'synchronize_intersect_ids' is True means the role of "join_role" will send intersect results and the others will get them.
                            Default by True.

    join_role: str, role who joins ids, supports "guest" and "host" only and effective only for raw. If it is "guest", the host will send its ids to guest and find the intersection of
               ids in guest; if it is "host", the guest will send its ids to host. Default by "guest".

    with_encode: bool, if True, it will use hash method for intersect ids. Effective only for "raw".

    encode_params: EncodeParam, it effective only for with_encode is True

    rsa_params: RSAParam, effective for rsa method only

    only_output_key: bool, if false, the results of intersection will include key and value which from input data; if true, it will just include key from input
                     data and the value will be empty or some useless character like "intersect_id"

    repeated_id_process: bool, if true, intersection will process the ids which can be repeatable

    repeated_id_owner: str, which role has the repeated ids

    with_sample_id: bool, data with sample id or not, default False; in ver 1.7 and above, this param is ignored

    join_method: str, choose 'inner_join' or 'left_join', if 'left_join', participants will all include repeated id owner's (imputed) ids in output, default 'inner_join'

    new_join_id: bool, whether to generate new id for repeated_id_owners' ids, only effective when join_method is 'left_join', default False

    ph_params: PHParam, effective for ph method only

    cardinality_only: boolean, whether to output intersection count(cardinality) only; if sync_cardinality is True,
        then sync cardinality count with host(s)

    sync_cardinality: boolean, whether to sync cardinality with all participants, default False,
        only effective when cardinality_only set to True

    """

    def __init__(self, intersect_method: str = consts.RAW, random_bit=128, sync_intersect_ids=True,
                 join_role=consts.GUEST,
                 with_encode=False, only_output_key=False, encode_params=EncodeParam(),
                 rsa_params=RSAParam(),
                 intersect_cache_param=IntersectCache(), repeated_id_process=False, repeated_id_owner=consts.GUEST,
                 with_sample_id=False, join_method=consts.INNER_JOIN, new_join_id=False,
                 allow_info_share: bool = False, info_owner=consts.GUEST, ph_params=PHParam(),
                 cardinality_only: bool = False, sync_cardinality: bool = False):
        super().__init__()
        self.intersect_method = intersect_method
        self.random_bit = random_bit
        self.sync_intersect_ids = sync_intersect_ids
        self.join_role = join_role
        self.with_encode = with_encode
        self.encode_params = copy.deepcopy(encode_params)
        self.rsa_params = copy.deepcopy(rsa_params)
        self.only_output_key = only_output_key
        self.intersect_cache_param = intersect_cache_param
        self.repeated_id_process = repeated_id_process
        self.repeated_id_owner = repeated_id_owner
        self.allow_info_share = allow_info_share
        self.info_owner = info_owner
        self.with_sample_id = with_sample_id
        self.join_method = join_method
        self.new_join_id = new_join_id
        self.ph_params = ph_params
        self.cardinality_only = cardinality_only
        self.sync_cardinality = sync_cardinality

    def check(self):
        descr = "intersect param's "

        self.intersect_method = self.check_and_change_lower(self.intersect_method,
                                                            [consts.RSA, consts.RAW, consts.PH],
                                                            f"{descr}intersect_method")

        if type(self.random_bit).__name__ not in ["int"]:
            raise ValueError("intersect param's random_bit {} not supported, should be positive integer".format(
                self.random_bit))

        if type(self.sync_intersect_ids).__name__ != "bool":
            raise ValueError(
                "intersect param's sync_intersect_ids {} not supported, should be bool type".format(
                    self.sync_intersect_ids))

        self.join_role = self.check_and_change_lower(self.join_role,
                                                     [consts.GUEST, consts.HOST],
                                                     f"{descr}join_role")

        if type(self.with_encode).__name__ != "bool":
            raise ValueError(
                "intersect param's with_encode {} not supported, should be bool type".format(
                    self.with_encode))

        if type(self.only_output_key).__name__ != "bool":
            raise ValueError(
                "intersect param's only_output_key {} not supported, should be bool type".format(
                    self.only_output_key))

        if type(self.repeated_id_process).__name__ != "bool":
            raise ValueError(
                "intersect param's repeated_id_process {} not supported, should be bool type".format(
                    self.repeated_id_process))

        self.repeated_id_owner = self.check_and_change_lower(self.repeated_id_owner,
                                                             [consts.GUEST],
                                                             f"{descr}repeated_id_owner")

        if type(self.allow_info_share).__name__ != "bool":
            raise ValueError(
                "intersect param's allow_info_sync {} not supported, should be bool type".format(
                    self.allow_info_share))

        self.info_owner = self.check_and_change_lower(self.info_owner,
                                                      [consts.GUEST, consts.HOST],
                                                      f"{descr}info_owner")

        self.check_boolean(self.with_sample_id, descr+"with_sample_id")
        self.join_method = self.check_and_change_lower(self.join_method, [consts.INNER_JOIN, consts.LEFT_JOIN],
                                                       f"{descr}join_method")
        self.check_boolean(self.new_join_id, descr+"new_join_id")

        if self.with_sample_id:
            LOGGER.warning(f"with_sample_id is ignored.")

        if self.join_method==consts.LEFT_JOIN:
            if not self.sync_intersect_ids:
                raise ValueError(f"Cannot perform left join without sync intersect ids or info share")
            if not self.allow_info_share:
                LOGGER.warning(f"when performing left_join, allow_info_share is always True.")

        self.encode_params.check()
        self.rsa_params.check()
        self.ph_params.check()
        self.check_boolean(self.cardinality_only, descr+"cardinality_only")
        self.check_boolean(self.sync_cardinality, descr+"sync_cardinality")
        LOGGER.debug("Finish intersect parameter check!")
        return True
