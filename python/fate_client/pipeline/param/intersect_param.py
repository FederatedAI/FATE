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

from pipeline.param.base_param import BaseParam
from pipeline.param import consts

DEFAULT_RANDOM_BIT = 128


class EncodeParam(BaseParam):
    """
    Define the hash method for raw intersect method

    Parameters
    ----------
    salt: str
        the src data string will be str = str + salt, default by empty string

    encode_method: {"none", "md5", "sha1", "sha224", "sha256", "sha384", "sha512", "sm3"}
        the hash method of src data string, support md5, sha1, sha224, sha256, sha384, sha512, sm3, default by None

    base64: bool
        if True, the result of hash will be changed to base64, default by False
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

        return True


class RAWParam(BaseParam):
    """
    Specify parameters for raw intersect method

    Parameters
    ----------
    use_hash: bool
        whether to hash ids for raw intersect

    salt: str
        the src data string will be str = str + salt, default by empty string

    hash_method: str
        the hash method of src data string, support md5, sha1, sha224, sha256, sha384, sha512, sm3, default by None

    base64: bool
        if True, the result of hash will be changed to base64, default by False

    join_role: {"guest", "host"}
        role who joins ids, supports "guest" and "host" only and effective only for raw.
        If it is "guest", the host will send its ids to guest and find the intersection of
        ids in guest; if it is "host", the guest will send its ids to host. Default by "guest";
    """

    def __init__(self, use_hash=False, salt='', hash_method='none', base64=False, join_role=consts.GUEST):
        super().__init__()
        self.use_hash = use_hash
        self.salt = salt
        self.hash_method = hash_method
        self.base64 = base64
        self.join_role = join_role

    def check(self):
        descr = "raw param's "

        self.check_boolean(self.use_hash, f"{descr}use_hash")
        self.check_string(self.salt, f"{descr}salt")

        self.hash_method = self.check_and_change_lower(self.hash_method,
                                                       ["none", consts.MD5, consts.SHA1, consts.SHA224,
                                                        consts.SHA256, consts.SHA384, consts.SHA512,
                                                        consts.SM3],
                                                       f"{descr}hash_method")

        self.check_boolean(self.base64, f"{descr}base_64")
        self.join_role = self.check_and_change_lower(self.join_role, [consts.GUEST, consts.HOST], f"{descr}join_role")

        return True


class RSAParam(BaseParam):
    """
    Specify parameters for RSA intersect method

    Parameters
    ----------
    salt: str
        the src data string will be str = str + salt, default ''

    hash_method: str
        the hash method of src data string, support sha256, sha384, sha512, sm3, default sha256

    final_hash_method: str
        the hash method of result data string, support md5, sha1, sha224, sha256, sha384, sha512, sm3, default sha256

    split_calculation: bool
        if True, Host & Guest split operations for faster performance, recommended on large data set

    random_base_fraction: positive float
        if not None, generate (fraction * public key id count) of r for encryption and reuse generated r;
        note that value greater than 0.99 will be taken as 1, and value less than 0.01 will be rounded up to 0.01

    key_length: int
        value >= 1024, bit count of rsa key, default 1024

    random_bit: positive int
        it will define the size of blinding factor in rsa algorithm, default 128

    """

    def __init__(self, salt='', hash_method='sha256', final_hash_method='sha256',
                 split_calculation=False, random_base_fraction=None, key_length=consts.DEFAULT_KEY_LENGTH,
                 random_bit=DEFAULT_RANDOM_BIT):
        super().__init__()
        self.salt = salt
        self.hash_method = hash_method
        self.final_hash_method = final_hash_method
        self.split_calculation = split_calculation
        self.random_base_fraction = random_base_fraction
        self.key_length = key_length
        self.random_bit = random_bit

    def check(self):
        descr = "rsa param's "
        self.check_string(self.salt, f"{descr}salt")

        self.hash_method = self.check_and_change_lower(self.hash_method,
                                                       [consts.SHA256, consts.SHA384, consts.SHA512, consts.SM3],
                                                       f"{descr}hash_method")

        self.final_hash_method = self.check_and_change_lower(self.final_hash_method,
                                                             [consts.MD5, consts.SHA1, consts.SHA224,
                                                              consts.SHA256, consts.SHA384, consts.SHA512,
                                                              consts.SM3],
                                                             f"{descr}final_hash_method")

        self.check_boolean(self.split_calculation, f"{descr}split_calculation")

        if self.random_base_fraction:
            self.check_positive_number(self.random_base_fraction, descr)
            self.check_decimal_float(self.random_base_fraction, f"{descr}random_base_fraction")

        self.check_positive_integer(self.key_length, f"{descr}key_length")
        if self.key_length < 1024:
            raise ValueError(f"key length must be >= 1024")
        self.check_positive_integer(self.random_bit, f"{descr}random_bit")

        return True


class DHParam(BaseParam):
    """
    Define the hash method for DH intersect method

    Parameters
    ----------
    salt: str
        the src data string will be str = str + salt, default ''

    hash_method: str
        the hash method of src data string, support none, md5, sha1, sha 224, sha256, sha384, sha512, sm3, default sha256

    key_length: int, value >= 1024
        the key length of the commutative cipher p, default 1024

    """

    def __init__(self, salt='', hash_method='sha256', key_length=consts.DEFAULT_KEY_LENGTH):
        super().__init__()
        self.salt = salt
        self.hash_method = hash_method
        self.key_length = key_length

    def check(self):
        descr = "dh param's "
        self.check_string(self.salt, f"{descr}salt")

        self.hash_method = self.check_and_change_lower(self.hash_method,
                                                       ["none", consts.MD5, consts.SHA1, consts.SHA224,
                                                        consts.SHA256, consts.SHA384, consts.SHA512,
                                                        consts.SM3],
                                                       f"{descr}hash_method")

        self.check_positive_integer(self.key_length, f"{descr}key_length")
        if self.key_length < 1024:
            raise ValueError(f"key length must be >= 1024")

        return True


class ECDHParam(BaseParam):
    """
    Define the hash method for ECDH intersect method

    Parameters
    ----------
    salt: str
        the src id will be str = str + salt, default ''

    hash_method: str
        the hash method of src id, support sha256, sha384, sha512, sm3, default sha256

    curve: str
        the name of curve, currently only support 'curve25519', which offers 128 bits of security
    """

    def __init__(self, salt='', hash_method='sha256', curve=consts.CURVE25519):
        super().__init__()
        self.salt = salt
        self.hash_method = hash_method
        self.curve = curve

    def check(self):
        descr = "ecdh param's "
        self.check_string(self.salt, f"{descr}salt")

        self.hash_method = self.check_and_change_lower(self.hash_method,
                                                       [consts.SHA256, consts.SHA384, consts.SHA512,
                                                        consts.SM3],
                                                       f"{descr}hash_method")

        self.curve = self.check_and_change_lower(self.curve, [consts.CURVE25519], f"{descr}curve")

        return True


class IntersectCache(BaseParam):
    def __init__(self, use_cache=False, id_type=consts.PHONE, encrypt_type=consts.SHA256):
        """

        Parameters
        ----------
        use_cache: whether to use cached ids; with ver1.7 and above, this param is ignored
        id_type: with ver1.7 and above, this param is ignored
        encrypt_type: with ver1.7 and above, this param is ignored
        """
        super().__init__()
        self.use_cache = use_cache
        self.id_type = id_type
        self.encrypt_type = encrypt_type

    def check(self):
        descr = "intersect_cache param's "
        # self.check_boolean(self.use_cache, f"{descr}use_cache")

        self.check_and_change_lower(self.id_type,
                                    [consts.PHONE, consts.IMEI],
                                    f"{descr}id_type")
        self.check_and_change_lower(self.encrypt_type,
                                    [consts.MD5, consts.SHA256],
                                    f"{descr}encrypt_type")


class IntersectPreProcessParam(BaseParam):
    """
    Specify parameters for pre-processing and cardinality-only mode

    Parameters
    ----------
    false_positive_rate: float
        initial target false positive rate when creating Bloom Filter,
        must be <= 0.5, default 1e-3

    encrypt_method: str
        encrypt method for encrypting id when performing cardinality_only task,
        supports rsa only, default rsa;
        specify rsa parameter setting with RSAParam

    hash_method: str
        the hash method for inserting ids, support md5, sha1, sha 224, sha256, sha384, sha512, sm3,
        default sha256

    preprocess_method: str
        the hash method for encoding ids before insertion into filter, default sha256,
        only effective for preprocessing

    preprocess_salt: str
        salt to be appended to hash result by preprocess_method before insertion into filter,
        default '', only effective for preprocessing

    random_state: int
        seed for random salt generator when constructing hash functions,
        salt is appended to hash result by hash_method when performing insertion, default None

    filter_owner: str
        role that constructs filter, either guest or host, default guest,
        only effective for preprocessing

    """

    def __init__(self, false_positive_rate=1e-3, encrypt_method=consts.RSA, hash_method='sha256',
                 preprocess_method='sha256', preprocess_salt='', random_state=None, filter_owner=consts.GUEST):
        super().__init__()
        self.false_positive_rate = false_positive_rate
        self.encrypt_method = encrypt_method
        self.hash_method = hash_method
        self.preprocess_method = preprocess_method
        self.preprocess_salt = preprocess_salt
        self.random_state = random_state
        self.filter_owner = filter_owner

    def check(self):
        descr = "intersect preprocess param's false_positive_rate "
        self.check_decimal_float(self.false_positive_rate, descr)
        self.check_positive_number(self.false_positive_rate, descr)
        if self.false_positive_rate > 0.5:
            raise ValueError(f"{descr} must be positive float no greater than 0.5")

        descr = "intersect preprocess param's encrypt_method "
        self.encrypt_method = self.check_and_change_lower(self.encrypt_method, [consts.RSA], descr)

        descr = "intersect preprocess param's random_state "
        if self.random_state:
            self.check_nonnegative_number(self.random_state, descr)

        descr = "intersect preprocess param's hash_method "
        self.hash_method = self.check_and_change_lower(self.hash_method,
                                                       [consts.MD5, consts.SHA1, consts.SHA224,
                                                        consts.SHA256, consts.SHA384, consts.SHA512,
                                                        consts.SM3],
                                                       descr)
        descr = "intersect preprocess param's preprocess_salt "
        self.check_string(self.preprocess_salt, descr)

        descr = "intersect preprocess param's preprocess_method "
        self.preprocess_method = self.check_and_change_lower(self.preprocess_method,
                                                             [consts.MD5, consts.SHA1, consts.SHA224,
                                                              consts.SHA256, consts.SHA384, consts.SHA512,
                                                              consts.SM3],
                                                             descr)

        descr = "intersect preprocess param's filter_owner "
        self.filter_owner = self.check_and_change_lower(self.filter_owner,
                                                        [consts.GUEST, consts.HOST],
                                                        descr)

        return True


class IntersectParam(BaseParam):
    """
    Define the intersect method

    Parameters
    ----------
    intersect_method: str
        it supports 'rsa', 'raw', 'dh', default by 'rsa'

    random_bit: positive int
        it will define the size of blinding factor in rsa algorithm, default 128
        note that this param will be deprecated in future, please use random_bit in RSAParam instead

    sync_intersect_ids: bool
        In rsa, 'sync_intersect_ids' is True means guest or host will send intersect results to the others, and False will not.
        while in raw, 'sync_intersect_ids' is True means the role of "join_role" will send intersect results and the others will get them.
        Default by True.

    join_role: str
        role who joins ids, supports "guest" and "host" only and effective only for raw.
        If it is "guest", the host will send its ids to guest and find the intersection of
        ids in guest; if it is "host", the guest will send its ids to host. Default by "guest";
        note this param will be deprecated in future version, please use 'join_role' in raw_params instead

    only_output_key: bool
        if false, the results of intersection will include key and value which from input data; if true, it will just include key from input
        data and the value will be empty or filled by uniform string like "intersect_id"

    with_encode: bool
        if True, it will use hash method for intersect ids, effective for raw method only;
        note that this param will be deprecated in future version, please use 'use_hash' in raw_params;
        currently if this param is set to True,
        specification by 'encode_params' will be taken instead of 'raw_params'.

    encode_params: EncodeParam
        effective only when with_encode is True;
        this param will be deprecated in future version, use 'raw_params' in future implementation

    raw_params: RAWParam
        effective for raw method only

    rsa_params: RSAParam
        effective for rsa method only

    dh_params: DHParam
        effective for dh method only

    ecdh_params: ECDHParam
        effective for ecdh method only

    join_method: {'inner_join', 'left_join'}
        if 'left_join', participants will all include sample_id_generator's (imputed) ids in output,
        default 'inner_join'

    new_sample_id: bool
        whether to generate new id for sample_id_generator's ids,
        only effective when join_method is 'left_join' or when input data are instance with match id,
        default False

    sample_id_generator: str
        role whose ids are to be kept,
        effective only when join_method is 'left_join' or when input data are instance with match id,
        default 'guest'

    intersect_cache_param: IntersectCacheParam
        specification for cache generation,
        with ver1.7 and above, this param is ignored.

    run_cache: bool
        whether to store Host's encrypted ids, only valid when intersect method is 'rsa', 'dh', or 'ecdh', default False

    cardinality_only: bool
        whether to output intersection count(cardinality);
        if sync_cardinality is True, then sync cardinality count with host(s)

    cardinality_method: string
        specify which intersect method to use for coutning cardinality, default "ecdh";
        note that with "rsa", estimated cardinality will be produced;
        while "dh" method outputs exact cardinality, it only supports single-host task

    sync_cardinality: bool
        whether to sync cardinality with all participants, default False,
        only effective when cardinality_only set to True

    run_preprocess: bool
        whether to run preprocess process, default False

    intersect_preprocess_params: IntersectPreProcessParam
        used for preprocessing and cardinality_only mode

    repeated_id_process: bool
        if true, intersection will process the ids which can be repeatable;
        in ver 1.7 and above,repeated id process
        will be automatically applied to data with instance id, this param will be ignored

    repeated_id_owner: str
        which role has the repeated id; in ver 1.7 and above, this param is ignored

    allow_info_share: bool
        in ver 1.7 and above, this param is ignored

    info_owner: str
        in ver 1.7 and above, this param is ignored

    with_sample_id: bool
        data with sample id or not, default False; in ver 1.7 and above, this param is ignored
    """

    def __init__(self, intersect_method: str = consts.RSA, random_bit=DEFAULT_RANDOM_BIT, sync_intersect_ids=True,
                 join_role=consts.GUEST, only_output_key: bool = False,
                 with_encode=False, encode_params=EncodeParam(),
                 raw_params=RAWParam(), rsa_params=RSAParam(), dh_params=DHParam(), ecdh_params=ECDHParam(),
                 join_method=consts.INNER_JOIN, new_sample_id: bool = False, sample_id_generator=consts.GUEST,
                 intersect_cache_param=IntersectCache(), run_cache: bool = False,
                 cardinality_only: bool = False, sync_cardinality: bool = False, cardinality_method=consts.ECDH,
                 run_preprocess: bool = False,
                 intersect_preprocess_params=IntersectPreProcessParam(),
                 repeated_id_process=False, repeated_id_owner=consts.GUEST,
                 with_sample_id=False, allow_info_share: bool = False, info_owner=consts.GUEST):
        super().__init__()
        self.intersect_method = intersect_method
        self.random_bit = random_bit
        self.sync_intersect_ids = sync_intersect_ids
        self.join_role = join_role
        self.with_encode = with_encode
        self.encode_params = copy.deepcopy(encode_params)
        self.raw_params = copy.deepcopy(raw_params)
        self.rsa_params = copy.deepcopy(rsa_params)
        self.only_output_key = only_output_key
        self.sample_id_generator = sample_id_generator
        self.intersect_cache_param = copy.deepcopy(intersect_cache_param)
        self.run_cache = run_cache
        self.repeated_id_process = repeated_id_process
        self.repeated_id_owner = repeated_id_owner
        self.allow_info_share = allow_info_share
        self.info_owner = info_owner
        self.with_sample_id = with_sample_id
        self.join_method = join_method
        self.new_sample_id = new_sample_id
        self.dh_params = copy.deepcopy(dh_params)
        self.cardinality_only = cardinality_only
        self.sync_cardinality = sync_cardinality
        self.cardinality_method = cardinality_method
        self.run_preprocess = run_preprocess
        self.intersect_preprocess_params = copy.deepcopy(intersect_preprocess_params)
        self.ecdh_params = copy.deepcopy(ecdh_params)

    def check(self):
        descr = "intersect param's "

        self.intersect_method = self.check_and_change_lower(self.intersect_method,
                                                            [consts.RSA, consts.RAW, consts.DH, consts.ECDH],
                                                            f"{descr}intersect_method")

        self.check_positive_integer(self.random_bit, f"{descr}random_bit")
        self.check_boolean(self.sync_intersect_ids, f"{descr}intersect_ids")
        self.join_role = self.check_and_change_lower(self.join_role,
                                                     [consts.GUEST, consts.HOST],
                                                     f"{descr}join_role")
        self.check_boolean(self.with_encode, f"{descr}with_encode")
        self.check_boolean(self.only_output_key, f"{descr}only_output_key")

        self.join_method = self.check_and_change_lower(self.join_method, [consts.INNER_JOIN, consts.LEFT_JOIN],
                                                       f"{descr}join_method")
        self.check_boolean(self.new_sample_id, f"{descr}new_sample_id")
        self.sample_id_generator = self.check_and_change_lower(self.sample_id_generator,
                                                               [consts.GUEST, consts.HOST],
                                                               f"{descr}sample_id_generator")

        if self.join_method == consts.LEFT_JOIN:
            if not self.sync_intersect_ids:
                raise ValueError(f"Cannot perform left join without sync intersect ids")

        self.check_boolean(self.run_cache, f"{descr} run_cache")
        self.encode_params.check()
        self.raw_params.check()
        self.rsa_params.check()
        self.dh_params.check()
        self.ecdh_params.check()
        self.check_boolean(self.cardinality_only, f"{descr}cardinality_only")
        self.check_boolean(self.sync_cardinality, f"{descr}sync_cardinality")
        self.check_boolean(self.run_preprocess, f"{descr}run_preprocess")
        self.intersect_preprocess_params.check()
        if self.cardinality_only:
            if self.cardinality_method not in [consts.RSA, consts.DH, consts.ECDH]:
                raise ValueError(f"cardinality-only mode only support rsa, dh, ecdh.")
            if self.cardinality_method == consts.RSA and self.rsa_params.split_calculation:
                raise ValueError(f"cardinality-only mode only supports unified calculation.")

        if self.run_preprocess:
            if self.intersect_preprocess_params.false_positive_rate < 0.01:
                raise ValueError(f"for preprocessing ids, false_positive_rate must be no less than 0.01")
            if self.cardinality_only:
                raise ValueError(f"cardinality_only mode cannot run preprocessing.")
        if self.run_cache:
            if self.intersect_method not in [consts.RSA, consts.DH, consts.ECDH]:
                raise ValueError(f"Only rsa, dh, ecdh method supports cache.")
            if self.intersect_method == consts.RSA and self.rsa_params.split_calculation:
                raise ValueError(f"RSA split_calculation does not support cache.")
            if self.cardinality_only:
                raise ValueError(f"Cache is not available for cardinality_only mode.")
            if self.run_preprocess:
                raise ValueError(f"Preprocessing does not support cache.")
        return True
