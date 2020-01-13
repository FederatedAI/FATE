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

from arch.api.utils.log_utils import getLogger
from federatedml.param.base_param import BaseParam
from federatedml.util import consts

LOGGER = getLogger()


class EncodeParam(BaseParam):
    """
    Define the encode method

    Parameters
    ----------
    salt: the src data string will be str = str + salt, default by empty string

    encode_method: str, the encode method of src data string, it support md5, sha1, sha224, sha256, sha384, sha512, default by None

    base64: bool, if True, the result of encode will be changed to base64, default by False
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
                                                         ["none", "md5", "sha1", "sha224", "sha256", "sha384",
                                                          "sha512"],
                                                         descr)

        if type(self.base64).__name__ != "bool":
            raise ValueError(
                "encode param's base64 {} not supported, should be bool type".format(self.base64))

        LOGGER.debug("Finish encode parameter check!")
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
                "encode param's salt {} not supported, should be bool type".format(
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
    intersect_method: str, it supports 'rsa' and 'raw', default by 'raw'

    random_bit: positive int, it will define the encrypt length of rsa algorithm. It effective only for intersect_method is rsa

    sync_intersect_ids: bool. In rsa, 'synchronize_intersect_ids' is True means guest or host will send intersect results to the others, and False will not.
                            while in raw, 'synchronize_intersect_ids' is True means the role of "join_role" will send intersect results and the others will get them.
                            Default by True.

    join_role: str, it supports "guest" and "host" only and effective only for raw. If it is "guest", the host will send its ids to guest and find the intersection of
                ids in guest; if it is "host", the guest will send its ids. Default by "guest".

    with_encode: bool, if True, it will use encode method for intersect ids. It effective only for "raw".

    encode_params: EncodeParam, it effective only for with_encode is True

    only_output_key: bool, if false, the results of intersection will include key and value which from input data; if true, it will just include key from input
                    data and the value will be empty or some useless character like "intersect_id"

    repeated_id_process: bool, if true, intersection will process the ids which can be repeatable

    repeated_id_owner: str, which role has the repeated ids
    """

    def __init__(self, intersect_method=consts.RAW, random_bit=128, sync_intersect_ids=True, join_role="guest",
                 with_encode=False, only_output_key=False, encode_params=EncodeParam(),
                 intersect_cache_param=IntersectCache(), repeated_id_process=False, repeated_id_owner="guest"):
        super().__init__()
        self.intersect_method = intersect_method
        self.random_bit = random_bit
        self.sync_intersect_ids = sync_intersect_ids
        self.join_role = join_role
        self.with_encode = with_encode
        self.encode_params = copy.deepcopy(encode_params)
        self.only_output_key = only_output_key
        self.intersect_cache_param = intersect_cache_param
        self.repeated_id_process = repeated_id_process
        self.repeated_id_owner = repeated_id_owner

    def check(self):
        descr = "intersect param's"

        self.intersect_method = self.check_and_change_lower(self.intersect_method,
                                                            [consts.RSA, consts.RAW],
                                                            descr)

        if type(self.random_bit).__name__ not in ["int"]:
            raise ValueError("intersect param's random_bit {} not supported, should be positive integer".format(
                self.random_bit))

        if type(self.sync_intersect_ids).__name__ != "bool":
            raise ValueError(
                "intersect param's sync_intersect_ids {} not supported, should be bool type".format(
                    self.sync_intersect_ids))

        self.join_role = self.check_and_change_lower(self.join_role,
                                                     [consts.GUEST, consts.HOST],
                                                     descr)

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
                                                             descr)

        self.encode_params.check()
        LOGGER.debug("Finish intersect parameter check!")
        return True
