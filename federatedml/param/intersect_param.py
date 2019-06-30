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
                "encode param's base64 {} not supported, should be bool type".format(encode_param.base64))

        LOGGER.debug("Finish encode parameter check!")
        return True


class IntersectParam(object):
    """
    Define the intersect method

    Parameters
    ----------
    intersect_method: str, it supports 'rsa' and 'raw', default by 'raw'

    random_bit: positive int, it will define the encrypt length of rsa algorithm. It effective only for intersect_method is rsa

    is_send_intersect_ids: bool. In rsa, 'is_send_intersect_ids' is True means guest will send intersect results to host, and False will not.
                            while in raw, 'is_send_intersect_ids' is True means the role of "join_role" will send intersect results and the other will get them.
                            Default by True.

    is_get_intersect_ids: bool, In rsa, it will get the results from other. It effective only for rsa and only be True will other's 'is_send_intersect_ids' is True.Default by True

    join_role: str, it supports "guest" and "host" only and effective only for raw. If it is "guest", the host will send its ids to guest and find the intersection of
                ids in guest; if it is "host", the guest will send its ids. Default by "guest".

    with_encode: bool, if True, it will use encode method for intersect ids. It effective only for "raw".

    encode_params: EncodeParam, it effective only for with_encode is True

    only_output_key: bool, if true, the results of intersection will include key and value which from input data; if false, it will just include key from input
                    data and the value will be empty or some useless character like "intersect_id"
    """

    def __init__(self, intersect_method=consts.RAW, random_bit=128, is_send_intersect_ids=True,
                 is_get_intersect_ids=True, join_role="guest", with_encode=False, encode_params=EncodeParam(),
                 only_output_key=False):
        self.intersect_method = intersect_method
        self.random_bit = random_bit
        self.is_send_intersect_ids = is_send_intersect_ids
        self.is_get_intersect_ids = is_get_intersect_ids
        self.join_role = join_role
        self.with_encode = with_encode
        self.encode_params = copy.deepcopy(encode_params)
        self.only_output_key = only_output_key

    def check_param(self):
        descr = "intersect param's"

        self.intersect_method = self.check_and_change_lower(self.intersect_method,
                                                                  [consts.RSA, consts.RAW],
                                                                  descr)

        if type(self.random_bit).__name__ not in ["int"]:
            raise ValueError("intersect param's random_bit {} not supported, should be positive integer".format(
                self.random_bit))

        if type(self.is_send_intersect_ids).__name__ != "bool":
            raise ValueError(
                "intersect param's is_send_intersect_ids {} not supported, should be bool type".format(
                    self.is_send_intersect_ids))

        if type(self.is_get_intersect_ids).__name__ != "bool":
            raise ValueError(
                "intersect param's is_get_intersect_ids {} not supported, should be bool type".format(
                    self.is_get_intersect_ids))

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
                    self.is_send_intersect_ids))

        self.encode_params.check()
        LOGGER.debug("Finish intersect parameter check!")
        return True

