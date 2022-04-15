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


def int_to_bytes(integer):
    """
    Convert an int to bytes
    :param integer:
    :return: bytes
    """
    return integer.to_bytes((integer.bit_length() + 7) // 8, 'big')
    # alternatively
    # return bytes.fromhex(hex(integer)[2:])


def bytes_to_int(bytes_arr):
    """
    Convert bytes to an int
    :param bytes_arr:
    :return: int
    """
    return int.from_bytes(bytes_arr, byteorder='big', signed=False)


def bytes_to_bin(bytes_arr):
    """
    Convert bytes to a binary number
    :param bytes_arr:
    :return: str, whose length must be a multiple of 8
    """
    res = bin(bytes_to_int(bytes_arr))[2:]
    return bin_compensate(res)


def int_to_binary_representation(integer):
    """
    integer = 2^e1 + 2^e2 + ... + 2^ek， e1 > ... > ek
    :param integer: int
    :return: [e1, e2, ..., ek]
    """
    bin_str = bin(integer)[2:]
    bin_len = len(bin_str)

    exponent_list = []
    for i in range(bin_len):
        if bin_str[i] == '1':
            exponent_list.append(bin_len - i - 1)

    return exponent_list


def str_to_bin(str_arr):
    """
    Convert a string to a binary number in string
    :param str_arr: str
    :return: str
    """
    res = ''
    for st in str_arr:
        char = bin(ord(st))[2:]
        res += bin_compensate(char)
    return res


def bin_to_str(bin_str_arr):
    """
    Convert binary number in string to string
    :param bin_str_arr: str, whose length must be a multiple of 8
    :return: str
    """
    res = ''
    for i in range(0, len(bin_str_arr), 8):
        res += chr(int(bin_str_arr[i:i + 8], 2))
    return res


def bin_compensate(bin_arr):
    """
    Compensate a binary number in string with zero till its length being a multiple of 8
    :param bin_arr: str
    :return: str
    """
    return '0' * (8 - len(bin_arr) % 8) + bin_arr


def str_to_int(str_arr):
    """

    :param str_arr: str
    :return: int
    """
    return int(str_to_bin(str_arr), 2)


def int_to_str(integer):
    """

    :param integer: int
    :return: str
    """
    return bin_to_str(bin_compensate(bin(integer)[2:]))


def str_to_bytes(str_arr):
    """
    'hello' -> b'hello'
    :param str_arr: str
    :return: bytes
    """
    return bytes(str_arr, 'utf-8')


def bytes_to_str(byte_arr):
    """
    b'hello' -> 'hello'
    :param byte_arr: bytes
    :return: str
    """
    return str(byte_arr, 'utf-8')
    # return str(byte_arr, 'utf-8')
