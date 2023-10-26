#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from hashlib import sha256
from typing import List

from ... import communicator as comm

"""
    I dont think random modular is secure enough, but we can live with it for testing purpose

"""


class BaseOT:
    """
    hardcoded public parameter
    log2(__prime) > 128

    __generator is a primitive root of __prime
    """

    __prime = 631276824160446938136046282957027762913
    __generator = 3
    __inverse__generator = pow(__generator, (__prime - 2), __prime)

    @staticmethod
    def string_xor(s1, s2):
        """
        XOR of two strings
        """
        return "".join(chr(ord(a) ^ ord(b)) for a, b in zip(s1, s2))

    def __init__(self, partner_rank):
        self.partner_rank = partner_rank
        return

    def send(self, message0s: List[str], message1s: List[str]):
        """
        sender's input is two message lists
        """
        if len(message0s) != len(message1s):
            raise ("inconsistent input size!")

        alphas = []
        masks_for_message1s = []
        for _i in range(len(message1s)):
            # pick a random element from Z_p
            alpha = random.randint(0, self.__prime - 1)
            alphas.append(alpha)

            # g^\alpha
            mask_for_message1 = pow(self.__generator, alpha, self.__prime)
            masks_for_message1s.append(mask_for_message1)

        # send mask_for_message1
        for i in range(len(message1s)):
            comm.get().send_obj(masks_for_message1s[i], self.partner_rank)

        # compute (g^\alpha)^-\alpha when waiting for response
        # (g^-1)^(\alpha^2) = (g^-1)^(\alpha^2 mod (p-1))
        dividers = []
        for i in range(len(message1s)):
            divider = pow(
                self.__inverse__generator,
                alphas[i] * alphas[i] % (self.__prime - 1),
                self.__prime,
            )
            dividers.append(divider)

        masks_for_choices = []

        # recv mask_for_choice
        for _i in range(len(message1s)):
            mask_for_choice = comm.get().recv_obj(self.partner_rank)
            masks_for_choices.append(mask_for_choice)

        for i in range(len(message1s)):
            masks_for_choices[i] = pow(masks_for_choices[i], alphas[i], self.__prime)

            # hash
            pad0 = sha256(str(masks_for_choices[i]).encode("utf-8")).hexdigest()
            pad1 = sha256(str(masks_for_choices[i] * dividers[i] % self.__prime).encode("utf-8")).hexdigest()

            if len(pad0) < len(message0s[i]):
                raise (str(i) + "-th message0 is too long")
            if len(pad1) < len(message1s[i]):
                raise (str(i) + "-th message1 is too long")
            # encrypt with one time pad
            message0_enc = self.string_xor(pad0, message0s[i])
            message1_enc = self.string_xor(pad1, message1s[i])

            # send message0, message1
            comm.get().send_obj(message0_enc, self.partner_rank)
            comm.get().send_obj(message1_enc, self.partner_rank)

    def receive(self, choices: List[bool]):
        """
        choice:
            false: pick message0
            true: pick message1
        """

        betas = []
        masks_for_choices = []
        for _i in range(len(choices)):
            # pick a random element from Z_p
            beta = random.randint(0, self.__prime - 1)
            mask_for_choice = pow(self.__generator, beta, self.__prime)
            betas.append(beta)
            masks_for_choices.append(mask_for_choice)

        masks_for_message1s = []
        for i in range(len(choices)):
            # recv mask_for_message1
            mask_for_message1 = comm.get().recv_obj(self.partner_rank)
            masks_for_message1s.append(mask_for_message1)
            if choices[i]:
                masks_for_choices[i] = (masks_for_choices[i] * mask_for_message1) % self.__prime

        for i in range(len(choices)):
            # send mask_for_choice
            comm.get().send_obj(masks_for_choices[i], self.partner_rank)

        keys = []
        for i in range(len(choices)):
            # compute the hash when waiting for response
            key = sha256(str(pow(masks_for_message1s[i], betas[i], self.__prime)).encode("utf-8")).hexdigest()
            keys.append(key)

        rst = []

        for i in range(len(choices)):
            # recv message0, message1
            message0_enc = comm.get().recv_obj(self.partner_rank)
            message1_enc = comm.get().recv_obj(self.partner_rank)

            if choices[i]:
                rst.append(self.string_xor(keys[i], message1_enc))
            else:
                rst.append(self.string_xor(keys[i], message0_enc))
        return rst
