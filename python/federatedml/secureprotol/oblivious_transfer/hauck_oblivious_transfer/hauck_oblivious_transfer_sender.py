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

from federatedml.secureprotol.oblivious_transfer.base_oblivious_transfer import ObliviousTransferKey
from federatedml.secureprotol.oblivious_transfer.hauck_oblivious_transfer.hauck_oblivious_transfer import \
    HauckObliviousTransfer
from federatedml.util import consts, LOGGER


class HauckObliviousTransferSender(HauckObliviousTransfer):
    """
    Hauck-OT for the sender (host)
    """

    def __init__(self):
        super(HauckObliviousTransferSender, self).__init__()

    def key_derivation(self, target_num):
        """
        Derive a list of keys for encryption and transmission
        :param target_num: N in k-N OT
        :return: List[ObliviousTransferKey]
        """
        LOGGER.info("enter sender key derivation phase for target num = {}".format(target_num))
        # 1. Choose a random scalar (y) from Z^q, calculate S and T to verify its legality
        y, s, t = self._gen_legal_y_s_t()

        # 2. Send S to the receiver, if it is illegal addressed by the receiver, regenerate y, S, T
        attempt_count = 0
        while True:
            self.transfer_variable.s.remote(s,
                                            suffix=(attempt_count,),
                                            role=consts.GUEST,
                                            idx=0)
            # federation.remote(obj=s,
            #                   name=self.transfer_variable.s.name,
            #                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.s, attempt_count),
            #                   role=consts.GUEST,
            #                   idx=0)
            LOGGER.info("sent S to guest for the {}-th time".format(attempt_count))
            s_legal = self.transfer_variable.s_legal.get(idx=0,
                                                         suffix=(attempt_count,))
            # s_legal = federation.get(name=self.transfer_variable.s_legal.name,
            #                          tag=self.transfer_variable.generate_transferid(self.transfer_variable.s_legal,
            #                                                                         attempt_count),
            #                          idx=0)
            if s_legal:
                LOGGER.info("receiver confirms the legality of S at {} attempt, will proceed".format(attempt_count))
                break
            else:
                LOGGER.info("receiver rejects this S at {} attempt, will regenerate S".format(attempt_count))
                y, s, t = self._gen_legal_y_s_t()
                attempt_count += 1

        # 3. Wait for the receiver to hash S to get T
        LOGGER.info("waiting for the receiver to hash S to get T")

        # 4. Get R = cT + xG from the receiver, also init the MAC
        r = self.transfer_variable.r.get(idx=0)
        # r = federation.get(name=self.transfer_variable.r.name,
        #                    tag=self.transfer_variable.generate_transferid(self.transfer_variable.r),
        #                    idx=0)
        LOGGER.info("got from guest R = " + r.output())
        self._init_mac(s, r)

        # 5. MAC and output the key list
        key_list = []
        yt = self.tec_arithmetic.mul(scalar=y, a=t)    # yT
        yr = self.tec_arithmetic.mul(scalar=y, a=r)    # yR
        for i in range(target_num):
            iyt = self.tec_arithmetic.mul(scalar=i, a=yt)    # iyT
            diff = self.tec_arithmetic.sub(a=yr, b=iyt)    # yR - iyT
            key = self._mac_tec_element(diff)
            # LOGGER.info("{}-th key generated".format(i))
            # LOGGER.info("key before MAC = " + diff.output())
            # LOGGER.info("key = {}".format(key))
            key_list.append(ObliviousTransferKey(i, key))

        LOGGER.info("all keys successfully generated")

        return key_list

    def _gen_legal_y_s_t(self):
        while True:
            y = self._gen_random_scalar()
            s = self.tec_arithmetic.mul(scalar=y, a=self.tec_arithmetic.get_generator())  # S = yG
            t = self._hash_tec_element(s)
            if self.tec_arithmetic.is_in_group(s) and not isinstance(t, int):
                # Both S and T are legal
                LOGGER.info("randomly generated y, S, T")
                return y, s, t
