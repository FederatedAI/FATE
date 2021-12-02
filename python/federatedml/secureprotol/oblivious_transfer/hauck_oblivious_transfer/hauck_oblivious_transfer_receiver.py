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

from federatedml.secureprotol.number_theory.group.twisted_edwards_curve_group import TwistedEdwardsCurveElement
from federatedml.secureprotol.oblivious_transfer.base_oblivious_transfer import ObliviousTransferKey
from federatedml.secureprotol.oblivious_transfer.hauck_oblivious_transfer.hauck_oblivious_transfer import \
    HauckObliviousTransfer
from federatedml.util import consts, LOGGER


class HauckObliviousTransferReceiver(HauckObliviousTransfer):
    """
    Hauck-OT for the receiver (guest)
    """

    def __init__(self):
        super(HauckObliviousTransferReceiver, self).__init__()

    def key_derivation(self, target):
        """
        Generate a key the corresponds to target
        :param target: k int >= 0 in k-N OT
        :return: ObliviousTransferKey
        """
        LOGGER.info("enter receiver key derivation phase for target = {}".format(target))
        # 1. Choose a random scalar from Z^q
        x = self._gen_random_scalar()   # x
        LOGGER.info("randomly generated scalar x")

        # 2. Get S = yG from the sender and check its legality
        attempt_count = 0
        while True:
            s = self.transfer_variable.s.get(idx=0,
                                             suffix=(attempt_count,))
            # s = federation.get(name=self.transfer_variable.s.name,
            #                    tag=self.transfer_variable.generate_transferid(self.transfer_variable.s, attempt_count),
            #                    idx=0)
            LOGGER.info("got from host S = " + s.output())
            s_legal, t = self._check_s_t_legal(s)
            self.transfer_variable.s_legal.remote(s_legal,
                                                  suffix=(attempt_count,),
                                                  role=consts.HOST,
                                                  idx=0)
            # federation.remote(obj=s_legal,
            #                   name=self.transfer_variable.s_legal.name,
            #                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.s_legal,
            #                                                                  attempt_count),
            #                   role=consts.HOST,
            #                   idx=0)
            if s_legal:
                LOGGER.info("S is legal at {} attempt".format(attempt_count))
                break
            else:
                LOGGER.info("S is illegal at {} attempt, will retry to get a legal S".format(attempt_count))
                attempt_count += 1

        # 3. Slack
        LOGGER.info("S is hashed to get T = " + t.output())

        # 4. Compute and send to the sender R = cT + xG, also init the MAC
        c = target
        ct = self.tec_arithmetic.mul(scalar=c, a=t)    # cT
        xg = self.tec_arithmetic.mul(scalar=x, a=self.tec_arithmetic.get_generator())     # xG
        r = self.tec_arithmetic.add(a=ct, b=xg)    # R = cT + xG
        self.transfer_variable.r.remote(r,
                                        role=consts.HOST,
                                        idx=0)
        # federation.remote(obj=r,
        #                   name=self.transfer_variable.r.name,
        #                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.r),
        #                   role=consts.HOST,
        #                   idx=0)
        LOGGER.info("sent to host R = " + r.output())
        self._init_mac(s, r)

        # 5. MAC and output the correct key
        xs = self.tec_arithmetic.mul(scalar=x, a=s)
        # LOGGER.info("target index = " + str(target))
        # LOGGER.info("target key before MAC = " + xs.output())
        target_key = self._mac_tec_element(xs)
        # LOGGER.info("target key = {}".format(target_key))

        return ObliviousTransferKey(target, target_key)

    def _check_s_t_legal(self, s):
        """
        Check if s is in the TEC group and t is valid
        :param s: TwistedEdwardsCurveElement
        :return:
        """
        t = self._hash_tec_element(s)
        return self.tec_arithmetic.is_in_group(s) and isinstance(t, TwistedEdwardsCurveElement), t
