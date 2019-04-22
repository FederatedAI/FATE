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

from arch.api import federation
import numpy as np
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HomoFederatedAggregator:
    # 把多个model聚合在一起
    def aggregate_model(self, transfer_variable, iter_num,
                        party_weights, host_encrypter):
        # Step 1: Send 自己model到所有的host

        model_transfer_id = transfer_variable.generate_transferid(transfer_variable.guest_model,
                                                                  iter_num)
        guest_model = federation.get(name=transfer_variable.guest_model.name,
                                     tag=model_transfer_id,
                                     idx=0)

        guest_model = np.array(guest_model)
        LOGGER.info("received guest model")
        host_model_transfer_id = transfer_variable.generate_transferid(transfer_variable.host_model,
                                                                       iter_num)
        host_models = federation.get(name=transfer_variable.host_model.name,
                                     tag=host_model_transfer_id,
                                     idx=-1)
        LOGGER.info("recevied host model")
        final_model = guest_model * party_weights[0]

        for idx, host_model in enumerate(host_models):
            encrypter = host_encrypter[idx]
            host_model = encrypter.decrypt_list(host_model)
            host_model = np.array(host_model)
            final_model = final_model + party_weights[idx + 1] * host_model
        # LOGGER.debug("Finish aggregate model, final model shape: {}".format(
        #     np.shape(final_model)))
        return final_model

    def aggregate_loss(self, transfer_variable, iter_num, party_weights, host_use_encryption):
        guest_loss_id = transfer_variable.generate_transferid(transfer_variable.guest_loss, iter_num)
        guest_loss = federation.get(name=transfer_variable.guest_loss.name,
                                    tag=guest_loss_id,
                                    idx=0)
        LOGGER.info("Received guest loss")
        # LOGGER.debug("guest_loss: {}".format(guest_loss))

        host_loss_id = transfer_variable.generate_transferid(transfer_variable.host_loss, iter_num)
        loss_party_weight = party_weights.copy()

        total_loss = loss_party_weight[0] * guest_loss
        for idx, use_encryption in enumerate(host_use_encryption):
            if use_encryption:
                loss_party_weight[idx] = 0
                continue
            host_loss = federation.get(name=transfer_variable.host_loss.name,
                                       tag=host_loss_id,
                                       idx=idx)
            LOGGER.info("Received loss from {}th host".format(idx))
            total_loss += loss_party_weight[idx] * host_loss

        total_loss /= sum(loss_party_weight)
        return total_loss

    @staticmethod
    def aggregate_grad_loss(acc, x):
        if acc is None and x is None:
            return None
        if acc is None:
            return x
        if x is None:
            return acc
        return acc[0] + x[0], acc[1] + x[1]

    @staticmethod
    def aggregate_grad(acc, x):
        if acc is None and x is None:
            return None
        if acc is None:
            return x
        if x is None:
            return acc
        return acc[0] + x[0], None
